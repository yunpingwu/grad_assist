# 图片元数据与正文图标记错位问题

> 状态:待修复 · 日期:2026-09-03
> 目标:消除「超长小节硬切后,图片 url 元数据只挂首个子块,而正文图标记可能落在其他子块」导致的图片引用丢失。

***

## 1. 背景与问题

图片回绑机制依赖**同 chunk 内**两组信息同时存在:

- 正文内嵌的图标记 `【图: 简介】`(切块时由图片行原位替换而来,参与向量检索);

- `metadata_json.images = [{url, description}]`(供召回端按简介把 url 回绑成可引用图片,见 `app/study_agent/tools/search.py` 的 `_collect_hit_images` / `_FIGURE_MARK_PATTERN`)。

**根因**:[split\_text\_and\_store.py](app/textbook_flow/nodes/split_text_and_store.py) 的 `chunk_textbook` 在超长小节按字符硬切(`_char_split`,chunk\_size=2000, overlap=200)时,`images` 与 `codes` **只挂** **`idx == 0`** **的子块**(见 `chunk_textbook` 中 `"images": images if idx == 0 else []`);而 `【图: 简介】` 文本由 `_char_split` 硬切,可能落在 `idx > 0` 的子块。

## 2. 错位场景与后果

```
超长小节(2400 字符, 含 1 张图, 图标记落在第 2 个子块)
├─ 子块0 (idx=0): images=[{url, desc}]   ← 有 url, 但正文无对应图标记
├─ 子块1 (idx=1): images=[]              ← 正文有「【图: 简介】」, 但无 url ✗
└─ 子块2 (idx=2): images=[]
```

| 被召回的子块            | 正文          | metadata\_json.images | 后果                    |
| ----------------- | ----------- | --------------------- | --------------------- |
| `idx == 0`(持 url) | 无对应图标记      | 含 url                 | 回绑无目标,url 浪费          |
| `idx > 0`(持图标记)   | 含 `【图: 简介】` | **空**                 | **url 丢失,LLM 无法引用该图** |

召回端 `_collect_hit_images` 返回空映射 → `_FIGURE_MARK_PATTERN` 无 url 可绑 → 图标记原样保留,图片引用能力退化为"只有简介、没有图"。

### 现状缓解因素(覆盖不全)

- `_merge_small_chunks` 合并时 `content` 与 `images` 同链拼接,恰好把"图标记文本"与"url 元数据"重新聚到同一 chunk——但仅在存在 < 500 字符碎片时才触发,无法覆盖全部错位;

- `_char_split` 的 overlap=200 使图标记可能同时出现在相邻两个子块,但也可能完整落在单个 `idx > 0` 子块内。

## 3. 候选方案

### 方案 A:精确归属(推荐)

让每张图的 url 挂到「`【图: 简介】` 文本实际所在的子块」,图-文严格对齐。改动点均在 `split_text_and_store.py`:

1. `_extract_images_and_code`:替换完成后,用 `re.finditer(r"【图: .*?】", clean_text)` 重新定位每个图标记的**字符偏移**,按顺序回填 `img["offset"]`(与原 images 一一对应,因均为原位替换);
2. `_char_split`:新增 `_char_split_with_offsets(text) -> list[tuple[str, int]]` 返回每个子块的 `(text, start)`(现仅 `chunk_textbook` 一处调用,可安全替代);
3. `chunk_textbook` 硬切分支:按 `img["offset"] ∈ [start, start+len(text))` 将 images 归属到对应子块(替换 `images if idx == 0 else []`),保持升序。

要点与边界:

- **overlap=200 重叠区**:图标记落入两个子块重叠区时,归属**第一个**包含它的子块即可,避免跨 hit 重复回绑;

- **`_merge_small_chunks`** **零改动**:合并方向为后块并入前块,`content` 与 `images` 天然同链对齐;

- **旧教材(读** **`full.md`,alt=文件名兜底)**:不受影响,offset 定位对文件名简介同样成立。

优点:绑定精确、无冗余、召回端零改动。缺点:改 3 处,需要覆盖重叠区边界的用例。

### 方案 B:全量挂载(最简单,1 行)

硬切分支改为 `"images": images`(每个子块都挂该 section 的全部图片)。

优点:实现 1 行,无 offset 跟踪。缺点:图-文绑定粒度变粗——一个 section 多图分布多子块时,不含该图的子块也携带其 url,易造成"挂错图"噪声;`metadata_json` 冗余。

### 建议

选**方案 A**:召回端已实现"回绑正文图标记",绑定粒度精确与之配套最合适;B 粒度粗反而引入新噪声。

## 4. 同类错位:codes

`codes if idx == 0 else []` 存在同一模式错位,且后果更严重:**`idx > 0`** **子块内提取的代码块根本不会独立入库**(`embed_and_store` 只为 `codes` 字段生成 `block_type="code"` 记录),即代码内容丢失,既不可检索也带不出正文。

与图片不同,代码块在正文中被删除、无文本落点,无法用"正文图标记偏移"方式归属,需按**原文本**中代码块的 offset 归属到子块,改动面更大。**建议与图片分开评估、单独排期**。

## 5. 影响面与验证方式

- **召回端(`search.py`)零改动**:错位修复仅涉及存储端 `images` 挂载位置;

- **存量数据**:已入库的旧教材 metadata 不会自动修正,需重新摄入(摄入幂等基于 collection 存在判定,需先注销/重建)或接受存量错位;

- **验证步骤**:

  1. 单章构造超长小节(>2000 字符、含图且图标记位于后段),跑 `split_text_and_store`;
  2. 检查各子块 `metadata_json.images`:图标记所在子块应持对应 url;
  3. 集成测试:问与图相关的问题,召回该子块,断言输出含 `![简介](url)`;
  4. 回归:多图小节、overlap 重叠区、旧教材(无富化副本)三组用例。

