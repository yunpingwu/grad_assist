# 教材图片简介方案(Image Caption)设计文档

> 状态:设计中(未实现) · 日期:2026-08-09
> 目标:让教材图片在问答回答中被真正利用——解析时用视觉模型为每张图生成简介,回答时纯文本模型依据简介自主决定是否在回答中插入图片引用。

---

## 1. 背景与根因

现状链路:

```
MinerU 解析 → full.md + images/ → 切块时图片上传 MinIO → metadata_json.images = {path, object_name, url}
回答时 generate_answer: LLM context 只有纯文本片段,无任何图片信息
```

**根因**:纯文本 LLM 在回答时完全"看不见"图片信息(简介、url),因此永远不会输出 `![]()` 图片引用,前端已支持的图片渲染能力形同虚设。`collect_image_options` 收集的候选图 `description` 只是文件路径,且从不进 LLM context。

**方案核心**:离线(解析时)为每张图生成 50-100 字简介 → 简介落入正文 `text`(参与向量检索)+ 随片段 metadata 携带 url → 在线回答时纯文本 LLM 看到"简介 + 引用格式",自主决定插入 `![图注](url)`,前端渲染。

---

## 2. 方案总览

```
┌─ 离线（教材解析时,一次性成本）─────────────────────────────────────┐
│  full.md ──提取图片──▶ 本地图片(base64) + 所在小节上下文            │
│        │                        │                                  │
│        │             视觉模型(Qwen-VL)生成 50-100 字简介           │
│        │                        │                                  │
│        ▼                        ▼                                  │
│  full_captioned.md(副本) ◀── 图片行 alt 替换为简介                  │
│        │                                                           │
│        ▼                                                           │
│  切块: content 尾部拼「【本节图片】简介…」→ 入库 Milvus text 字段   │
│        metadata_json.images 补 description(简介)+ url               │
└────────────────────────────────────────────────────────────────────┘

┌─ 在线（回答时,零增量成本）─────────────────────────────────────────┐
│  召回片段 → context = 正文 + 「图片候选: 简介 ｜ 引用: ![图注](url)」│
│        ▼                                                           │
│  纯文本 LLM 判断相关性 → 输出 ![图注](url) → 前端 splitAnswer 渲染   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 详细设计

### 3.1 离线:视觉简介生成(新增节点 `caption_images`)

**位置**:Graph 流水线 `parse_to_md` 之后、`split_text_and_store` 之前:

```
load_textbook → split_contents → split → parse_to_md → caption_images(新) → split_text_and_store
```

**输入**:每章的 `full.md` + 同目录 `images/`(本地文件)。

**输出**:每章生成 `full_captioned.md`(**full.md 的副本**,不改 MinerU 原产物)。

**副本中图片行的改写**:

```markdown
<!-- 原 full.md -->
![](images/1.jpg)

<!-- full_captioned.md -->
![二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序](images/1.jpg)
```

- 相对路径 `images/1.jpg` **保持不变**,切块时 `_IMAGE_PATTERN` 提取逻辑与 `upload_and_map` 完全兼容;
- alt 文本 = 视觉模型生成的简介(50-100 字),`images.description` 直接取自 alt,无需 sidecar;
- 视觉调用失败/降级的图,alt 用文件名兜底(如 `images/1.jpg`)。

**视觉模型调用**:

| 项 | 设计 |
|---|---|
| 模型 | 新增 `get_visual_llm_client()`,用 `.env` 的 `VISUAL_MODEL=qwen3.7-flash` + 阿里云 `compatible-mode` base_url(复用 `init_chat_model` 模式) |
| 传图方式 | **图片本地文件转 base64**(`data:image/...;base64,...`)。理由:`MINIO_ENDPOINT=159.75.220.155:9000` 为裸 IP,阿里云侧未必能回访,base64 不依赖可达性 |
| 上下文 | prompt 同时注入该图片所在小节(去图后的前 200-300 字正文),帮助模型理解图片在教材中的角色 |
| 简介要求 | prompt 限定:中文、50-100 字、说明图的内容与在教材中的作用 |
| 并发/重试 | 每章内 2-4 张并行;单图失败重试 1 次;仍失败则 alt 兜底文件名,**不阻断入库**(旁路降级风格) |

**幂等**:`full_captioned.md` 已存在 → 整个节点跳过(对应"mineru_split 已存在则复用"的既有幂等策略);已存在的副本不会重复调用视觉模型、不重复计费。

**task 进度**:节点内 `update_task` 更新一次进度(0.8 → 0.9 区间),与现有节点风格一致。

### 3.2 切块与入库:`split_text_and_store` 改动

`chunk_textbook` 读取优先级:**`full_captioned.md` 存在则读它,否则读 `full.md`**(向后兼容旧教材)。

对每个 section(现状逻辑上扩展):

1. `_extract_images_and_code` 提取图片:alt 已是简介;
2. `upload_and_map` 上传并取得 url(现状不变);
3. **content 尾部拼接简介文本**(满足"简介放在正文 text 中"):

   ```
   # 第3章 栈 > ## 栈的基本操作
   <正文…>

   【本节图片】
   1. 二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序
   ```

   实现:对 `images` 列表按 `[图{i}] {description}` 拼接,追加到 content;
4. `images` 项补 `description`(来自 alt):

   ```json
   {"path": "images/1.jpg", "object_name": "textbook/…/images/1.jpg", "url": "http://159.75.220.155:9000/bucket/…", "description": "二叉树中序遍历的示意图:…"}
   ```

5. `embed_and_store` **无需改动**:content 已含简介 → Milvus `text` 字段自然携带简介参与 embedding(用户要求);`metadata_json` 序列化整个 images dict,自动带上 description。

> 说明:简介进入 `text` 后,正文向量中混入 50-100 字/图 的说明文本。对 500-2000 字的 chunk,占比小、稀释可控;收益是"问某张图的内容"也能被检索召回。

### 3.3 在线回答:context 注入 + prompt 规则

**`ask_llm` context 组装**(`generate_answer.py:72-78` 扩展):

```text
[片段1] # 第3章 栈 > ## 栈的基本操作
正文…

[片段1图片候选]
- 简介:二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序
  引用: ![二叉树中序遍历的示意图](http://159.75.220.155:9000/bucket/textbook/…/1.jpg)
```

- 数据来源:片段 `entity.metadata_json.images`(已有 `description` + `url`);
- **url 只在回答 context 注入、不进 content 正文**:避免长 URL 污染 embedding 向量;LLM 决策时仍能看到完整可引用的 markdown。

**`generate_answer.prompt` 追加规则**:

```
- 若某张图的简介与回答内容直接相关、能辅助理解,在回答中合适位置插入图片引用 ![图注](url);
- 图注用该图的简介;不相关的图不要引用;严禁编造不存在的 url。
```

**前端:零改动**。`splitAnswer` 已支持 `![]()` → `<img>`;历史回显 `collect_image_options` 的 description 自动升级为真实简介。

### 3.4 兼容性与降级

| 场景 | 行为 |
|---|---|
| 旧教材(无 `full_captioned.md`) | 读 `full.md`,图片无简介,`description` 用文件名兜底,行为与现状一致 |
| 视觉调用失败 | alt=文件名兜底,不阻断入库 |
| 回答时无图片候选 | prompt 规则保证 LLM 正常纯文本回答 |

---

## 4. 数据示例(全链路)

**full_captioned.md**:

```markdown
# 第3章 栈

## 栈的基本操作

栈是一种后进先出的数据结构。

![二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序](images/1.jpg)
```

**入库 chunk(content,即 Milvus text 字段)**:

```text
# 第3章 栈 > ## 栈的基本操作
栈是一种后进先出的数据结构。

【本节图片】
1. 二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序
```

**metadata_json**:

```json
{"images": [{"path": "images/1.jpg", "object_name": "textbook/C语言程序设计/第3章 栈/images/1.jpg", "url": "http://159.75.220.155:9000/grad-assist/textbook/…", "description": "二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序"}]}
```

**LLM 回答(可能输出)**:

```markdown
二叉树的中序遍历顺序为左子树→根→右子树,访问过程如下图所示:

![二叉树中序遍历的示意图:根节点居中,左子树在左、右子树在右,箭头标注访问顺序](http://159.75.220.155:9000/grad-assist/textbook/…/1.jpg)
```

---

## 5. 改动文件清单

**新增**:

| 文件 | 内容 |
|---|---|
| `app/textbook_agent/nodes/caption_images.py` | 新节点:读 full.md → 并行调视觉模型 → 写 full_captioned.md(含幂等/降级/进度) |
| `app/utils/caption_util.py` | base64 编码、视觉模型调用封装、简介 prompt |

**修改**:

| 文件 | 改动 |
|---|---|
| `app/clients/llm.py` | 新增 `get_visual_llm_client()`(VISUAL_MODEL) |
| `app/textbook_agent/graph.py` | 插入 `caption_images` 节点 |
| `app/textbook_agent/nodes/split_text_and_store.py` | `chunk_textbook` 读副本优先;content 拼「【本节图片】」;images 补 description |
| `app/query_agent/nodes/generate_answer.py` | context 注入「图片候选」段 |
| `app/prompts/generate_answer.prompt` | 加图片引用规则 |

**前端**:无改动。

---

## 6. 成本与风险

**成本**:
- 离线:图片数 × 1 次 Qwen-VL-flash 调用(单图几十 token 输出;flash 档低价);`full_captioned.md` 幂等保证重跑不重复计费;
- 在线:零增量(纯文本模型,无视觉调用)。

**风险与对策**:
- 视觉简介偶尔不准确 → 简介仅作辅助决策,LLM 仍以正文为准,可不引用;
- 简介混入 text 稀释正文向量 → 占比小(50-100 字 vs 500-2000 字 chunk),若实测影响大可改为独立图片块入库;
- 阿里云侧无法访问 MinIO url → 已用 base64 传图规避。

---

## 7. 待实现时的验证计划

1. 单章实测:跑 `caption_images` → 检查 `full_captioned.md` 图片行 alt;
2. 切块入库:检查 chunk content 含「【本节图片】」、`metadata_json.images.description` 非空;
3. 问答实测:问"那张二叉树示意图讲了什么" → 召回含简介的片段;问与图相关的问题 → 回答中出现 `![…](url)` 且前端渲染正常;
4. 幂等:重跑解析,确认不重复调用视觉模型;
5. 降级:临时断开视觉模型,确认入库不中断。
