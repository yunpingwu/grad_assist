# 样本（按检索配置分目录）

每个子目录是一个**检索配置**，里面的文件是该配置下的「答案 + 它那一轮的上下文」。

```
samples/
├── hyb_50/     # 非深度：混合召回（dense/sparse 各 0.5）取 top-5
└── rerank/     # 深度：混合召回 + HyDE 两路扩候选池 → RRF 融合 → 交叉编码精排
```

## 文件格式

```jsonc
{
  "schema": 1,
  "dataset": "os",              // 对应 datasets/os.json
  "collection": "tb_a7874c71f752b140",
  "config": "hyb_50",
  "top_k": 5,
  "generation": {               // 这批答案的来历
    "provider": "deepseek", "model": "deepseek-flash", "temperature": 0.1,
    "prompt": "answer.prompt@409463638ea5",   // 冻结 prompt 的内容哈希
    "rerank_fusion_alpha": null, "candidate_pool": null,   // 仅 rerank 配置有值
    "generated_at": "2026-10-16T10:41:40+08:00"
  },
  "samples": [
    { "id": "OS-20260918-001",
      "retrieved_chunk_ids": ["v1_tb_...", "...×5"],
      "response": "进程由三部分组成：…" }
  ]
}
```

## 为什么按配置分，而不是塞进题集

`response` 和 `retrieved_chunk_ids` 是**成对的**：Faithfulness 判的是「答案里每句话能不能在
**它那一轮的上下文**里找到依据」。rerank 的 top-5 和 hyb_50 的 top-5 实测有 24% 的内容不同
（50 题里没有一题完全不同），拿 A 配置的答案配 B 配置的上下文，分数没有意义。

题集（question / gold / reference）是 config 无关的稳定资产，两者生命周期不同，所以分开放。

## 口径

由 [`../rag.py`](../rag.py) 的 `retrieve()` + `answer()` 产出：固定配置、冻结 prompt
（`../prompts/answer.prompt`）、单轮无历史、喂模型时 `wrap_untrusted` 包边界。
**只存 chunk id 不存正文**——正文取回时按 id 查 Milvus，可顺带校验；存副本就是被删掉的
`gold_evidence` 那种做法。

重跑：`uv run python -m eval.rag.rag <配置> [数据集键…]`（已有样本默认跳过，`--force` 覆盖）。
