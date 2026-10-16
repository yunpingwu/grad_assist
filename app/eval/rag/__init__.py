"""ragas 版检索评测包（见 docs/ragas-retrieval-eval-design.md）。

本包是仓库里唯一 import ragas 的地方——**不要**让它出现在 app/api 或
app/study_agent 的导入路径上（服务进程模块级就建图连库，多一个 datasets/tiktoken
级别的重依赖会直接拖慢启动与 CI）。

datasets/ 下是随代码入库的评测集（v2 格式，见 docs/eval-dataset-schema.md）。
"""
