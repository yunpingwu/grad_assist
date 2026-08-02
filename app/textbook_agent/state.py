from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class TextBookState(TypedDict):
    """教材摄入流水线（load_textbook → spilt_contents → split → parse_to_md → split_text_and_store）的状态。

    仅保留摄入所需字段；检索/对话相关字段已随检索图（retrieval graph）迁移。
    """

    # ===== 入口路由 =====
    # 是否已有教材在向量库中，由上层编排决定走摄入流程还是检索流程
    textbook_exists: bool
    # 教材文件路径（本地路径或 URL）
    textbook_path: NotRequired[str]
    # 教材名称（摄入目录标识，检索时按此过滤）
    textbook_name: NotRequired[str]

    # ===== 任务追踪 =====
    # 任务 ID（由上层编排创建，节点内通过 task_util.update_task 上报进度，供 SSE 推送）
    task_id: NotRequired[str]

    # ===== 摄入中间产物 =====
    # 章节分割后的子 PDF 路径列表
    sub_pdf_paths: NotRequired[list[str]]
    # 目录页 mineru 解析后解压的目录路径列表
    extracted_contents_dirs: NotRequired[list[str]]
    # 章节 mineru 解析后解压的目录路径列表
    extracted_dirs: NotRequired[list[str]]
    # 向量化入库是否完成
    ingestion_done: NotRequired[bool]
