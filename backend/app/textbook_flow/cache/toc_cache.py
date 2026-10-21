"""目录树进程内缓存：按 task_id 缓存展示载荷，miss 时从磁盘懒重建，终态删除。"""

from pathlib import Path

from cachetools import TTLCache

from app.textbook_flow.nodes.toc_tree import build_toc_payload

_TOC_CACHE_TTL = 2 * 3600  # 摄入任务小时级，2 小时足够覆盖断线重连窗口

# task_id -> [{textbook, chapters 三层}]，与 SSE toc 事件载荷同构
_toc_cache: TTLCache[str, list[dict]] = TTLCache(maxsize=64, ttl=_TOC_CACHE_TTL)


def get_tocs(task_id: str, textbook_path: str | Path) -> list[dict]:
    """取该任务全部教材的目录树载荷：命中缓存直接返回，miss 时从磁盘懒重建。

    目录尚未解析（mineru_toc 缺失）时返回空列表且不落缓存，
    避免把「还没有树」这一中间态固化下来。
    """
    cached = _toc_cache.get(task_id)
    if cached is not None:
        return cached
    tocs = build_toc_payload(Path(textbook_path))
    if tocs:
        _toc_cache[task_id] = tocs
    return tocs


def pop_tocs(task_id: str) -> None:
    """终态（done/error）删除该任务的目录树缓存，完成即删。"""
    _toc_cache.pop(task_id, None)
