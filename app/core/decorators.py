"""通用函数装饰器：自动记录函数进入 / 退出日志。"""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable

from app.core.logger import logger


def log_node(func: Callable) -> Callable:
    """装饰器：在函数执行前后打印日志。

    用法::

        @log_node
        def my_func(state: TextBookState) -> dict:
            ...

    支持同步和异步函数。
    """

    node_name = func.__name__

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"━━━ 进入 {node_name} ━━━")
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.error(f"━━━ {node_name} 异常 ━━━  耗时 {time.perf_counter() - t0:.3f}s")
            raise
        finally:
            logger.info(f"━━━ 退出 {node_name} ━━━  耗时 {time.perf_counter() - t0:.3f}s")

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"━━━ 进入 {node_name} ━━━")
        t0 = time.perf_counter()
        try:
            return await func(*args, **kwargs)
        except Exception:
            logger.error(f"━━━ {node_name} 异常 ━━━  耗时 {time.perf_counter() - t0:.3f}s")
            raise
        finally:
            logger.info(f"━━━ 退出 {node_name} ━━━  耗时 {time.perf_counter() - t0:.3f}s")

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
