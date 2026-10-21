"""通用函数装饰器：自动记录函数进入 / 退出日志；瞬时故障重试。"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langgraph.errors import GraphInterrupt

from app.core.logger import logger

_F = TypeVar("_F", bound=Callable[..., Any])


def log_node(func: Callable) -> Callable:
    """在函数执行前后打印日志，同步/异步均支持。"""

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
        except GraphInterrupt:
            raise
        except Exception:
            logger.error(f"━━━ {node_name} 异常 ━━━  耗时 {time.perf_counter() - t0:.3f}s")
            raise
        finally:
            logger.info(f"━━━ 退出 {node_name} ━━━  耗时 {time.perf_counter() - t0:.3f}s")

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def retry(
    attempts: int = 3,
    base_delay: float = 0.4,
    max_delay: float = 3.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    name: str | None = None,
) -> Callable[[_F], _F]:
    """对 ``exceptions`` 中的瞬时故障做指数退避重试，同步/异步均支持。"""
    def decorate(func: _F) -> _F:
        label = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= attempts:
                        logger.error(f"{label} 重试 {attempts} 次仍失败: {exc}")
                        raise
                    logger.warning(f"{label} 第 {attempt} 次失败({type(exc).__name__})，{delay:.2f}s 后重试: {exc}")
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise AssertionError("unreachable")  # pragma: no cover

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= attempts:
                        logger.error(f"{label} 重试 {attempts} 次仍失败: {exc}")
                        raise
                    logger.warning(f"{label} 第 {attempt} 次失败({type(exc).__name__})，{delay:.2f}s 后重试: {exc}")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise AssertionError("unreachable")  # pragma: no cover

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorate
