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
    """装饰器：对瞬时故障做指数退避重试（同步 / 异步均支持）。

    用法::

        @retry(attempts=3, exceptions=(ConnectionError, TimeoutError))
        def fetch(...): ...

    Args:
        attempts: 最多尝试次数（含首次，attempts=1 等价不重试）。
        base_delay: 首次重试前的等待秒数，之后按 2 倍指数增长。
        max_delay: 单次等待上限，防止退避过长。
        exceptions: 视为「可重试瞬时故障」的异常类型；不在其中的异常直接抛出。
        name: 日志中的函数别名，缺省用被装饰函数名。

    Returns:
        同签名的包装函数；重试耗尽后抛出最后一次异常。
    """
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
