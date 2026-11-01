"""FastAPI 应用装配与启动入口（``python -m app.main``）；路由汇总见 ``app/api/router.py``。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core import logger


async def _warmup_local_models() -> None:
    """预热本地 embedding / reranker 单例，把冷启动延迟消化在启动阶段。

    两个模型均为懒加载单例，若不预热，首个检索/精排请求会背数秒~十几秒的权重
    加载与首次前向开销。预热失败仅告警降级，不阻断启动（回退到首个请求懒加载）。
    """
    # 延迟导入：仅预热阶段加载模型模块，保持启动装配轻量
    from app.ml.embedding import agenerate_embeddings
    from app.ml.reranker import arerank_scores

    try:
        await agenerate_embeddings(["__warmup__"])
        logger.info("BGE-M3 embedding 模型预热完成")
    except Exception as exc:
        logger.warning(f"embedding 模型预热失败，首个检索请求将懒加载: {exc}")

    try:
        await arerank_scores("__warmup__", ["__warmup__"])
        logger.info("BGE-Reranker 模型预热完成")
    except Exception as exc:
        logger.warning(f"reranker 模型预热失败，首个 deep 请求将懒加载: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：初始化 PostgreSQL checkpoint 与本地模型，退出时释放外部连接。"""
    from app.infra import postgres_client
    from app.services import study as study_service
    from app.services import textbook as textbook_service

    await postgres_client.open_pools()
    try:
        # AsyncPostgresSaver 构造时要求运行中的事件循环，图只能在 lifespan 内装配；
        # setup() 建表（幂等），必须在接收请求前完成。
        study_service.init_graph(postgres_client.get_study_pool())
        textbook_service.init_graph(postgres_client.get_textbook_pool())
        await study_service.setup_checkpointer()
        await textbook_service.setup_checkpointer()

        await _warmup_local_models()
        yield
    finally:
        from app.infra import milvus_client, minio_client

        milvus_client.disconnect_milvus()
        minio_client.disconnect_minio()
        await postgres_client.close_pools()
        logger.info("外部连接已优雅释放")


app = FastAPI(
    title="Textbook Agent",
    description="一个将教材向量化后存储入向量数据库的langgraph流程",
    version="0.1.0",
    lifespan=lifespan,
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 教材摄入与统一教材助手路由：共享同一 app、端口与 CORS
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
