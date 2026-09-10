"""FastAPI 应用入口：负责应用实例、生命周期与路由装配。

教材摄入（textbook_service）与统一教材助手（study_service，含对话/资料生成与
检索问答）均以 APIRouter 挂载于此，共享同一端口与 CORS 配置，统一由本模块作为
uvicorn 启动入口。
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.study_service import router as study_router
from app.api.textbook_service import router as textbook_router
from app.core import logger


async def _warmup_local_models() -> None:
    """预热本地 embedding / reranker 单例，把冷启动延迟消化在启动阶段。

    两个模型均为懒加载单例，若不预热，首个检索/精排请求会背数秒~十几秒的权重
    加载与首次前向开销。预热失败仅告警降级，不阻断启动（回退到首个请求懒加载）。
    """
    # 延迟导入：仅预热阶段加载模型模块，保持启动装配轻量
    from app.utils.embedding_util import generate_embeddings
    from app.utils.reranker_util import compute_rerank_scores

    try:
        await asyncio.to_thread(generate_embeddings, ["__warmup__"])
        logger.info("BGE-M3 embedding 模型预热完成")
    except Exception as exc:
        logger.warning(f"embedding 模型预热失败，首个检索请求将懒加载: {exc}")

    try:
        await asyncio.to_thread(compute_rerank_scores, "__warmup__", ["__warmup__"])
        logger.info("BGE-Reranker 模型预热完成")
    except Exception as exc:
        logger.warning(f"reranker 模型预热失败，首个 deep 请求将懒加载: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预热本地模型，退出时优雅释放外部连接（均幂等）。"""
    await _warmup_local_models()
    yield

    from app.clients import milvus_client, minio_client, mongo_client

    milvus_client.disconnect_milvus()
    minio_client.disconnect_minio()
    mongo_client.disconnect_mongo()
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
app.include_router(textbook_router)
app.include_router(study_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
