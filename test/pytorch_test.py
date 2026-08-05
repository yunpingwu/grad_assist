# 测试本地pytorch是否可用（gpu）
import torch

from app.core import logger

if __name__ == '__main__':
    try:
        logger.info(f"pytorch: {torch.__version__}")
        logger.info(f"cuda: {torch.cuda.is_available()}")
        logger.info(f"cuda: {torch.cuda.get_device_name(0)}")
        logger.info(f"cuda: {torch.cuda.current_device()}")
    except Exception as e:
        logger.error(f"pytorch: {e}")