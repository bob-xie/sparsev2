import logging  # 导入日志模块，用于记录运行时信息

import hydra  # 导入Hydra配置管理库
from nuplan.planning.script.builders.logging_builder import build_logger  # 从nuPlan导入日志构建器
from omegaconf import DictConfig  # 导入OmegaConf的DictConfig类型，用于配置管理

# 导入Metric缓存核心函数
from navsim.planning.metric_caching.caching import cache_data
# 导入worker池构建器，用于并行处理
from navsim.planning.script.builders.worker_pool_builder import build_worker

# 创建日志记录器实例，记录该模块的日志
logger = logging.getLogger(__name__)

# ==================== 配置路径 ====================
CONFIG_PATH = "config/metric_caching"  # 配置文件所在目录
CONFIG_NAME = "default_metric_caching"  # 默认配置文件名


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Metric缓存的主入口函数
    
    :param cfg: Omegaconf配置字典，包含所有必要的配置参数
    """
    # 配置日志系统，根据cfg中的日志配置初始化日志输出
    build_logger(cfg)

    # 构建worker实例，支持多进程/分布式处理
    worker = build_worker(cfg)

    # 开始预计算并缓存所有Metric数据
    logger.info("Starting Metric Caching...")  # 记录开始日志
    
    # 检查是否使用分布式Ray模式
    if cfg.worker == "ray_distributed" and cfg.worker.use_distributed:
        # 分布式Ray模式不支持此任务，抛出异常
        raise AssertionError("ray in distributed mode will not work with this job")
    
    # 执行缓存数据函数，传入配置和worker
    cache_data(cfg=cfg, worker=worker)


if __name__ == "__main__":
    # 当脚本直接运行时调用main函数
    main()
