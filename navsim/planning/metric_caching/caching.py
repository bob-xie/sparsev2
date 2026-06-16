import gc                                                           # 垃圾回收模块，用于显式触发内存回收
import logging                                                      # 日志模块，用于记录运行时信息
import os                                                           # 操作系统接口，用于获取环境变量
import uuid                                                         # UUID生成模块，用于生成唯一线程ID
from pathlib import Path                                             # 路径处理模块，用于文件路径操作
from typing import Any, Dict, List, Optional, Union                 # 类型注解，提高代码可读性和类型检查

from hydra.utils import instantiate                                  # Hydra工具函数，用于动态实例化对象
from nuplan.planning.training.experiments.cache_metadata_entry import (  # 导入缓存元数据相关类
    CacheMetadataEntry,                                             # 缓存元数据条目类，存储单个场景的缓存信息
    CacheResult,                                                    # 缓存结果类，存储成功/失败统计
    save_cache_metadata,                                            # 保存缓存元数据到CSV文件的函数
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool  # 多线程/分布式工作池接口
from nuplan.planning.utils.multithreading.worker_utils import worker_map  # 任务分发工具函数
from omegaconf import DictConfig                                    # OmegaConf配置字典类型

from navsim.common.dataclasses import Scene, SensorConfig            # 导入场景和传感器配置数据类
from navsim.common.dataloader import SceneFilter, SceneLoader        # 导入场景过滤器和加载器
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor  # 导入指标缓存处理器
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario  # 导入导航场景类

logger = logging.getLogger(__name__)                                # 创建当前模块的日志记录器


def cache_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """
    在多个工作进程中并行执行场景DB文件的缓存。
    :param args: 字典列表，每个字典包含以下项:
        "log_file": 日志文件名
        "tokens": 该日志文件对应的场景token列表
        "cfg": 配置字典
    :return: CacheResult列表，每个元素包含成功/失败统计
    """

    # 定义内部包装函数，帮助内存垃圾回收
    # 这样所有变量在函数结束后都会超出作用域，便于Python GC清理
    # 这对于处理大型数据集时节省内存是必要的
    def cache_scenarios_internal(args: List[Dict[str, Union[Path, DictConfig]]]) -> List[CacheResult]:
        """
        内部函数，在单个工作进程中处理分配的场景。
        :param args: 该工作进程需要处理的任务列表
        :return: 包含单个CacheResult的列表
        """
        
        def cache_single_scenario(
            scene_dict: Dict[str, Any], processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            """
            处理单个原始场景，计算并保存指标缓存。
            :param scene_dict: 场景帧字典列表，包含场景的所有帧数据
            :param processor: 指标缓存处理器
            :return: 缓存元数据条目（成功）或None（失败）
            """
            # 从帧字典列表构建Scene对象
            scene = Scene.from_scene_dict_list(
                scene_dict,                                          # 场景帧字典列表
                None,                                                # 传感器数据路径，此处不需要
                num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,  # 历史帧数
                num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,    # 未来帧数
                sensor_config=SensorConfig.build_no_sensors(),       # 不加载传感器数据
            )
            # 将Scene包装为NavSimScenario，用于指标计算
            scenario = NavSimScenario(
                scene,                                               # 场景对象
                map_root=os.environ["NUPLAN_MAPS_ROOT"],             # 地图数据根目录
                map_version="nuplan-maps-v1.0",                      # 地图版本
            )

            # 调用处理器计算并保存指标缓存
            return processor.compute_and_save_metric_cache(scenario)

        def cache_single_synthetic_scenario(
            scene_path: Path, processor: MetricCacheProcessor
        ) -> Optional[CacheMetadataEntry]:
            """
            处理单个合成场景，计算并保存指标缓存。
            :param scene_path: 合成场景文件路径
            :param processor: 指标缓存处理器
            :return: 缓存元数据条目（成功）或None（失败）
            """
            # 从磁盘加载合成场景
            scene = Scene.load_from_disk(scene_path, None, SensorConfig.build_no_sensors())
            # 将Scene包装为NavSimScenario
            scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")

            # 调用处理器计算并保存指标缓存
            return processor.compute_and_save_metric_cache(scenario)

        # 获取节点ID，用于分布式训练（默认为0）
        node_id = int(os.environ.get("NODE_RANK", 0))
        # 生成唯一线程ID，用于日志追踪
        thread_id = str(uuid.uuid4())

        # 从任务参数中提取日志文件名列表
        log_names = [a["log_file"] for a in args]
        # 从任务参数中提取所有场景token
        tokens = [t for a in args for t in a["tokens"]]
        # 获取配置字典（从第一个任务参数中提取）
        cfg: DictConfig = args[0]["cfg"]

        # 实例化场景过滤器
        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        # 设置过滤器只加载指定的日志文件
        scene_filter.log_names = log_names
        # 设置过滤器只加载指定的场景token
        scene_filter.tokens = tokens
        # 创建场景加载器，加载该工作进程负责的场景
        scene_loader = SceneLoader(
            synthetic_sensor_path=None,                              # 不加载合成传感器数据
            original_sensor_path=None,                               # 不加载原始传感器数据
            data_path=Path(cfg.navsim_log_path),                     # 日志数据路径
            synthetic_scenes_path=Path(cfg.synthetic_scenes_path),   # 合成场景路径
            scene_filter=scene_filter,                               # 场景过滤器
            sensor_config=SensorConfig.build_no_sensors(),           # 不加载传感器
        )

        # 创建指标缓存处理器前的断言检查
        assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"

        # 创建指标缓存处理器
        processor = MetricCacheProcessor(
            cache_path=cfg.metric_cache_path,                        # 缓存保存路径
            force_feature_computation=cfg.force_feature_computation, # 是否强制重新计算
            proposal_sampling=instantiate(cfg.proposal_sampling),    # 提议采样配置
        )

        # 记录该工作进程提取到的场景数量
        logger.info(f"Extracted {len(scene_loader)} scenarios for thread_id={thread_id}, node_id={node_id}.")
        # 记录scene_loader的数据结构信息
        logger.info(f"[cache_scenarios_internal] scene_loader 结构:")
        logger.info(f"  - scene_frames_dicts 数量: {len(scene_loader.scene_frames_dicts)}")
        logger.info(f"  - synthetic_scenes 数量: {len(scene_loader.synthetic_scenes)}")

        # 初始化统计变量
        num_failures = 0                                            # 失败场景数
        num_successes = 0                                           # 成功场景数
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []  # 所有缓存元数据列表

        # 遍历所有原始场景
        for idx, (token, scene_dict) in enumerate(scene_loader.scene_frames_dicts.items()):
            # 记录当前处理进度
            logger.info(
                f"Processing scenario {idx + 1} / {len(scene_loader.scene_frames_dicts)} in thread_id={thread_id}, node_id={node_id}"
            )
            # 只在处理第一个场景时打印数据结构信息
            if idx == 0:
                logger.info(f"[cache_scenarios_internal] scene_dict 数据结构:")
                logger.info(f"  - token: {token}")
                logger.info(f"  - 帧数量: {len(scene_dict)}")
                if scene_dict:
                    logger.info(f"  - 第一帧键: {list(scene_dict[0].keys())}")
                    first_frame = scene_dict[0]
                    logger.info(f"    - token: {first_frame.get('token', 'N/A')[:20]}...")
                    logger.info(f"    - timestamp: {first_frame.get('timestamp', 'N/A')}")
                    logger.info(f"    - ego_status keys: {list(first_frame.get('ego_status', {}).keys())}")
            # 处理单个场景，计算并保存指标缓存
            file_cache_metadata = cache_single_scenario(scene_dict, processor)
            # 显式触发垃圾回收，释放内存
            gc.collect()

            # 根据返回结果更新统计信息
            if file_cache_metadata:
                logger.info(f"  ✓ 成功 - token: {token}, file_name: {file_cache_metadata.file_name}")
            else:
                logger.info(f"  ✗ 失败 - token: {token}")

            # 更新失败计数
            num_failures += 0 if file_cache_metadata else 1
            # 更新成功计数
            num_successes += 1 if file_cache_metadata else 0
            # 添加到元数据列表
            all_file_cache_metadata += [file_cache_metadata]

        # 遍历所有合成场景（如果有）
        for idx, (scene_path, _) in enumerate(scene_loader.synthetic_scenes.values()):
            # 记录当前处理进度
            logger.info(
                f"Processing synthetic scenario {idx + 1} / {len(scene_loader.synthetic_scenes)} in thread_id={thread_id}, node_id={node_id}"
            )
            # 处理单个合成场景
            file_cache_metadata = cache_single_synthetic_scenario(scene_path, processor)
            # 显式触发垃圾回收
            gc.collect()

            # 更新统计信息
            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        # 记录该工作进程处理完成
        logger.info(f"Finished processing scenarios for thread_id={thread_id}, node_id={node_id}")
        # 返回缓存结果
        return [
            CacheResult(
                failures=num_failures,                               # 失败数量
                successes=num_successes,                             # 成功数量
                cache_metadata=all_file_cache_metadata,              # 所有缓存元数据
            )
        ]

    # 调用内部函数处理任务
    result = cache_scenarios_internal(args)

    # 强制触发垃圾回收，清理未使用的资源
    gc.collect()

    # 返回处理结果
    return result


def cache_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """
    Metric缓存的主函数，负责初始化数据加载器、分发任务、汇总结果。
    :param cfg: Omegaconf配置字典，包含所有必要的配置参数
    :param worker: 工作池对象，用于任务的分布式执行
    """
    # 断言检查缓存路径必须存在
    assert cfg.metric_cache_path is not None, f"Cache path cannot be None when caching, got {cfg.metric_cache_path}"

    # 创建场景加载器，用于获取所有场景的token列表（用于任务分发）
    # TODO: 后续可以从元数据推断每个日志的token，避免在这里加载场景
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,                                  # 不加载合成传感器数据
        original_sensor_path=None,                                   # 不加载原始传感器数据
        data_path=Path(cfg.navsim_log_path),                         # 日志数据路径
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),       # 合成场景路径
        scene_filter=instantiate(cfg.train_test_split.scene_filter), # 场景过滤器配置
        sensor_config=SensorConfig.build_no_sensors(),               # 不加载传感器数据
    )

    # 记录SceneLoader的数据结构信息
    logger.info("[cache_data] SceneLoader 数据结构:")
    logger.info(f"  - navsim_log_path: {cfg.navsim_log_path}")
    logger.info(f"  - synthetic_scenes_path: {cfg.synthetic_scenes_path}")
    logger.info(f"  - scene_frames_dicts 数量: {len(scene_loader.scene_frames_dicts)}")
    logger.info(f"  - synthetic_scenes 数量: {len(scene_loader.synthetic_scenes)}")
    logger.info(f"  - 总场景数: {len(scene_loader)}")

    # 获取每个日志文件对应的token列表，用于任务分发
    tokens_per_log = scene_loader.get_tokens_list_per_log()
    # 记录tokens_per_log的数据结构信息
    logger.info(f"[cache_data] get_tokens_list_per_log() 结构:")
    logger.info(f"  - 日志文件数: {len(tokens_per_log)}")
    if tokens_per_log:
        sample_log = list(tokens_per_log.keys())[0]
        logger.info(f"  - 示例日志: {sample_log}")
        logger.info(f"  - 示例场景数: {len(tokens_per_log[sample_log])}")
        if tokens_per_log[sample_log]:
            logger.info(f"  - 示例token: {tokens_per_log[sample_log][0]}")

    # 构建任务列表，每个任务包含一个日志文件的信息
    data_points = [
        {
            "cfg": cfg,                                             # 配置字典
            "log_file": log_file,                                   # 日志文件名
            "tokens": tokens_list,                                  # 该日志文件对应的token列表
        }
        for log_file, tokens_list in tokens_per_log.items()
    ]
    # 记录开始缓存的日志文件数量
    logger.info("Starting metric caching of %s files...", str(len(data_points)))
    # 记录data_points的数据结构信息
    logger.info(f"[cache_data] data_points 数据结构:")
    logger.info(f"  - 长度: {len(data_points)}")
    if data_points:
        logger.info(f"  - 每个元素结构: ['cfg', 'log_file', 'tokens']")
        logger.info(f"  - 示例 log_file: {data_points[0]['log_file']}")
        logger.info(f"  - 示例 tokens 数量: {len(data_points[0]['tokens'])}")

    # 使用worker_map将任务分发到多个工作进程并行执行
    cache_results = worker_map(worker, cache_scenarios, data_points)

    # 记录cache_results的数据结构信息
    logger.info(f"[cache_data] cache_results 数据结构:")
    logger.info(f"  - 长度: {len(cache_results)}")
    if cache_results:
        logger.info(f"  - 每个元素类型: CacheResult")
        logger.info(f"  - 示例 successes: {cache_results[0].successes}")
        logger.info(f"  - 示例 failures: {cache_results[0].failures}")
        logger.info(f"  - 示例 cache_metadata 长度: {len(cache_results[0].cache_metadata)}")

    # 汇总所有工作进程的结果
    num_success = sum(result.successes for result in cache_results)   # 总成功数
    num_fail = sum(result.failures for result in cache_results)       # 总失败数
    num_total = num_success + num_fail                               # 总场景数

    # 根据结果记录不同的日志信息
    if num_fail == 0:
        logger.info(
            "Completed dataset caching! All %s features and targets were cached successfully.",
            str(num_total),
        )
    else:
        logger.info(
            "Completed dataset caching! Failed features and targets: %s out of %s",
            str(num_fail),
            str(num_total),
        )

    # 从所有CacheResult中提取有效的缓存元数据条目
    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    # 记录cached_metadata的数据结构信息
    logger.info(f"[cache_data] cached_metadata 数据结构:")
    logger.info(f"  - 长度: {len(cached_metadata)}")
    if cached_metadata:
        sample_meta = cached_metadata[0]
        logger.info(f"  - 每个元素类型: CacheMetadataEntry")
        logger.info(f"  - 示例 file_name: {sample_meta.file_name}")

    # 获取节点ID（分布式训练用）
    node_id = int(os.environ.get("NODE_RANK", 0))
    # 记录正在保存元数据CSV文件
    logger.info(f"Node {node_id}: Storing metadata csv file containing cache paths for valid features and targets...")
    # 将缓存元数据保存到CSV文件，用于后续训练/评估时加载缓存
    save_cache_metadata(cached_metadata, Path(cfg.metric_cache_path), node_id)
    # 记录元数据保存完成
    logger.info("Done storing metadata csv file.")