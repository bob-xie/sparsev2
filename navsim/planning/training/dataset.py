from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig
from pathlib import Path
import logging
import pickle
import gzip
import os

import torch
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """Helper function to load pickled feature/target from path."""
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """Helper function to save feature/target to pickle."""
    # Use compresslevel = 1 to compress the size but also has fast write and read.
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapper for feature/target datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        test_mode: bool = True,
        log_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)
        self.test_mode = test_mode

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders
        self._target_builders = target_builders
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,
            feature_builders=self._feature_builders,
            target_builders=self._target_builders,
            log_names=self.log_names,
        )
        self.tokens = list(self._valid_cache_paths.keys())

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        features, targets, token = self._load_scene_with_token(idx)
        if hasattr(self._feature_builders[0], 'pipeline'):
            features, targets, token = self._feature_builders[0].pipeline(features, targets, token, self.test_mode)

        return (features, targets, token)

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        log_names: List[Path],
    ) -> Dict[str, Path]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, Path] = {}

        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []
                for builder in feature_builders + target_builders:
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _load_scene_with_token(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """
        token = self.tokens[idx]
        token_path = self._valid_cache_paths[token]

        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        targets['token_path'] = str(token_path)
        return (features, targets, token)


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch数据集类，用于加载自动驾驶场景数据。
    
    支持两种模式：
    1. 缓存模式：从预计算的缓存文件加载特征和目标
    2. 实时计算模式：实时从原始数据计算特征和目标
    
    继承自 torch.utils.data.Dataset，可用于 PyTorch DataLoader。
    """

    def __init__(
        self,
        scene_loader: SceneLoader,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        test_mode: bool = True,
        cache_path: Optional[str] = None,
        force_cache_computation: bool = False,
        cfg: Optional[DictConfig] = None,
    ):
        """
        初始化数据集。
        
        :param scene_loader: 场景加载器，负责加载场景数据
        :param feature_builders: 特征构建器列表，用于提取输入特征
        :param target_builders: 目标构建器列表，用于构建训练目标
        :param test_mode: 是否为测试模式，影响数据处理流程
        :param cache_path: 缓存目录路径，为None时不使用缓存
        :param force_cache_computation: 是否强制重新计算缓存（覆盖已有缓存）
        :param cfg: Hydra配置对象，包含额外的配置参数
        """
        super().__init__()
        
        # 第155-158行：保存核心组件
        self._scene_loader = scene_loader  # 场景加载器
        self.test_mode = test_mode          # 测试模式标志
        self._feature_builders = feature_builders  # 特征构建器列表
        self._target_builders = target_builders    # 目标构建器列表

        # 第160-164行：缓存相关配置
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None  # 缓存路径
        self._force_cache_computation = force_cache_computation  # 强制缓存标志
        # 加载已有的有效缓存路径
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )
        self._cfg = cfg  # 配置对象

        # 第167-168行：如果指定了缓存路径，自动执行缓存
        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
    ) -> Dict[str, Path]:
        """
        加载有效的缓存路径，验证每个场景的所有特征和目标文件是否都存在。
        
        :param cache_path: 训练缓存文件夹目录
        :param feature_builders: 特征构建器列表
        :param target_builders: 目标构建器列表
        :return: 字典，键为场景token，值为该场景的缓存目录路径
        """

        valid_cache_paths: Dict[str, Path] = {}

        # 只有当缓存路径存在且是目录时才加载
        if (cache_path is not None) and cache_path.is_dir():
            # 遍历所有日志目录
            for log_path in cache_path.iterdir():
                # 遍历每个日志目录下的场景token目录
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []
                    # 检查所有特征和目标构建器的缓存文件是否存在
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())
                    # 只有所有缓存文件都存在时，才认为该场景有效
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths

    def _cache_scene_with_token(self, token: str) -> None:
        """
        计算单个场景的特征和目标，并保存到缓存。
        
        :param token: 要缓存的场景唯一标识符
        """

        # 从场景加载器获取场景对象
        scene = self._scene_loader.get_scene_from_token(token)
        # 获取智能体输入数据
        agent_input = scene.get_agent_input()

        # 构建缓存目录路径：cache_path/log_name/token/
        metadata = scene.scene_metadata
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)  # 创建目录（如果不存在）

        # 计算并保存特征
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = builder.compute_features(agent_input)  # 使用构建器计算特征sparsedrive_features.py
            dump_feature_target_to_pickle(data_dict_path, data_dict)  # 保存到缓存

        # 计算并保存目标
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            # 合成场景不需要计算目标（用于warmup阶段）
            if token in self._scene_loader.synthetic_scenes:
                data_dict = {}
            else:
                data_dict = builder.compute_targets(scene, self._cfg)  # 使用构建器计算目标
            dump_feature_target_to_pickle(data_dict_path, data_dict)  # 保存到缓存

        # 更新有效缓存路径字典
        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        从缓存加载单个场景的特征和目标。
        
        :param token: 要加载的场景唯一标识符
        :return: 元组，包含特征字典和目标字典
        """

        # 获取该场景的缓存路径
        token_path = self._valid_cache_paths[token]

        # 加载特征
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)  # 从缓存加载
            features.update(data_dict)  # 合并到特征字典

        # 加载目标
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)  # 从缓存加载
            targets.update(data_dict)  # 合并到目标字典

        return (features, targets)

    def cache_dataset(self) -> None:
        """
        将整个数据集缓存到缓存文件夹。
        
        如果force_cache_computation=True，则重新计算所有场景；
        否则只计算尚未缓存的场景。
        """

        # 确保缓存路径已设置
        assert self._cache_path is not None, "Dataset did not receive a cache path!"
        # 创建缓存目录（如果不存在）
        os.makedirs(self._cache_path, exist_ok=True)

        # 确定需要缓存的场景token
        if self._force_cache_computation:
            # 强制重新计算所有场景
            tokens_to_cache = self._scene_loader.tokens
        else:
            # 只计算尚未缓存的场景（集合差集）
            tokens_to_cache = set(self._scene_loader.tokens) - set(self._valid_cache_paths.keys())
            tokens_to_cache = list(tokens_to_cache)
            # 打印日志信息
            logger.info(
                f"""
                Starting caching of {len(tokens_to_cache)} tokens.
                Note: Caching tokens within the training loader is slow. Only use it with a small number of tokens.
                You can cache large numbers of tokens using the `run_dataset_caching.py` python script.
                """
            )

        # 遍历所有需要缓存的场景，显示进度条
        for token in tqdm(tokens_to_cache, desc="Caching Dataset"):
            self._cache_scene_with_token(token)

    def __len__(self) -> None:
        """
        返回数据集的样本数量。
        
        :return: 样本数量
        """
        # 直接委托给场景加载器
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        根据索引获取样本。
        
        如果配置了缓存路径，则从缓存加载；否则实时计算。
        
        :param idx: 样本索引
        :return: 元组，包含特征字典、目标字典和场景token
        """

        # 获取指定索引的场景token
        token = self._scene_loader.tokens[idx]
        features: Dict[str, torch.Tensor] = {}
        targets: Dict[str, torch.Tensor] = {}

        # 判断是否使用缓存
        if self._cache_path is not None:
            # 确保该场景已缓存
            assert (
                token in self._valid_cache_paths.keys()
            ), f"The token {token} has not been cached yet, please call cache_dataset first!"

            # 从缓存加载特征和目标
            features, targets = self._load_scene_with_token(token)
        else:
            # 实时计算特征和目标（不使用缓存）
            scene = self._scene_loader.get_scene_from_token(self._scene_loader.tokens[idx])
            agent_input = scene.get_agent_input()
            # 计算特征
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            # 计算目标（合成场景除外）
            for builder in self._target_builders:
                if token in self._scene_loader.synthetic_scenes:
                    targets.update({})
                else:
                    targets.update(builder.compute_targets(scene, self._cfg))

        # 如果特征构建器有pipeline方法，执行额外处理（如数据增强、归一化等）
        if hasattr(self._feature_builders[0], 'pipeline'):
            features, targets, token = self._feature_builders[0].pipeline(features, targets, token, self.test_mode)
        
        # 添加token路径到目标字典（用于后续处理或调试）
        targets['token_path'] = str(self._valid_cache_paths[token])
        return (features, targets, token)
