# 导入类型注解模块，用于函数参数和返回值的类型提示
from typing import Dict, List, Optional, Tuple

# 导入 OmegaConf 配置管理库
from omegaconf import DictConfig

# 导入路径处理模块
from pathlib import Path

# 导入日志模块
import logging

# 导入 pickle 序列化模块（用于Python对象序列化）
import pickle

# 导入 gzip 压缩模块（用于压缩缓存文件）
import gzip

# 导入操作系统模块（用于文件目录操作）
import os

# 导入 PyTorch 张量库
import torch

# 导入进度条库
from tqdm import tqdm

# 导入场景加载器
from navsim.common.dataloader import SceneLoader

# 导入特征和目标构建器的抽象基类
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)


def load_feature_target_from_pickle(path: Path) -> Dict[str, torch.Tensor]:
    """
    从 pickle 文件加载特征或目标数据

    :param path: pickle 文件的路径（通常是 .gz 压缩文件）
    :return: 包含张量的字典，键为特征/目标名称，值为对应的 PyTorch 张量
    """
    # 使用 gzip 打开压缩文件并加载 pickle 数据
    with gzip.open(path, "rb") as f:
        data_dict: Dict[str, torch.Tensor] = pickle.load(f)
    return data_dict


def dump_feature_target_to_pickle(path: Path, data_dict: Dict[str, torch.Tensor]) -> None:
    """
    将特征或目标数据保存到 pickle 文件

    :param path: 保存的文件路径（通常带 .gz 扩展名）
    :param data_dict: 包含张量的字典
    """
    # 使用 gzip 压缩保存，compresslevel=1 平衡压缩率和读写速度
    with gzip.open(path, "wb", compresslevel=1) as f:
        pickle.dump(data_dict, f)


class CacheOnlyDataset(torch.utils.data.Dataset):
    """
    仅从缓存加载特征/目标的数据集包装器

    继承自 torch.utils.data.Dataset，用于在已有缓存的情况下快速加载数据。
    不依赖原始传感器数据，适用于缓存已生成完毕的常规训练场景。

    数据目录结构：
        cache_path/
        ├── log_name_1/
        │   ├── token_1/
        │   │   ├── feature_builder_name.gz      # 特征缓存文件
        │   │   └── target_builder_name.gz      # 目标缓存文件
        │   └── token_2/
        │       └── ...
        └── log_name_2/
            └── ...
    """

    def __init__(
        self,
        cache_path: str,                                       # 缓存文件夹路径，如 "exp/data_cache_navmini"
        feature_builders: List[AbstractFeatureBuilder],         # 特征构建器列表，用于解析特征缓存
        target_builders: List[AbstractTargetBuilder],           # 目标构建器列表，用于解析目标缓存
        test_mode: bool = True,                                # 测试模式标志：True=验证模式（禁用数据增强），False=训练模式（启用数据增强）
        log_names: Optional[List[str]] = None,                 # 日志名称列表，用于筛选缓存文件；为None时使用cache_path下所有目录
    ):
        """
        初始化仅缓存的数据集

        :param cache_path: 缓存文件夹目录路径
        :param feature_builders: 特征构建器列表
        :param target_builders: 目标构建器列表
        :param test_mode: 是否为测试/验证模式（影响数据增强）
        :param log_names: 可选的日志名称列表，用于筛选缓存文件
        """
        super().__init__()  # 调用父类 torch.utils.data.Dataset 的初始化

        # 确保缓存路径存在
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)  # 保存缓存路径为 Path 对象
        self.test_mode = test_mode  # 保存测试模式标志

        # 处理日志名称列表
        if log_names is not None:
            # 如果指定了日志名称，只保留在缓存中存在的日志目录
            # 将字符串列表转换为 Path 对象列表，并过滤存在的目录
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            # 如果没有指定日志名称，使用缓存目录下的所有子目录作为日志
            self.log_names = [log_name for log_name in self._cache_path.iterdir()]

        self._feature_builders = feature_builders  # 保存特征构建器列表
        self._target_builders = target_builders    # 保存目标构建器列表

        # 加载有效的缓存路径，建立 token 到缓存目录的映射
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            cache_path=self._cache_path,          # 缓存目录路径
            feature_builders=self._feature_builders,  # 特征构建器列表
            target_builders=self._target_builders,    # 目标构建器列表
            log_names=self.log_names,             # 日志名称列表
        )
        self.tokens = list(self._valid_cache_paths.keys())  # 提取所有有效的 token 列表

    def __len__(self) -> int:
        """
        返回数据集中样本的数量

        :return: 有效样本的数量
        """
        return len(self.tokens)  # 返回 token 列表的长度

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        根据索引加载并返回样本的特征和目标

        :param idx: 样本的整数索引
        :return: 元组 (features, targets, token)
                 - features: 特征字典，键为特征名，值为 PyTorch 张量
                 - targets: 目标字典，键为目标名，值为 PyTorch 张量
                 - token: 场景的唯一标识字符串
        """
        # 加载指定索引的特征、目标和 token
        features, targets, token = self._load_scene_with_token(idx)

        # 如果第一个特征构建器有 pipeline 方法，执行数据增强等处理
        # pipeline 方法通常包含数据增强、归一化等操作
        if hasattr(self._feature_builders[0], 'pipeline'):
            features, targets, token = self._feature_builders[0].pipeline(
                features, targets, token, self.test_mode
            )

        return (features, targets, token)

    @staticmethod
    def _load_valid_caches(
        cache_path: Path,                                     # 缓存目录的 Path 对象
        feature_builders: List[AbstractFeatureBuilder],        # 特征构建器列表
        target_builders: List[AbstractTargetBuilder],          # 目标构建器列表
        log_names: List[Path],                               # 日志名称的 Path 对象列表
    ) -> Dict[str, Path]:
        """
        静态方法：加载有效的缓存路径

        遍历缓存目录，验证每个场景的所有特征和目标缓存文件是否都存在。
        只有当所有构建器对应的缓存文件都存在时，才认为该场景有效。

        :param cache_path: 缓存文件夹目录
        :param feature_builders: 特征构建器列表
        :param target_builders: 目标构建器列表
        :param log_names: 日志名称的 Path 对象列表
        :return: 字典，键为 token 字符串，值为该 token 对应的缓存目录 Path 对象
        """

        valid_cache_paths: Dict[str, Path] = {}  # 初始化有效缓存路径字典

        # 遍历所有日志目录，显示进度条
        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name  # 拼接日志的完整路径

            # 遍历日志目录下的每个 token 目录
            for token_path in log_path.iterdir():
                found_caches: List[bool] = []  # 存储每个构建器缓存文件是否存在的布尔列表

                # 检查所有特征和目标构建器的缓存文件是否存在
                for builder in feature_builders + target_builders:
                    # 构建缓存文件路径：token_path/builder_name.gz
                    data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                    found_caches.append(data_dict_path.is_file())  # 检查文件是否存在

                # 只有当所有构建器的缓存文件都存在时，才认为该场景有效
                if all(found_caches):
                    valid_cache_paths[token_path.name] = token_path  # 添加到有效缓存字典

        return valid_cache_paths  # 返回有效缓存路径字典

    def _load_scene_with_token(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        根据索引加载单个场景的特征和目标

        :param idx: 样本的整数索引
        :return: 元组 (features, targets, token)
        """
        token = self.tokens[idx]  # 根据索引获取对应的 token
        token_path = self._valid_cache_paths[token]  # 获取该 token 对应的缓存目录路径

        # 加载特征
        features: Dict[str, torch.Tensor] = {}  # 初始化特征字典
        for builder in self._feature_builders:  # 遍历每个特征构建器
            # 构建缓存文件路径
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            # 从缓存文件加载特征数据
            data_dict = load_feature_target_from_pickle(data_dict_path)
            # 合并到特征字典（使用 update 方法，多个构建器的结果会合并）
            features.update(data_dict)

        # 加载目标
        targets: Dict[str, torch.Tensor] = {}  # 初始化目标字典
        for builder in self._target_builders:  # 遍历每个目标构建器
            # 构建缓存文件路径
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            # 从缓存文件加载目标数据
            data_dict = load_feature_target_from_pickle(data_dict_path)
            # 合并到目标字典
            targets.update(data_dict)

        # 将 token 对应的缓存路径保存到目标字典中（用于后续处理或调试）
        targets['token_path'] = str(token_path)

        return (features, targets, token)


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch 数据集类，用于加载自动驾驶场景数据

    支持两种数据加载模式：
    1. 缓存模式：从预计算的缓存文件加载特征和目标（通过 cache_path 指定）
    2. 实时计算模式：直接从原始传感器数据计算特征和目标（不指定 cache_path）

    支持在训练开始前自动缓存数据集，通过 force_cache_computation 控制是否覆盖已有缓存。

    继承自 torch.utils.data.Dataset，可直接用于 PyTorch DataLoader。
    """

    def __init__(
        self,
        scene_loader: SceneLoader,                             # 场景加载器，负责从原始日志加载场景数据
        feature_builders: List[AbstractFeatureBuilder],         # 特征构建器列表，用于从场景提取输入特征
        target_builders: List[AbstractTargetBuilder],           # 目标构建器列表，用于从场景构建训练目标
        test_mode: bool = True,                                # 测试模式标志：True=验证模式，False=训练模式
        cache_path: Optional[str] = None,                      # 缓存目录路径，为 None 时不使用缓存
        force_cache_computation: bool = False,                  # 是否强制重新计算缓存（True=覆盖已有缓存）
        cfg: Optional[DictConfig] = None,                      # Hydra 配置对象，包含数据集配置参数
    ):
        """
        初始化数据集

        :param scene_loader: 场景加载器实例
        :param feature_builders: 特征构建器列表
        :param target_builders: 目标构建器列表
        :param test_mode: 是否为测试/验证模式
        :param cache_path: 缓存目录路径，为 None 时不使用缓存
        :param force_cache_computation: 是否强制重新计算缓存
        :param cfg: Hydra 配置对象
        """
        super().__init__()  # 调用父类初始化

        # 保存核心组件
        self._scene_loader = scene_loader      # 场景加载器
        self.test_mode = test_mode            # 测试模式标志
        self._feature_builders = feature_builders  # 特征构建器列表
        self._target_builders = target_builders    # 目标构建器列表

        # 处理缓存路径
        self._cache_path: Optional[Path] = Path(cache_path) if cache_path else None  # 转换或设为 None
        self._force_cache_computation = force_cache_computation  # 保存强制缓存标志

        # 加载已有的有效缓存路径
        self._valid_cache_paths: Dict[str, Path] = self._load_valid_caches(
            self._cache_path, feature_builders, target_builders
        )
        self._cfg = cfg  # 保存配置对象

        # 如果指定了缓存路径，自动执行缓存
        if self._cache_path is not None:
            self.cache_dataset()

    @staticmethod
    def _load_valid_caches(
        cache_path: Optional[Path],                           # 缓存目录路径
        feature_builders: List[AbstractFeatureBuilder],        # 特征构建器列表
        target_builders: List[AbstractTargetBuilder],          # 目标构建器列表
    ) -> Dict[str, Path]:
        """
        加载有效的缓存路径，验证每个场景的所有特征和目标文件是否都存在

        :param cache_path: 缓存文件夹目录
        :param feature_builders: 特征构建器列表
        :param target_builders: 目标构建器列表
        :return: 字典，键为场景 token，值为该场景的缓存目录路径
        """

        valid_cache_paths: Dict[str, Path] = {}  # 初始化有效缓存字典

        # 只有当缓存路径存在且是目录时才加载
        if (cache_path is not None) and cache_path.is_dir():
            # 遍历所有日志目录
            for log_path in cache_path.iterdir():
                # 遍历每个日志目录下的场景 token 目录
                for token_path in log_path.iterdir():
                    found_caches: List[bool] = []  # 存储每个构建器缓存是否存在

                    # 检查所有特征和目标构建器的缓存文件是否存在
                    for builder in feature_builders + target_builders:
                        data_dict_path = token_path / (builder.get_unique_name() + ".gz")
                        found_caches.append(data_dict_path.is_file())

                    # 只有所有缓存文件都存在时，才认为该场景有效
                    if all(found_caches):
                        valid_cache_paths[token_path.name] = token_path

        return valid_cache_paths  # 返回有效缓存路径字典

    def _cache_scene_with_token(self, token: str) -> None:
        """
        计算单个场景的特征和目标，并保存到缓存

        :param token: 要缓存的场景唯一标识符
        """

        # 从场景加载器获取场景对象
        scene = self._scene_loader.get_scene_from_token(token)

        # 获取智能体输入数据（包含传感器数据、自车状态等）
        agent_input = scene.get_agent_input()

        # 获取场景元数据
        metadata = scene.scene_metadata

        # 构建缓存目录路径：cache_path/log_name/token/
        token_path = self._cache_path / metadata.log_name / metadata.initial_token
        os.makedirs(token_path, exist_ok=True)  # 创建目录（如果不存在）

        # 计算并保存特征
        for builder in self._feature_builders:
            # 构建缓存文件路径
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            # 使用构建器计算特征
            data_dict = builder.compute_features(agent_input)
            # 保存到缓存文件
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        # 计算并保存目标
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")

            # 合成场景不需要计算目标（用于 warmup 阶段）
            if token in self._scene_loader.synthetic_scenes:
                data_dict = {}
            else:
                # 使用构建器计算目标
                data_dict = builder.compute_targets(scene, self._cfg)

            # 保存到缓存文件
            dump_feature_target_to_pickle(data_dict_path, data_dict)

        # 更新有效缓存路径字典
        self._valid_cache_paths[token] = token_path

    def _load_scene_with_token(self, token: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        从缓存加载单个场景的特征和目标

        :param token: 要加载的场景唯一标识符
        :return: 元组 (features, targets)
        """

        # 获取该场景的缓存路径
        token_path = self._valid_cache_paths[token]

        # 加载特征
        features: Dict[str, torch.Tensor] = {}
        for builder in self._feature_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            features.update(data_dict)

        # 加载目标
        targets: Dict[str, torch.Tensor] = {}
        for builder in self._target_builders:
            data_dict_path = token_path / (builder.get_unique_name() + ".gz")
            data_dict = load_feature_target_from_pickle(data_dict_path)
            targets.update(data_dict)

        return (features, targets)

    def cache_dataset(self) -> None:
        """
        将整个数据集缓存到缓存文件夹

        如果 force_cache_computation=True，则重新计算所有场景；
        否则只计算尚未缓存的场景。

        注意：在训练数据加载器中缓存 tokens 较慢，
        建议使用 `run_dataset_caching.py` 脚本进行大规模缓存。
        """

        # 确保缓存路径已设置
        assert self._cache_path is not None, "Dataset did not receive a cache path!"

        # 创建缓存目录（如果不存在）
        os.makedirs(self._cache_path, exist_ok=True)

        # 确定需要缓存的场景 token
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

    def __len__(self) -> int:
        """
        返回数据集的样本数量

        :return: 样本数量
        """
        # 直接委托给场景加载器
        return len(self._scene_loader)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        根据索引获取样本

        如果配置了缓存路径，则从缓存加载；否则实时计算。

        :param idx: 样本索引
        :return: 元组 (features, targets, token)
        """

        # 获取指定索引的场景 token
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

        # 如果特征构建器有 pipeline 方法，执行额外处理（如数据增强、归一化等）
        if hasattr(self._feature_builders[0], 'pipeline'):
            features, targets, token = self._feature_builders[0].pipeline(
                features, targets, token, self.test_mode
            )

        # 添加 token 路径到目标字典（用于后续处理或调试）
        targets['token_path'] = str(self._valid_cache_paths[token])
        return (features, targets, token)
