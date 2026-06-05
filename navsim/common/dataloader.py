from __future__ import annotations

import lzma
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
import pickle

from tqdm import tqdm

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache

FrameList = List[Dict[str, Any]]


def filter_scenes(data_path: Path, scene_filter: SceneFilter) -> Tuple[Dict[str, FrameList], List[str]]:
    """
    从数据集中加载场景，并应用场景过滤配置。
    
    :param data_path: 日志文件夹的根目录
    :param scene_filter: 场景过滤配置类，包含过滤规则
    :return: 元组，包含两个元素：
             1. 过滤后的场景字典，键为场景token，值为帧列表（FrameList）
             2. 最终帧token列表，用于过滤第二阶段的合成场景
    """

    def split_list(input_list: List[Any], num_frames: int, frame_interval: int) -> List[List[Any]]:
        """
        根据采样规格将帧列表分割成多个场景片段。
        
        :param input_list: 原始帧列表
        :param num_frames: 每个场景包含的帧数
        :param frame_interval: 场景之间的帧间隔（步长）
        :return: 分割后的场景片段列表
        """
        # 使用滑动窗口方式分割：从索引0开始，每隔frame_interval取num_frames个帧
        return [input_list[i : i + num_frames] for i in range(0, len(input_list), frame_interval)]

    # 第31行：存储过滤后的场景，键为token，值为帧列表
    filtered_scenes: Dict[str, FrameList] = {}
    
    # 第33行：跟踪原始场景的最后一帧token，用于关联合成场景
    final_frame_tokens: List[str] = []
    
    # 第34行：加载停止标志，当达到max_scenes时停止
    stop_loading: bool = False

    # 第37行：获取数据路径下的所有日志文件
    log_files = list(data_path.iterdir())
    
    # 第38-39行：如果配置了log_names过滤，只保留指定的日志文件
    if scene_filter.log_names is not None:
        log_files = [log_file for log_file in log_files 
                     if log_file.name.replace(".pkl", "") in scene_filter.log_names]

    # 第41-45行：判断是否需要按token过滤
    if scene_filter.tokens is not None:
        filter_tokens = True
        tokens = set(scene_filter.tokens)  # 转换为set提高查找效率
    else:
        filter_tokens = False

    # 第47行：遍历所有日志文件，显示加载进度条
    for log_pickle_path in tqdm(log_files, desc="Loading logs"):

        # 第49行：加载pickle文件，获取帧字典列表
        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        # 打印pickle文件名和对应的帧数量
        print(f"[filter_scenes] 加载文件: {log_pickle_path.name}, 帧数: {len(scene_dict_list)}")
        
        # 第50行：按滑动窗口分割帧列表，生成场景片段
        for frame_list in split_list(scene_dict_list, scene_filter.num_frames, scene_filter.frame_interval):
            
            # 第52-53行：过滤帧数不足的场景（最后一个片段可能不足）
            if len(frame_list) < scene_filter.num_frames:
                continue

            # 第56-57行：过滤没有路线信息的场景（has_route=True时）
            # 检查历史帧的最后一帧是否有roadblock_ids（路线信息）
            if scene_filter.has_route and len(frame_list[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                continue

            # 第60-62行：按token过滤场景
            # 取历史帧的最后一帧的token作为场景标识
            token = frame_list[scene_filter.num_history_frames - 1]["token"]
            if filter_tokens and token not in tokens:
                continue
                #每个场景的token都不同,一个pickle文件可能包含多个场景
            # 第64行：将符合条件的场景加入结果字典
            filtered_scenes[token] = frame_list
            
            # 第65行：记录场景的最后一帧token（用于关联合成场景）
            final_frame_token = frame_list[scene_filter.num_frames - 1]["token"]
            # 注意：如果num_future_frames > proposal_sampling帧数，这里的索引可能不正确
            final_frame_tokens.append(final_frame_token)

            # 第69-71行：检查是否达到最大场景数限制
            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                stop_loading = True
                break

        # 第73-74行：外层循环也需要检查停止标志
        if stop_loading:
            break

    # 第76行：返回过滤后的场景字典和最终帧token列表
    return filtered_scenes, final_frame_tokens


def filter_synthetic_scenes(
    data_path: Path, scene_filter: SceneFilter, stage1_scenes_final_frames_tokens: List[str]
) -> Dict[str, Tuple[Path, str]]:
    # Load all the synthetic scenes that belong to the original scenes already loaded
    loaded_scenes: Dict[str, Tuple[Path, str, int]] = {}
    synthetic_scenes_paths = list(data_path.iterdir())

    filter_logs = scene_filter.log_names is not None
    filter_tokens = scene_filter.synthetic_scene_tokens is not None

    for scene_path in tqdm(synthetic_scenes_paths, desc="Loading synthetic scenes"):
        synthetic_scene = Scene.load_from_disk(scene_path, None, None)

        # if a token is requested specifically, we load it even if it is not related to the original scenes loaded
        if filter_tokens and synthetic_scene.scene_metadata.initial_token not in scene_filter.synthetic_scene_tokens:
            continue

        # filter by log names
        log_name = synthetic_scene.scene_metadata.log_name
        if filter_logs and log_name not in scene_filter.log_names:
            continue

        # if we don't filter for tokens explicitly, we load only the synthetic scenes required to run a second stage for the original scenes loaded
        if (
            not filter_tokens
            and synthetic_scene.scene_metadata.corresponding_original_scene not in stage1_scenes_final_frames_tokens
        ):
            continue

        loaded_scenes.update({synthetic_scene.scene_metadata.initial_token: [scene_path, log_name]})

    return loaded_scenes


class SceneLoader:
    """简单的场景数据加载器，从日志文件中加载场景。"""

    def __init__(
        self,
        data_path: Path,
        original_sensor_path: Path,
        scene_filter: SceneFilter,
        synthetic_sensor_path: Path = None,
        synthetic_scenes_path: Path = None,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
    ):
        """
        初始化场景数据加载器。
        
        :param data_path: 日志文件夹的根目录
        :param original_sensor_path: 原始传感器数据根目录
        :param scene_filter: 场景过滤配置类
        :param synthetic_sensor_path: 合成传感器数据根目录（可选）
        :param synthetic_scenes_path: 合成场景根目录（可选）
        :param sensor_config: 传感器加载配置，默认为无传感器
        """

        # 第134行：调用filter_scenes函数过滤原始场景
        # 返回两个值：过滤后的场景字典，以及第1阶段场景的最后一帧token列表
        self.scene_frames_dicts, stage1_scenes_final_frames_tokens = filter_scenes(data_path, scene_filter)
        
        # 第135-138行：保存各种配置和路径
        self._synthetic_sensor_path = synthetic_sensor_path  # 合成传感器数据路径
        self._original_sensor_path = original_sensor_path    # 原始传感器数据路径
        self._scene_filter = scene_filter                    # 场景过滤器配置
        self._sensor_config = sensor_config                  # 传感器配置

        # 第140行：检查是否需要包含合成场景
        if scene_filter.include_synthetic_scenes:
            # 第141-143行：确保合成场景路径已提供
            assert (
                synthetic_scenes_path is not None
            ), "当设置include_synthetic_scenes为True时，synthetic_scenes_path不能为None。"
            
            # 第144-148行：过滤合成场景
            self.synthetic_scenes = filter_synthetic_scenes(
                data_path=synthetic_scenes_path,
                scene_filter=scene_filter,
                stage1_scenes_final_frames_tokens=stage1_scenes_final_frames_tokens,
            )
            # 第149行：保存合成场景的token集合
            self.synthetic_scenes_tokens = set(self.synthetic_scenes.keys())
        else:
            # 第151-152行：如果不包含合成场景，初始化为空
            self.synthetic_scenes = {}
            self.synthetic_scenes_tokens = set()

    @property
    def tokens(self) -> List[str]:
        """
        获取所有场景标识符列表（原始场景 + 合成场景）。
        
        :return: 场景标识符列表
        """
        # 第159行：合并原始场景和合成场景的token
        return list(self.scene_frames_dicts.keys()) + list(self.synthetic_scenes.keys())

    @property
    def tokens_stage_one(self) -> List[str]:
        """
        获取第1阶段（原始）场景的标识符列表。
        
        :return: 场景标识符列表
        """
        # 第167行：只返回原始场景的token
        return list(self.scene_frames_dicts.keys())

    @property
    def reactive_tokens_stage_two(self) -> List[str]:
        """
        获取第2阶段的反应式合成场景的标识符列表。
        
        :return: 场景标识符列表
        """
        # 第175行：从配置中获取反应式合成场景的初始token
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        
        # 第176-177行：如果没有配置，则返回None
        if reactive_synthetic_initial_tokens is None:
            return None
        
        # 第178行：返回两个集合的交集（既在合成场景中，又在反应式配置中的token）
        return list(set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens))

    @property
    def non_reactive_tokens_stage_two(self) -> List[str]:
        """
        获取第2阶段的非反应式合成场景的标识符列表。
        
        :return: 场景标识符列表
        """
        # 第186行：从配置中获取非反应式合成场景的初始token
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        
        # 第187-188行：如果没有配置，则返回None
        if non_reactive_synthetic_initial_tokens is None:
            return None
        
        # 第189行：返回两个集合的交集
        return list(set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens))

    @property
    def reactive_tokens(self) -> List[str]:
        """
        获取原始场景和反应式合成场景的标识符列表。
        
        :return: 场景标识符列表
        """
        # 第197行：从配置中获取反应式合成场景的初始token
        reactive_synthetic_initial_tokens = self._scene_filter.reactive_synthetic_initial_tokens
        
        # 第198-199行：如果没有配置，只返回原始场景
        if reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        
        # 第200-202行：返回原始场景 + 反应式合成场景
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(reactive_synthetic_initial_tokens)
        )

    @property
    def non_reactive_tokens(self) -> List[str]:
        """
        获取原始场景和非反应式合成场景的标识符列表。
        
        :return: 场景标识符列表
        """
        # 第210行：从配置中获取非反应式合成场景的初始token
        non_reactive_synthetic_initial_tokens = self._scene_filter.non_reactive_synthetic_initial_tokens
        
        # 第211-212行：如果没有配置，只返回原始场景
        if non_reactive_synthetic_initial_tokens is None:
            return list(self.scene_frames_dicts.keys())
        
        # 第213-215行：返回原始场景 + 非反应式合成场景
        return list(self.scene_frames_dicts.keys()) + list(
            set(self.synthetic_scenes_tokens) & set(non_reactive_synthetic_initial_tokens)
        )

    def __len__(self) -> int:
        """
        魔法方法：返回可加载的场景总数。
        
        :return: 场景数量
        """
        # 第221行：返回所有token的数量
        return len(self.tokens)

    def __getitem__(self, idx) -> str:
        """
        魔法方法：通过索引获取场景标识符。
        
        :param idx: 场景索引
        :return: 场景唯一标识符
        """
        # 第228行：返回指定索引的token
        return self.tokens[idx]

    def get_scene_from_token(self, token: str) -> Scene:
        """
        根据场景标识符字符串加载完整的场景对象。
        
        :param token: 场景标识符字符串
        :return: Scene数据类对象
        """
        # 第236行：确保token在已知的token列表中
        assert token in self.tokens
        
        # 第237行：判断是否是合成场景
        if token in self.synthetic_scenes:
            # 第238-242行：从磁盘加载合成场景
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],  # 合成场景文件路径
                sensor_blobs_path=self._synthetic_sensor_path,  # 合成传感器数据路径
                sensor_config=self._sensor_config,  # 传感器配置
            )
        else:
            # 第244-250行：从原始场景字典列表构建场景
            return Scene.from_scene_dict_list(
                self.scene_frames_dicts[token],  # 原始场景帧字典列表
                self._original_sensor_path,  # 原始传感器数据路径
                num_history_frames=self._scene_filter.num_history_frames,  # 历史帧数
                num_future_frames=self._scene_filter.num_future_frames,  # 未来帧数
                sensor_config=self._sensor_config,  # 传感器配置
            )

    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        根据场景标识符字符串加载智能体输入对象。
        
        :param token: 场景标识符字符串
        :return: AgentInput数据类对象
        """
        # 第258行：确保token在已知的token列表中
        assert token in self.tokens
        
        # 第259行：判断是否是合成场景
        if token in self.synthetic_scenes:
            # 第260-264行：加载合成场景并获取智能体输入
            return Scene.load_from_disk(
                file_path=self.synthetic_scenes[token][0],  # 合成场景文件路径
                sensor_blobs_path=self._synthetic_sensor_path,  # 合成传感器数据路径
                sensor_config=self._sensor_config,  # 传感器配置
            ).get_agent_input()  # 获取智能体输入
        else:
            # 第266-271行：直接从原始场景字典列表构建智能体输入
            return AgentInput.from_scene_dict_list(
                self.scene_frames_dicts[token],  # 原始场景帧字典列表
                self._original_sensor_path,  # 原始传感器数据路径
                num_history_frames=self._scene_filter.num_history_frames,  # 历史帧数
                sensor_config=self._sensor_config,  # 传感器配置
            )

    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        根据过滤条件收集每个日志文件对应的场景token列表。
        
        :return: 字典，键为日志文件名，值为该日志下的token列表
        """
        # 第279行：初始化字典
        tokens_per_logs: Dict[str, List[str]] = {}
        
        # 第280-285行：遍历原始场景，按日志名分组
        for token, scene_dict_list in self.scene_frames_dicts.items():
            log_name = scene_dict_list[0]["log_name"]  # 从第一帧获取日志名
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(token)  # 日志已存在，追加token
            else:
                tokens_per_logs.update({log_name: [token]})  # 日志不存在，创建新列表

        # 第287-291行：遍历合成场景，同样按日志名分组
        for scene_path, log_name in self.synthetic_scenes.values():
            if tokens_per_logs.get(log_name):
                tokens_per_logs[log_name].append(scene_path.stem)  # 追加文件名作为token
            else:
                tokens_per_logs.update({log_name: [scene_path.stem]})  # 创建新列表

        # 第293行：返回结果字典
        return tokens_per_logs


class MetricCacheLoader:
    """Simple dataloader for metric cache."""

    def __init__(self, cache_path: Path, file_name: str = "metric_cache.pkl"):
        """
        Initializes the metric cache loader.
        :param cache_path: directory of cache folder
        :param file_name: file name of cached files, defaults to "metric_cache.pkl"
        """

        self._file_name = file_name
        self.metric_cache_paths = self._load_metric_cache_paths(cache_path)

    def _load_metric_cache_paths(self, cache_path: Path) -> Dict[str, Path]:
        """
        Helper function to load all cache file paths from folder.
        :param cache_path: directory of cache folder
        :return: dictionary of token and file path
        """
        metadata_dir = cache_path / "metadata"
        metadata_file = [file for file in metadata_dir.iterdir() if ".csv" in str(file)][0]
        with open(str(metadata_file), "r") as f:
            cache_paths = f.read().splitlines()[1:]
        metric_cache_dict = {cache_path.split("/")[-2]: cache_path for cache_path in cache_paths}
        return metric_cache_dict

    @property
    def tokens(self) -> List[str]:
        """
        :return: list of scene identifiers for loading.
        """
        return list(self.metric_cache_paths.keys())

    def __len__(self):
        """
        :return: number for scenes possible to load.
        """
        return len(self.metric_cache_paths)

    def __getitem__(self, idx: int) -> MetricCache:
        """
        :param idx: index of cache to cache to load
        :return: metric cache dataclass
        """
        return self.get_from_token(self.tokens[idx])

    def get_from_token(self, token: str) -> MetricCache:
        """
        Load metric cache from scene identifier
        :param token: unique identifier of scene
        :return: metric cache dataclass
        """
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        return metric_cache

    def to_pickle(self, path: Path) -> None:
        """
        Dumps complete metric cache into pickle.
        :param path: directory of cache folder
        """
        full_metric_cache = {}
        for token in tqdm(self.tokens):
            full_metric_cache[token] = self.get_from_token(token)
        with open(path, "wb") as f:
            pickle.dump(full_metric_cache, f)
