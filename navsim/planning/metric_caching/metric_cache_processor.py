import pathlib                                                     # 路径处理模块，用于文件路径操作
from typing import Any, Dict, List, Optional, Tuple                 # 类型注解，提高代码可读性和类型检查

import numpy as np                                                  # 数值计算库，用于数组和矩阵操作
from nuplan.common.actor_state.agent import Agent                   # 智能体类，代表动态障碍物（车辆、行人等）
from nuplan.common.actor_state.oriented_box import OrientedBox      # 有向包围盒类，用于表示障碍物的位置和形状
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D  # 状态表示类，SE2位姿和2D向量
from nuplan.common.actor_state.static_object import StaticObject     # 静态对象类，代表静态障碍物（建筑物等）
from nuplan.common.actor_state.tracked_objects import TrackedObjects  # 跟踪对象集合类
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES  # 智能体类型枚举（车辆、行人等）
from nuplan.common.geometry.convert import absolute_to_relative_poses  # 绝对坐标转相对坐标的函数
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject  # 地图对象类
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData  # 地图数据类型和交通灯状态
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario  # 场景抽象基类
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer  # 仿真历史缓冲区
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # 检测跟踪观测类型
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput  # 规划器初始化和输入类
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration  # 仿真迭代类
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory  # 插值轨迹类
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # 轨迹采样配置类
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry  # 缓存元数据条目类

from navsim.common.dataclasses import Trajectory                     # 轨迹数据类
from navsim.common.enums import SceneFrameType                       # 场景类型枚举（原始/合成）
from navsim.planning.metric_caching.metric_cache import MapParameters, MetricCache  # 指标缓存相关类
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator  # 状态插值工具类
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario  # 导航场景类
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation  # PDM观测类
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner  # PDM闭环规划器
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy  # 批量IDM策略


class MetricCacheProcessor:
    """用于在NAVSIM中创建指标缓存的类。"""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        proposal_sampling: TrajectorySampling,
    ):
        """
        初始化类。
        :param cache_path: 缓存保存路径，如果为None则不缓存。
        :param force_feature_computation: 如果为True，即使缓存已存在也会重新计算。
        :param proposal_sampling: 轨迹采样配置。
        """
        # 将缓存路径转换为Path对象（如果提供）
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        # 是否强制重新计算缓存
        self._force_feature_computation = force_feature_computation

        # 为TTC指标额外增加1秒的观测时间
        future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
        # 创建扩展的轨迹采样配置
        future_sampling = TrajectorySampling(num_poses=future_poses, interval_length=proposal_sampling.interval_length)
        # 保存原始的提议采样配置
        self._proposal_sampling = proposal_sampling
        # 地图查询半径（米）
        self._map_radius = 100

        # 创建PDM闭环规划器实例，用于生成基准轨迹
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=future_sampling,                       # 轨迹采样配置
            proposal_sampling=self._proposal_sampling,                  # 提议采样配置
            idm_policies=BatchIDMPolicy(                               # 批量IDM策略，生成多个候选轨迹
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],       # 速度限制比例（5个不同速度）
                fallback_target_velocity=15.0,                         # 后备目标速度（m/s）
                min_gap_to_lead_agent=1.0,                             # 与前车最小距离（米）
                headway_time=1.5,                                      # 车头时距（秒）
                accel_max=1.5,                                         # 最大加速度（m/s²）
                decel_max=3.0,                                         # 最大减速度（m/s²）
            ),
            lateral_offsets=[-1.0, 1.0],                               # 横向偏移（米），生成居中、左偏、右偏轨迹
            map_radius=self._map_radius,                               # 地图查询半径
        )

    def _get_planner_inputs(self, scenario: AbstractScenario) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        从场景对象创建规划器输入参数。
        :param scenario: nuPlan框架的场景对象
        :return: 规划器输入和初始化对象的元组
        """

        # 初始化规划器，包含路线信息、任务目标和地图API
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),    # 获取路线的roadblock ID列表
            mission_goal=scenario.get_mission_goal(),                  # 获取任务目标
            map_api=scenario.map_api,                                  # 获取地图API接口
        )

        # 创建仿真历史缓冲区，包含初始自车状态和初始跟踪对象
        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,                                             # 缓冲区大小为1（只保留当前状态）
            ego_states=[scenario.initial_ego_state],                   # 初始自车状态
            observations=[scenario.initial_tracked_objects],           # 初始跟踪对象
        )

        # 创建规划器输入对象
        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),  # 当前仿真迭代（第0帧）
            history=history,                                           # 仿真历史缓冲区
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),  # 当前交通灯状态
        )

        # 返回规划器输入和初始化对象
        return planner_input, planner_initialization

    def _interpolate_gt_observation(self, scenario: NavSimScenario) -> List[DetectionsTracks]:
        """
        将检测跟踪数据插值到更高时间分辨率的辅助函数。
        :param scenario: nuPlan框架的场景接口
        :return: 插值后的检测跟踪数据列表
        """

        # 状态向量维度：(时间, x坐标, y坐标, 航向角, x速度, y速度)
        state_size = 6

        # 获取时间范围（提议采样的时间范围）
        time_horizon = self._proposal_sampling.time_horizon
        # 原始采样间隔（2Hz，每0.5秒一个点）
        resolution_step = 0.5
        # 目标插值间隔（提议采样的间隔，通常为0.1秒，即10Hz）
        interpolate_step = self._proposal_sampling.interval_length

        # 获取场景的数据库间隔（原始数据的时间间隔）
        scenario_step = scenario.database_interval

        # 以2Hz采样检测跟踪数据，生成时间戳数组
        relative_time_s = np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step

        # 计算原始数据的索引，按resolution_step间隔采样
        gt_indices = np.arange(
            0,
            int(time_horizon / scenario_step) + 1,
            int(resolution_step / scenario_step),
        )
        # 获取原始检测跟踪数据（每0.5秒一个）
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration) for iteration in gt_indices
        ]

        # 存储每个跟踪对象的状态序列
        detection_tracks_states: Dict[str, Any] = {}
        # 存储每个跟踪对象的初始信息（用于重建对象）
        unique_detection_tracks: Dict[str, Any] = {}

        # 遍历原始检测跟踪数据
        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            # 遍历当前帧的所有跟踪对象
            for tracked_object in detection_track.tracked_objects:
                # 获取对象的唯一标识
                token = tracked_object.track_token

                # 初始化状态向量（时间, x, y, heading, velo_x, velo_y）
                tracked_state = np.zeros(state_size, dtype=np.float64)
                # 填充位置和时间信息
                tracked_state[:4] = (
                    time_s,                                             # 时间戳
                    tracked_object.center.x,                            # x坐标
                    tracked_object.center.y,                            # y坐标
                    tracked_object.center.heading,                      # 航向角
                )

                # 如果是动态智能体（车辆、行人等），额外提取速度信息
                if tracked_object.tracked_object_type in AGENT_TYPES:
                    tracked_state[4:] = (
                        tracked_object.velocity.x,                     # x方向速度
                        tracked_object.velocity.y,                     # y方向速度
                    )

                # 如果是新发现的对象，初始化状态列表
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # 如果对象已存在，追加新状态
                else:
                    detection_tracks_states[token].append(tracked_state)

        # 为每个跟踪对象创建时间插值器
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            # 将状态列表转换为numpy数组
            states = np.array(states_list, dtype=np.float64)
            # 创建状态插值器
            detection_interpolators[token] = StateInterpolator(states)

        # 以10Hz生成插值后的时间戳数组
        interpolated_time_s = np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float) * interpolate_step

        # 存储插值后的检测跟踪数据
        interpolated_detection_tracks = []
        # 遍历每个目标时间戳
        for time_s in interpolated_time_s:
            # 存储当前时间戳的所有插值对象
            interpolated_tracks = []
            # 遍历每个跟踪对象的插值器
            for token, interpolator in detection_interpolators.items():
                # 获取原始对象信息
                initial_detection_track = unique_detection_tracks[token]
                # 在目标时间戳处插值
                interpolated_state = interpolator.interpolate(time_s)

                # 如果只有一个时间点的数据，直接使用原始对象
                if interpolator.start_time == interpolator.end_time:
                    interpolated_tracks.append(initial_detection_track)

                # 如果插值成功
                elif interpolated_state is not None:

                    # 获取对象类型和元数据
                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = initial_detection_track.metadata  # 复制元数据

                    # 使用插值后的状态创建有向包围盒
                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),             # 位置和航向角
                        initial_detection_track.box.length,            # 长度（保持不变）
                        initial_detection_track.box.width,             # 宽度（保持不变）
                        initial_detection_track.box.height,            # 高度（保持不变）
                    )

                    # 如果是动态智能体，创建Agent对象
                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])  # 速度向量

                        detection_track = Agent(
                            tracked_object_type=tracked_type,          # 对象类型
                            oriented_box=oriented_box,                 # 包围盒
                            velocity=velocity,                         # 速度
                            metadata=initial_detection_track.metadata, # 元数据
                        )
                    # 如果是静态对象，创建StaticObject对象
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,          # 对象类型
                            oriented_box=oriented_box,                 # 包围盒
                            metadata=metadata,                         # 元数据
                        )

                    # 添加到当前时间戳的插值列表
                    interpolated_tracks.append(detection_track)
            # 将当前时间戳的所有对象打包为DetectionsTracks
            interpolated_detection_tracks.append(DetectionsTracks(TrackedObjects(interpolated_tracks)))
        # 返回插值后的检测跟踪数据列表
        return interpolated_detection_tracks

    def _build_pdm_observation(
        self,
        interpolated_detection_tracks: List[DetectionsTracks],
        interpolated_traffic_light_data: List[List[TrafficLightStatusData]],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        """
        构建PDM观测对象。
        :param interpolated_detection_tracks: 插值后的检测跟踪数据
        :param interpolated_traffic_light_data: 插值后的交通灯状态
        :param route_lane_dict: 路线车道字典
        :return: PDM观测对象
        """
        # 创建PDM观测对象
        pdm_observation = PDMObservation(
            self._proposal_sampling,                                   # 轨迹采样配置
            self._proposal_sampling,                                   # 观测采样配置（与轨迹相同）
            self._map_radius,                                          # 地图查询半径
            observation_sample_res=1,                                  # 观测采样分辨率
            extend_observation_for_ttc=False,                          # 是否为TTC扩展观测
        )
        # 更新PDM观测对象的检测跟踪和交通灯数据
        pdm_observation.update_detections_tracks(
            interpolated_detection_tracks,                             # 插值后的检测跟踪数据
            interpolated_traffic_light_data,                           # 插值后的交通灯状态
            route_lane_dict,                                           # 路线车道字典
            compute_traffic_light_data=True,                           # 是否计算交通灯数据
        )
        # 返回构建好的PDM观测对象
        return pdm_observation

    def _interpolate_traffic_light_status(self, scenario: NavSimScenario) -> List[List[TrafficLightStatusData]]:
        """
        将交通灯状态插值到更高时间分辨率。
        :param scenario: nuPlan框架的场景接口
        :return: 插值后的交通灯状态列表
        """

        # 获取时间范围和插值间隔
        time_horizon = self._proposal_sampling.time_horizon
        interpolate_step = self._proposal_sampling.interval_length

        # 获取场景的数据库间隔
        scenario_step = scenario.database_interval
        # 计算原始数据的索引
        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, 1, dtype=int)

        # 存储插值后的交通灯状态
        traffic_light_status = []
        # 遍历每个原始索引
        for iteration in gt_indices:
            # 获取当前帧的交通灯状态
            current_status_list = list(scenario.get_traffic_light_status_at_iteration(iteration=iteration))
            # 将当前状态重复多次，以匹配插值后的时间分辨率
            for _ in range(int(scenario_step / interpolate_step)):
                traffic_light_status.append(current_status_list)

        # 如果原始间隔等于插值间隔，直接返回
        if scenario_step == interpolate_step:
            return traffic_light_status
        # 否则，截断多余的状态（避免越界）
        else:
            return traffic_light_status[: -int(scenario_step / interpolate_step) + 1]

    def _load_route_dicts(
        self, scenario: NavSimScenario, route_roadblock_ids: List[str]
    ) -> Tuple[Dict[str, RoadBlockGraphEdgeMapObject], Dict[str, LaneGraphEdgeMapObject]]:
        """
        加载路线相关的字典（roadblock和lane）。
        :param scenario: 导航场景对象
        :param route_roadblock_ids: 路线的roadblock ID列表
        :return: roadblock字典和lane字典的元组
        """
        # 去重，保持顺序
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        # 初始化roadblock和lane字典
        route_roadblock_dict = {}
        route_lane_dict = {}

        # 遍历每个roadblock ID
        for id_ in route_roadblock_ids:
            # 从地图API获取roadblock对象（先尝试ROADBLOCK层）
            block = scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            # 如果找不到，尝试ROADBLOCK_CONNECTOR层
            block = block or scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            # 将roadblock添加到字典
            route_roadblock_dict[block.id] = block

            # 遍历roadblock内的所有车道
            for lane in block.interior_edges:
                # 将车道添加到字典
                route_lane_dict[lane.id] = lane

        # 返回roadblock字典和lane字典
        return route_roadblock_dict, route_lane_dict

    def _build_file_path(self, scenario: NavSimScenario) -> pathlib.Path:
        """
        构建缓存文件的保存路径。
        :param scenario: 导航场景对象
        :return: 缓存文件路径
        """
        return (
            # 路径格式：缓存根目录/日志名/场景类型/token/metric_cache.pkl
            (self._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl")
            if self._cache_path  # 如果缓存路径不为None
            else None            # 否则返回None
        )

    def compute_and_save_metric_cache(self, scenario: NavSimScenario) -> Optional[CacheMetadataEntry]:
        """
        计算并保存指标缓存。
        :param scenario: 导航场景对象
        :return: 缓存元数据条目（成功）或None（失败）
        """
        # 构建缓存文件路径
        file_name = self._build_file_path(scenario)
        # 断言检查缓存路径不能为None
        assert file_name is not None, "Cache path can not be None for saving cache."
        # 如果缓存文件已存在且不需要强制重新计算，直接返回元数据条目
        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)
        # 计算指标缓存
        metric_cache = self.compute_metric_cache(scenario)
        # 将缓存保存到磁盘
        metric_cache.dump()
        # 返回缓存元数据条目
        return CacheMetadataEntry(metric_cache.file_path)

    def _extract_ego_future_trajectory(self, scenario: NavSimScenario) -> Trajectory:
        """
        从场景中提取自车的未来轨迹（人类驾驶轨迹）。
        :param scenario: 导航场景对象
        :return: 自车未来轨迹
        """
        # 创建轨迹采样配置（使用场景的数据库间隔）
        ego_trajectory_sampling = TrajectorySampling(
            time_horizon=self._proposal_sampling.time_horizon,          # 时间范围
            interval_length=scenario.database_interval,                  # 采样间隔（场景原始间隔）
        )
        # 获取自车未来轨迹的状态序列
        future_ego_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0,                                            # 从第0帧开始
                time_horizon=ego_trajectory_sampling.time_horizon,      # 时间范围
                num_samples=ego_trajectory_sampling.num_poses,          # 采样数量
            )
        )
        # 获取初始自车状态
        initial_ego_state = scenario.get_ego_state_at_iteration(0)
        # 检查nuPlan和navsim的行为差异：nuPlan不返回初始状态，而navsim返回
        if future_ego_states[0].time_point != initial_ego_state.time_point:
            # 需要在转换为相对坐标之前添加初始状态
            future_ego_states = [initial_ego_state] + future_ego_states

        # 提取所有未来状态的后轴位置
        future_ego_poses = [state.rear_axle for state in future_ego_states]
        # 将绝对坐标转换为相对坐标（以第一帧为原点），并去掉第一帧（原点）
        relative_future_states = absolute_to_relative_poses(future_ego_poses)[1:]
        # 创建Trajectory对象返回
        return Trajectory(
            # 将位姿转换为numpy数组（x, y, heading）
            poses=np.array([[pose.x, pose.y, pose.heading] for pose in relative_future_states]),
            trajectory_sampling=ego_trajectory_sampling,                 # 轨迹采样配置
        )

    def compute_metric_cache(self, scenario: NavSimScenario) -> MetricCache:
        """
        计算指标缓存的主函数。
        :param scenario: 导航场景对象
        :return: 指标缓存对象
        """
        # 构建缓存文件路径
        file_name = self._build_file_path(scenario)

        # 通过token长度判断是否为合成场景（合成场景token长度为17）
        is_synthetic_scene = len(scenario.token) == 17

        # 初始化并运行PDM-Closed规划器，生成基准轨迹
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)

        # 加载路线相关的roadblock和lane字典
        route_roadblock_dict, route_lane_dict = self._load_route_dicts(
            scenario, planner_initialization.route_roadblock_ids
        )

        # 将检测跟踪数据插值到10Hz
        interpolated_detection_tracks = self._interpolate_gt_observation(scenario)
        # 将交通灯状态插值到目标分辨率
        interpolated_traffic_light_status = self._interpolate_traffic_light_status(scenario)

        # 构建PDM观测对象
        observation = self._build_pdm_observation(
            interpolated_detection_tracks=interpolated_detection_tracks,
            interpolated_traffic_light_data=interpolated_traffic_light_status,
            route_lane_dict=route_lane_dict,
        )
        # 提取未来的跟踪对象（从第2帧开始，去掉当前帧）
        future_tracked_objects = interpolated_detection_tracks[1:]

        # 提取过去1.5秒的人类驾驶轨迹
        past_human_trajectory = InterpolatedTrajectory(
            [ego_state for ego_state in scenario.get_ego_past_trajectory(0, 1.5)]
        )

        # 如果不是合成场景，提取人类驾驶的未来轨迹
        if not is_synthetic_scene:
            human_trajectory = self._extract_ego_future_trajectory(scenario)
        # 合成场景没有人类驾驶轨迹
        else:
            human_trajectory = None

        # 创建并返回MetricCache对象
        return MetricCache(
            file_path=file_name,                                       # 缓存文件路径
            log_name=scenario.log_name,                                # 日志名称
            scene_type=SceneFrameType.SYNTHETIC if is_synthetic_scene else SceneFrameType.ORIGINAL,  # 场景类型
            timepoint=scenario.start_time,                             # 起始时间点
            trajectory=pdm_closed_trajectory,                          # PDM-Closed规划器生成的基准轨迹
            human_trajectory=human_trajectory,                         # 人类驾驶的地面真值轨迹
            past_human_trajectory=past_human_trajectory,               # 过去的人类驾驶轨迹
            ego_state=scenario.initial_ego_state,                      # 自车初始状态
            observation=observation,                                    # PDM观测对象（包含障碍物和交通灯）
            centerline=self._pdm_closed._centerline,                   # 道路中心线
            route_lane_ids=list(self._pdm_closed._route_lane_dict.keys()),  # 路线车道ID列表
            drivable_area_map=self._pdm_closed._drivable_area_map,     # 可行驶区域地图
            past_detections_tracks=[                                   # 过去的检测跟踪数据（去掉最后一个，避免重复）
                dt for dt in scenario.get_past_tracked_objects(iteration=0, time_horizon=1.5, num_samples=3)
            ][:-1],
            current_tracked_objects=[scenario.initial_tracked_objects],  # 当前帧的跟踪对象
            future_tracked_objects=future_tracked_objects,              # 未来的检测跟踪数据（插值后）
            map_parameters=MapParameters(                              # 地图参数
                map_root=scenario.map_root,                            # 地图根目录
                map_version=scenario.map_version,                      # 地图版本
                map_name=scenario.map_api.map_name,                    # 地图名称
            ),
        )