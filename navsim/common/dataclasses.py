from __future__ import annotations

import io
import os
import pickle
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyquaternion import Quaternion

# 尝试导入 nuplan，如果失败则创建占位类
try:
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.abstract_map import AbstractMap
    from nuplan.common.maps.maps_datatypes import TrafficLightStatuses
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
except ImportError:
    class StateSE2:
        def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
            self.x = x
            self.y = y
            self.heading = heading
    
    class AbstractMap:
        pass
    
    class TrafficLightStatuses:
        pass
    
    def get_maps_api(*args, **kwargs):
        return None
    
    MAP_LOCATIONS = []
    
    class LidarPointCloud:
        @staticmethod
        def from_buffer(*args, **kwargs):
            class FakePC:
                points = None
            return FakePC()
    
    class DetectionsTracks:
        pass
    
    class TrajectorySampling:
        def __init__(self, time_horizon=4, interval_length=0.5, num_poses=None):
            self.time_horizon = time_horizon
            self.interval_length = interval_length
            self.num_poses = num_poses if num_poses else int(time_horizon / interval_length) + 1

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None
    image_path: Optional[Path] = None

    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None

    camera_path: Optional[Path] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
    ) -> Cameras:
        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                data_dict[camera_identifier] = Camera(
                    image=np.array(Image.open(image_path)),
                    image_path=image_path,
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                    camera_path=camera_dict[camera_name]["data_path"],
                )
            else:
                data_dict[camera_identifier] = Camera()
        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    lidar_pc: Optional[npt.NDArray[np.float32]] = None
    lidar_path: Optional[Path] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        with open(lidar_path, "rb") as fp:
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        if "lidar_pc" in sensor_names and lidar_path is not None:
            global_lidar_path = sensor_blobs_path / lidar_path
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc, lidar_path)
        return Lidar()


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False


@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]


@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""

    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int

    corresponding_original_scene: str = None
    corresponding_original_initial_token: str = None


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]
    extended_traffic_light_data: Optional[List[TrafficLightStatuses]] = None
    extended_detections_tracks: Optional[List[DetectionsTracks]] = None


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    include_synthetic_scenes: bool = False
    all_mapping: Optional[Dict[Tuple[str, str], List[Tuple[str, str]]]] = None
    synthetic_scene_tokens: Optional[List[str]] = None

    reactive_synthetic_initial_tokens: Optional[List[str]] = None
    non_reactive_synthetic_initial_tokens: Optional[List[str]] = None

    def __post_init__(self):
        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

    @property
    def num_frames(self) -> int:
        return self.num_history_frames + self.num_future_frames


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=include,
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float

    multiplicative_metrics_prod: float
    weighted_metrics: npt.NDArray[np.float64]
    weighted_metrics_array: npt.NDArray[np.float64]

    pdm_score: float

    @classmethod
    def get_empty_results(cls) -> PDMResults:
        return PDMResults(
            no_at_fault_collisions=np.nan,
            drivable_area_compliance=np.nan,
            driving_direction_compliance=np.nan,
            traffic_light_compliance=np.nan,
            ego_progress=np.nan,
            time_to_collision_within_bound=np.nan,
            lane_keeping=np.nan,
            history_comfort=np.nan,
            multiplicative_metrics_prod=np.nan,
            weighted_metrics=np.nan,
            weighted_metrics_array=np.nan,
            pdm_score=np.nan,
        )