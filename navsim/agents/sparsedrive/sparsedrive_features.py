from enum import IntEnum
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig
from pathlib import Path
import pickle
import copy

from PIL import Image
import cv2
import random
import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

from .sparsedrive_config import SparseDriveConfig


class SparseDriveFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for TransFuser."""

    def __init__(self, config: SparseDriveConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "sparsedrive_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """
        camera_keys = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0"]
        camera_features = []
        for cameras in agent_input.cameras:
            camera_feature = {key: {} for key in camera_keys}
            for key in camera_keys:
                camera_data = getattr(cameras, key)
                camera_feature[key]["image_path"] = camera_data.image_path
                camera_feature[key]["sensor2lidar_rotation"] = camera_data.sensor2lidar_rotation
                camera_feature[key]["sensor2lidar_translation"] = camera_data.sensor2lidar_translation
                camera_feature[key]["intrinsics"] = camera_data.intrinsics
                camera_feature[key]["distortion"] = camera_data.distortion
            camera_features.append(camera_feature)

        return camera_features

    def pipeline(self, features, targets, token, test_mode, vis=False):
        camera_features = features["camera_feature"]

        ## only last frame used
        frame_info = camera_features[-1]
        results = self.get_camera_params(frame_info)
        results = self.load_images(results)
        results = self.resize_crop_flip_img(results, test_mode)
        results, targets = self.ego_rotation(results, targets, test_mode)
        results = self.photo_metric_distortion(results, test_mode)
        results = self.normalize_img(results)
        results = self.data_adapter(results)

        features["camera_feature"] = results
        return features, targets, token

    def get_camera_params(self, frame_info):
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsic = []
        cam2lidar_rts = []
        distortions = []
        for cam in self._config.cams:
            cam_info = frame_info[cam]
            ## image path
            image_path = cam_info["image_path"]
            image_paths.append(cam_info["image_path"])
            ## distortion
            distortions.append(cam_info["distortion"])
            ## cam2lidar
            cam2lidar_rt = np.eye(4)
            cam2lidar_rt[:3, :3] = cam_info["sensor2lidar_rotation"]
            cam2lidar_rt[:3, 3] = cam_info["sensor2lidar_translation"]
            cam2lidar_rts.append(cam2lidar_rt)
            ## lidar2cam
            lidar2cam_rt = np.eye(4)
            lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
            lidar2cam_t = (
                cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
            )
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            ## intrinsic
            intrinsic = copy.deepcopy(cam_info["intrinsics"])
            cam_intrinsic.append(intrinsic)
            ## lidar2img
            viewpad = np.eye(4)
            viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt.T
            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt.T)

        results = dict(
            image_paths=image_paths,
            distortions=distortions,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam2lidar=cam2lidar_rts,
            cam_intrinsic=cam_intrinsic,
        )

        return results

    def load_images(self, results):
        image_paths = results["image_paths"]
        imgs = [np.array(Image.open(str(image_path))) for image_path in image_paths]
        if self._config.to_bgr:
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        results["imgs"] = imgs
        results["img_shape"] = [x.shape[:2] for x in imgs]
        return results

    def resize_crop_flip_img(self, results, test_mode):
        H, W = self._config.H, self._config.W
        fH, fW = self._config.final_dim
        if not test_mode:
            resize = np.random.uniform(*self._config.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self._config.bot_pct_lim))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self._config.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self._config.rot_lim)
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self._config.bot_pct_lim)) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
        }

        imgs = results["imgs"]
        lidar2img = results["lidar2img"]
        cam_intrinsic = results["cam_intrinsic"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(
                imgs[i], aug_config,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            lidar2img[i] = mat @ lidar2img[i]
            cam_intrinsic[i][:3, :3] = mat[:3, :3] @ cam_intrinsic[i][:3, :3]

        results["imgs"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        results["lidar2img"] = lidar2img
        results["cam_intrinsic"] = cam_intrinsic

        return results
    
    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array(
                [[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]]
            )
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def ego_rotation(self, results, targets, test_mode):
        if not test_mode:
            angle = np.random.uniform(*self._config.rot3d_range)
        else:
            angle = 0
        
        if angle == 0:
            return results, targets

        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_T_2d = rot_mat[:2, :2].T
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["imgs"])
        lidar2img = results["lidar2img"]
        lidar2cam = results["lidar2cam"]
        for view in range(num_view):
            lidar2img[view] = lidar2img[view] @ rot_mat_inv
            lidar2cam[view] = lidar2cam[view] @ rot_mat_inv
        results["lidar2img"] = lidar2img
        results["lidar2cam"] = lidar2cam

        def wrap_to_pi_half_open(angle):
            """Map angle (rad) to [-pi, pi)."""
            return (angle + np.pi) % (2 * np.pi) - np.pi

        path = targets["path"].numpy()
        path[:, :2] = (path[:, :2] @ rot_mat_T_2d)
        path[:, 2] += angle
        path[:, 2] = wrap_to_pi_half_open(path[:, 2])
        targets["path"] = torch.tensor(path)

        trajectory = targets["trajectory"].numpy()
        trajectory[:, :2] = (trajectory[:, :2] @ rot_mat_T_2d)
        trajectory[:, 2] += angle
        trajectory[:, 2] = wrap_to_pi_half_open(trajectory[:, 2])
        targets["trajectory"] = torch.tensor(trajectory)

        return results, targets
            
    def photo_metric_distortion(self, results, test_mode):
        if test_mode or not self._config.photo_metric_distortion:
            return results

        brightness_delta = 32
        contrast_range = (0.5, 1.5)
        saturation_range = (0.5, 1.5)
        hue_delta = 18
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

        imgs = results["imgs"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if np.random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            if self._config.to_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if np.random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            if self._config.to_bgr:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results["imgs"] = new_imgs
        return results

    def normalize_img(self, results):
        mean = np.array(self._config.img_mean, dtype=np.float32)
        std = np.array(self._config.img_std, dtype=np.float32)

        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))

        imgs = results["imgs"]
        for i in range(len(imgs)):
            img = imgs[i].copy().astype(np.float32)
            cv2.subtract(img, mean, img)  # inplace
            cv2.multiply(img, stdinv, img)  # inplace
            imgs[i] = img

        results["imgs"] = imgs

        return results

    def data_adapter(self, results):
        results.pop("image_paths")
        for key in ['distortions', 'lidar2img', 'lidar2cam', 'cam2lidar', 'cam_intrinsic']:
            results[key] = torch.tensor(np.stack(results[key]))

        imgs = [img.transpose(2, 0, 1) for img in results["imgs"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        imgs = torch.tensor(imgs)
        results["imgs"] = imgs

        results["projection_mat"] = results["lidar2img"].float()
        results["image_wh"] = np.ascontiguousarray(
            np.array(results["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )

        return results


class SparseDriveTargetBuilder(AbstractTargetBuilder):
    """Output target builder for TransFuser."""

    def __init__(self, config: SparseDriveConfig):
        """
        Initializes target builder.
        :param config: global config dataclass of TransFuser
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "sparsedrive_target"

    def compute_targets(self, scene: Scene, cfg: DictConfig) -> Dict[str, torch.Tensor]:
        """
        计算训练目标（标签），包括未来轨迹、路径、路径掩码和速度。
        
        :param scene: Scene数据类对象，包含场景的完整信息（帧数据、元数据、地图等）
        :param cfg: Hydra配置对象，包含数据集路径等配置信息
        :return: 目标字典，包含训练所需的各种标签
        """
        # 1. 从场景中提取未来轨迹（以当前帧自车后轴为原点的局部坐标）
        # trajectory_sampling.num_poses 指定轨迹的点数（默认约为 4s / 0.5s = 8 个点）[num_poses, 3]（x, y, theta）
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )

        # 2. 计算侧向路径（lateral path）和纵向速度（longitudinal velocity）
        # 获取日志文件路径：navsim_log_path/log_name.pkl
        data_path = Path(cfg.navsim_log_path)  # 日志文件根目录
        log_name = scene.scene_metadata.log_name  # 当前场景所属的日志文件名
        initial_token = scene.scene_metadata.initial_token  # 当前场景的初始帧token，历史帧的最后一帧
        
        # 构建完整的日志pickle文件路径
        log_pickle_path = data_path / f"{log_name}.pkl"
        # 加载该日志的所有帧数据，从原始数据获取，因为原始数据可以获得更多的未来数据
        scene_dict_list = pickle.load(open(log_pickle_path, "rb"))
        
        # 遍历帧列表，找到当前场景对应的起始帧
        for idx, scene_dict in enumerate(scene_dict_list):
            token = scene_dict["token"]  # 当前帧的token
            if token != initial_token:  # 跳过直到找到匹配的起始帧
                continue
            
            # 从当前帧开始，计算未来路径和路径掩码
            path, path_mask = self._get_future_path(idx, scene_dict_list)
            
            # 计算纵向速度：通过相邻轨迹点的距离差除以时间间隔
            # pad_trajectory: 在轨迹前添加一个零点，使速度维度与轨迹一致
            pad_trajectory = torch.cat([torch.zeros(1, 2), trajectory[:, :2]], dim=0) #取轨迹的前两列，即x和y
            # 计算相邻帧之间的距离差，再除以时间间隔得到速度
            velocity = torch.norm(pad_trajectory[1:] - pad_trajectory[:-1], dim=-1) / self._config.vel_time_interval
            break  # 找到后立即退出循环

        # 3. 返回目标字典
        return {
            "trajectory": trajectory,      # 未来轨迹 [num_poses, 3] (x, y, heading)
            "path": path,                  # 未来路径点 [len_path, 3] (x, y, heading)
            "path_mask": path_mask,        # 路径掩码 [len_path]，标记有效路径点
            "velocity": velocity,          # 纵向速度 [num_poses]
        }

    def _get_future_path(self, idx, scene_dict_list):
        """
        计算未来路径点（等间距采样）。
        
        与 trajectory 不同，path 是按**距离间隔**均匀采样的，而非时间间隔。
        这更适合路径规划任务，因为路径规划关注的是空间上的均匀分布。
        
        :param idx: 当前帧在日志序列中的索引
        :param scene_dict_list: 完整的日志帧字典列表
        :return: (path, path_mask) - 路径点数组和有效掩码
        """
        # 从配置获取路径参数
        num_pts = self._config.len_path       # 路径点数量（如 50）
        interval = self._config.path_interval  # 路径点间隔（如 0.5 米）
        
        # 初始化变量
        global_ego_poses = []      # 存储全局坐标系下的自车位姿序列
        distances = [0.0]          # 存储相邻帧之间的距离（第一个元素为0）
        accumulated_distance = 0.0 # 累计行驶距离
        max_dis = num_pts * interval  # 需要的最大距离
        
        # 1. 从当前帧开始，收集足够的全局位姿数据
        for frame_idx in range(idx, len(scene_dict_list)):
            scene_frame = scene_dict_list[frame_idx]  # 获取当前帧数据
            
            # 提取自车在全局坐标系下的位姿
            ego_translation = scene_frame["ego2global_translation"]  # 平移向量 [x, y, z]
            ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])  # 四元数
            ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )  # 转换为 [x, y, heading] 格式
            
            # 计算与上一帧的距离（从第二帧开始）
            if global_ego_poses:
                prev_pose = global_ego_poses[-1]
                distance = np.linalg.norm(ego_pose[:2] - prev_pose[:2])  # 只计算 xy 平面距离
                distances.append(distance)
                accumulated_distance += distance
            
            # 添加到位姿列表
            global_ego_poses.append(ego_pose)
            
            # 如果累计距离足够，停止收集
            if accumulated_distance > max_dis:
                break
        
        # 2. 将全局坐标系转换为局部坐标系（以当前帧自车为原点）
        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]),  # 当前帧位姿作为局部坐标系原点
            np.array(global_ego_poses, dtype=np.float64)
        )
        
        # 3. 等间距采样：通过线性插值生成均匀分布的路径点
        distances = np.cumsum(distances)  # 累计距离数组
        target_distance = np.arange(1, (num_pts + 1), 1) * interval  # 目标采样距离 [interval, 2*interval, ...]
        
        # 使用线性插值在距离维度上采样
        path = np.array(
            [
                np.interp(target_distance, distances, local_ego_poses[:, 0]),  # x 坐标插值
                np.interp(target_distance, distances, local_ego_poses[:, 1]),  # y 坐标插值
                np.interp(target_distance, distances, local_ego_poses[:, 2]),  # heading 插值
            ]
        ).T  # 转置后形状为 [num_pts, 3]
        
        # 4. 将航向角限制在 [-pi, pi) 范围内
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # 5. 构建路径掩码：标记有效路径点
        path_mask = np.ones(num_pts, dtype=np.float32)  # 默认所有点都有效
        # 计算实际有效的路径点数量（考虑日志数据不足的情况）
        valid_points = min(num_pts, int(np.floor(accumulated_distance / interval)))
        path_mask[valid_points:] = 0  # 超出范围的点标记为无效
        
        # 返回路径点和掩码
        return torch.tensor(path), torch.tensor(path_mask)