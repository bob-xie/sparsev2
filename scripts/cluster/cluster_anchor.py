import os  # 标准库：操作系统相关功能

# 设置BLAS/MKL多线程环境变量为1，避免多线程冲突（用于NumPy/SciPy的底层线性代数库）
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS线程数限制
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP线程数限制
os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL线程数限制

# 并发执行库：ThreadPoolExecutor用于并行加载数据，as_completed获取已完成的任务
from concurrent.futures import ThreadPoolExecutor, as_completed

# typing模块：提供类型注解支持
from typing import Callable, List, Optional, Tuple

# 绘图库：用于可视化聚类结果
import matplotlib.pyplot as plt

# NumPy：数值计算核心库
import numpy as np

# SciKit-Learn：机器学习库，KMeans用于轨迹聚类
from sklearn.cluster import KMeans

# tqdm：进度条库，显示数据加载进度
from tqdm import tqdm

# 从navsim项目中导入数据加载函数
from navsim.planning.training.dataset import load_feature_target_from_pickle

# ==================== 聚类数量配置 ====================
K_PATH = 1024  # 路径聚类的数量，生成1024条代表性路径
K_VELOCITY = 256  # 速度聚类的数量，生成256种代表性速度序列

# ==================== 路径配置 ====================
CACHE_PATH = "exp/data_cache_navtrain"  # 训练数据缓存目录路径
VIS_DIR = "vis"  # 可视化图像输出目录
CKPT_DIR = "ckpt/kmeans"  # 聚类结果（词汇表）保存目录
DT = 0.5  # 时间间隔（秒），与轨迹采样间隔一致，每0.5秒一个速度值

# 获取所有日志文件夹名称
LOG_NAMES = os.listdir(CACHE_PATH)


def interp1d_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    带外推的一维线性插值函数

    参数：
        x: 目标插值点的横坐标数组（可能超出xp的范围）
        xp: 已知数据点的横坐标数组（必须单调递增）
        fp: 已知数据点的纵坐标数组

    返回：
        y: 在x处的插值结果，超出xp范围时使用线性外推
    """
    # 将输入转换为float类型的NumPy数组
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    # 使用NumPy的标准线性插值（仅在xp范围内有效）
    y = np.interp(x, xp, fp)

    # ==================== 左侧外推处理 ====================
    # 计算xp起点处的斜率（用于外推）
    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    # 找出x中小于xp最小值的点（需要左侧外推）
    left_mask = x < xp[0]
    # 线性外推公式：y = y0 + m * (x - x0)
    y[left_mask] = fp[0] + m_left * (x[left_mask] - xp[0])

    # ==================== 右侧外推处理 ====================
    # 计算xp终点处的斜率（用于外推）
    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    # 找出x中大于xp最大值的点（需要右侧外推）
    right_mask = x > xp[-1]
    # 线性外推公式：y = y_last + m * (x - x_last)
    y[right_mask] = fp[-1] + m_right * (x[right_mask] - xp[-1])

    return y  # 返回插值（带外推）结果


def interp_trajectory(
    path_cluster: np.ndarray,
    velocity_cluster: np.ndarray,
    interp_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = np.interp,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将路径和速度组合，通过插值生成完整的轨迹词汇表

    参数：
        path_cluster: 路径聚类中心，形状 [K_PATH, 50, 3]
                     K_PATH=1024条路径，每条路径50个点(x,y,heading)
        velocity_cluster: 速度聚类中心，形状 [K_VELOCITY, 8]
                        K_VELOCITY=256种速度，每种速度序列8个时间步
        interp_func: 插值函数，默认np.interp，可传入带外推的interp1d_extrap

    返回：
        trajectory: 组合后的完整轨迹，形状 [K_PATH, K_VELOCITY, 8, 3]
                   表示1024×256=262144种路径-速度组合，每种组合8个轨迹点
        trajectory_mask: 轨迹有效性掩码，形状 [K_PATH, K_VELOCITY, 8]
                       标记每种组合中哪些轨迹点是有效的（1=有效，0=无效）
    """
    num_velocity = velocity_cluster.shape[1]  # 获取速度序列长度，默认=8

    # 初始化输出数组
    trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))  # [1024, 256, 8, 3]
    trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))  # [1024, 256, 8]，初始全1（有效）

    # 遍历所有路径-速度组合
    for i in range(K_PATH):  # 遍历1024条路径
        for j in range(K_VELOCITY):  # 遍历256种速度
            # 获取当前路径和速度
            path = path_cluster[i]  # 当前路径，形状 [50, 3]
            velocity = velocity_cluster[j]  # 当前速度序列，形状 [8]

            # ==================== 计算目标行驶距离 ====================
            # velocity: [8] 个速度值（m/s），每个速度维持DT=0.5秒
            # np.cumsum: 累计求和，得到每个时间步结束时的总行驶距离
            # 例如：velocity=[10,10,10,10,10,10,10,10] m/s
            #      target_distance=[5,10,15,20,25,30,35,40] 米
            target_distance = np.cumsum(velocity * DT, axis=0)

            # ==================== 计算路径的累计距离 ====================
            # 在路径起点添加一个[0,0,0]，便于计算距离
            pad_path = np.concatenate([np.zeros((1, 3)), path], axis=0)  # 形状 [51, 3]

            # 计算相邻路径点在x,y平面上的距离（欧氏距离）
            # 只考虑x,y坐标，不考虑heading
            segment_distances = np.linalg.norm(pad_path[1:, :2] - pad_path[:-1, :2], axis=-1)
            # 对距离进行累计求和，得到每个路径点的累计距离
            distance = np.concatenate([np.zeros((1,)), segment_distances.cumsum(axis=0)])  # 形状 [51]

            # ==================== 插值生成轨迹点 ====================
            # 根据目标距离在路径上进行插值，得到每个时间步的(x,y,heading)
            interp_traj = np.array(
                [
                    # 在path[:,0]（x坐标）上插值
                    interp_func(target_distance, distance, pad_path[:, 0]),
                    # 在path[:,1]（y坐标）上插值
                    interp_func(target_distance, distance, pad_path[:, 1]),
                    # 在path[:,2]（heading坐标）上插值
                    interp_func(target_distance, distance, pad_path[:, 2]),
                ]
            ).T  # 转置，形状变为 [8, 3]

            # 将heading标准化到 [-π, π) 范围内
            interp_traj[:, 2] = (interp_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi

            # 保存插值后的轨迹
            trajectory[i, j] = interp_traj  # 形状 [8, 3]

            # ==================== 计算轨迹有效性掩码 ====================
            max_dist = distance[-1]  # 路径总长度（最后一个累计距离）
            # 判断每个目标距离是否在路径范围内
            # 例如：max_dist=50米，target_distance=[5,10,15,20,25,30,35,40]
            #      valid=[True,True,True,True,True,True,True,True]（假设路径足够长）
            valid = target_distance <= max_dist
            # 将超出路径范围的轨迹点标记为无效（0）
            trajectory_mask[i, j, ~valid] = 0.0
            #轨迹进行了外推，trajectory_mask没有外推，根据原始max_dist判断是否超出路径范围
    return trajectory, trajectory_mask


def load_one(cache_path: str, log_name: str, token: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    加载单个场景的路径和速度数据

    参数：
        cache_path: 缓存目录路径
        log_name: 日志文件夹名称
        token: 场景唯一标识符

    返回：
        path: 路径数组，形状 [50, 3]，如果路径不完整则返回None
        velocity: 速度数组，形状 [8]
    """
    # 构建缓存文件的完整路径
    # 路径格式：cache_path/log_name/token/sparsedrive_target.gz
    data_path = os.path.join(cache_path, log_name, token, "sparsedrive_target.gz")
    # 从压缩的pickle文件加载目标数据
    data = load_feature_target_from_pickle(data_path)
    # 确保数据中包含path字段
    assert "path" in data

    # ==================== 路径数据处理 ====================
    # 检查路径掩码是否全部为True（表示路径完整）
    if data["path_mask"].all():
        # 深拷贝路径数据，避免修改原始缓存
        path = np.array(data["path"], copy=True)
        # 将heading标准化到 [-π, π) 范围内
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
    else:
        # 路径不完整（如日志数据不足），返回None
        path = None

    # ==================== 速度数据处理 ====================
    # 直接从缓存数据中获取速度序列
    velocity = data["velocity"]  # 形状 [8]

    return path, velocity


def load_all_parallel(
    cache_path: str, log_names: List[str], max_workers: int = 64
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    使用多线程并行加载所有场景的路径和速度数据

    参数：
        cache_path: 缓存目录路径
        log_names: 日志文件夹名称列表
        max_workers: 最大并行线程数

    返回：
        paths: 路径列表，每个元素形状 [50, 3]
        velocities: 速度列表，每个元素形状 [8]
    """
    paths: List[np.ndarray] = []  # 存储所有有效路径
    velocities: List[np.ndarray] = []  # 存储所有速度序列
    tasks = []  # 存储加载任务元组(log_name, token)

    # ==================== 构建任务列表 ====================
    # 遍历所有日志文件夹
    for log_name in log_names:
        # 获取该日志下的所有场景token
        tokens = os.listdir(os.path.join(cache_path, log_name))
        # 为每个token创建加载任务
        for token in tokens:
            tasks.append((log_name, token))

    # ==================== 并行加载数据 ====================
    # 创建线程池执行器
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有加载任务
        futures = [
            executor.submit(load_one, cache_path, log_name, token)
            for (log_name, token) in tasks
        ]

        # 遍历已完成的任务
        for fut in tqdm(as_completed(futures), total=len(futures)):
            # 获取加载结果
            path, velocity = fut.result()
            # 只保留完整路径的场景
            if path is not None:
                paths.append(path)
            # 速度序列始终保留（即使路径不完整）
            velocities.append(velocity)

    return paths, velocities


def visualize(path_cluster: np.ndarray, velocity_cluster: np.ndarray, trajectory: np.ndarray) -> None:
    """
    可视化路径、速度和轨迹的聚类结果，保存为PNG图像

    参数：
        path_cluster: 路径聚类中心，形状 [1024, 50, 3]
        velocity_cluster: 速度聚类中心，形状 [256, 8]
        trajectory: 组合轨迹，形状 [1024, 256, 8, 3]
    """
    # 创建可视化输出目录
    os.makedirs(VIS_DIR, exist_ok=True)

    # ==================== 可视化路径聚类 ====================
    # 绘制所有1024条路径
    for j in range(K_PATH):
        # 绘制路径的x-y坐标
        plt.plot(path_cluster[j, :, 0], path_cluster[j, :, 1])
    # 保存路径图像
    plt.savefig(f"{VIS_DIR}/path_{K_PATH}.png", bbox_inches="tight")
    plt.close()  # 关闭当前图像，释放内存

    # ==================== 可视化速度聚类 ====================
    num_velocity = velocity_cluster.shape[1]  # 速度序列长度，默认8
    # 使用Spectral色谱生成颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, num_velocity))
    x = np.arange(K_VELOCITY)  # x轴：0到255的速度序列索引
    plt.figure(figsize=(10, 4))  # 创建新图像

    # 绘制第一个时间步的速度柱状图
    plt.bar(x, velocity_cluster[:, 0], color=colors[0], label="0-0.5 s")
    bottom = velocity_cluster[:, 0].copy()  # 记录当前底部位置

    # 堆叠绘制其余时间步的速度
    for i in range(1, num_velocity):
        plt.bar(
            x,
            velocity_cluster[:, i],  # 当前时间步的速度
            bottom=bottom,  # 堆叠在 previous 时间步之上
            color=colors[i],  # 使用不同颜色
            label=f"{i * DT:.1f}-{(i + 1) * DT:.1f} s",  # 图例标签
        )
        bottom += velocity_cluster[:, i]  # 更新底部位置

    # 设置坐标轴标签和标题
    plt.xlabel("sequence index")  # x轴：速度序列索引
    plt.ylabel("speed (m/s)")  # y轴：速度(m/s)
    plt.title("Stacked speed histogram")  # 标题：堆叠速度直方图
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")  # 图例放在右侧
    plt.tight_layout()  # 调整布局
    # 保存速度图像
    plt.savefig(f"{VIS_DIR}/velocity_{K_VELOCITY}.png", bbox_inches="tight")
    plt.close()

    # ==================== 可视化轨迹 ====================
    # 绘制所有路径-速度组合的轨迹
    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            plt.plot(trajectory[i, j, :, 0], trajectory[i, j, :, 1])
    # 保存轨迹图像
    plt.savefig(f"{VIS_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.png", bbox_inches="tight")
    plt.close()


def save_outputs(
    path_cluster: np.ndarray,
    velocity_cluster: np.ndarray,
    trajectory: np.ndarray,
    trajectory_mask: np.ndarray,
) -> None:
    """
    将聚类生成的词汇表保存到文件

    参数：
        path_cluster: 路径聚类中心，形状 [1024, 50, 3]
        velocity_cluster: 速度聚类中心，形状 [256, 8]
        trajectory: 组合轨迹，形状 [1024, 256, 8, 3]
        trajectory_mask: 轨迹掩码，形状 [1024, 256, 8]
    """
    # 创建输出目录（如果不存在）
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 保存路径词汇表为 .npy 文件
    np.save(f"{CKPT_DIR}/path_{K_PATH}.npy", path_cluster)
    # 保存速度词汇表为 .npy 文件
    np.save(f"{CKPT_DIR}/velocity_{K_VELOCITY}.npy", velocity_cluster)
    # 使用npz格式保存轨迹和掩码（压缩格式）
    np.savez(
        f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
        trajectory=trajectory,  # 轨迹数组
        trajectory_mask=trajectory_mask,  # 掩码数组
    )


def main():
    """
    主函数：执行完整的词汇表生成流程

    流程：
    1. 并行加载所有训练数据
    2. 对路径进行K-Means聚类
    3. 对速度进行K-Means聚类
    4. 组合路径和速度生成轨迹词汇表
    5. 可视化结果
    6. 保存词汇表
    """
    # ==================== 步骤1: 加载训练数据 ====================
    # 并行加载所有场景的路径和速度数据
    paths, velocities = load_all_parallel(CACHE_PATH, LOG_NAMES, max_workers=64)
    #训练数据的路径数量 N 远大于 1024
    # 打印加载的样本数量
    print("total path / velocity length:", len(paths), len(velocities))

    # ==================== 步骤2: 路径K-Means聚类 ====================
    #_get_future_path  path 是按**距离间隔**均匀采样的，而非时间间隔
    num_pts = paths[0].shape[0]  # 获取路径点数量（默认50）
    # 将路径展平：形状从 [N, 50, 3] 变为 [N, 150]
    # 其中150 = 50点 × 3坐标(x,y,heading)
    paths_flatten = np.stack(paths).reshape(len(paths), -1)
    # 执行K-Means聚类，n_clusters=K_PATH=1024
    path_cluster = KMeans(n_clusters=K_PATH).fit(paths_flatten).cluster_centers_
    # 恢复形状：形状从 [1024, 150] 变为 [1024, 50, 3]
    path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
    # 将heading标准化到 [-π, π)
    path_cluster[:, :, 2] = (path_cluster[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

    # ==================== 步骤3: 速度K-Means聚类 ====================
    #velocity = torch.norm(pad_trajectory[1:] - pad_trajectory[:-1], dim=-1) / self._config.vel_time_interval
    # 将速度列表堆叠为数组：形状 [N, 8]
    velocities = np.stack(velocities)
    # 执行K-Means聚类，n_clusters=K_VELOCITY=256
    velocity_cluster = KMeans(n_clusters=K_VELOCITY).fit(velocities).cluster_centers_

    # ==================== 步骤4: 组合生成轨迹词汇表 ====================
    # 使用带外推的插值函数，将路径和速度组合成完整轨迹
    trajectory, trajectory_mask = interp_trajectory(
        path_cluster, velocity_cluster, interp_func=interp1d_extrap
    )

    # ==================== 步骤5-6: 可视化和保存 ====================
    visualize(path_cluster, velocity_cluster, trajectory)  # 生成可视化图像
    save_outputs(path_cluster, velocity_cluster, trajectory, trajectory_mask)  # 保存词汇表


if __name__ == "__main__":
    # 当脚本直接运行时调用main函数
    main()
