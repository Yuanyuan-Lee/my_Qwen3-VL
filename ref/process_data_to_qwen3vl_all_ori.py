import os
import json
import numpy as np
import cv2
import h5py
import transforms3d as t3d
import concurrent.futures
import logging  # 新增

# 日志配置
def setup_logger(save_dir):
    log_path = os.path.join(save_dir, "process_data.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler()
        ]
    )

def to_bgr_image(img_data):
    """Convert various image representations to an OpenCV BGR uint8 ndarray."""
    # Case 1: compressed bytes from HDF5 (variable-length)
    if isinstance(img_data, (bytes, np.void)):
        buf = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
    else:
        # Ensure numpy array
        if not isinstance(img_data, np.ndarray):
            img = np.asarray(img_data)
        else:
            img = img_data

    # Handle channel-first (C,H,W) -> (H,W,C)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))

    # Convert dtype to uint8 if needed
    if img.dtype != np.uint8:
        # Heuristic: assume [0,1] floats
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # If RGBA -> BGR
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # If RGB -> BGR
    elif img.ndim == 3 and img.shape[2] == 3:
        # Heuristic: many datasets store RGB; OpenCV expects BGR.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

def xyzrpy_to_matrix(x, y, z, roll, pitch, yaw):
    """将xyzrpy转换为4x4变换矩阵"""
    rotation_matrix = t3d.euler.euler2mat(roll, pitch, yaw, 'rxyz')
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]
    return transform_matrix

def matrix_to_xyzrpy(matrix):
    """将4x4变换矩阵转换为xyzrpy"""
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    
    # 提取欧拉角（固定轴顺序：roll(x), pitch(y), yaw(z)）
    roll, pitch, yaw = t3d.euler.mat2euler(rotation_matrix, 'rxyz')
    
    return np.array([translation[0], translation[1], translation[2], roll, pitch, yaw])

def world_to_camera_transform(world_pose, camera_matrix):
    """
    将世界坐标系下的位姿转换到相机坐标系下
    
    参数:
        world_pose: [x, y, z, roll, pitch, yaw] 或 [x, y, z, roll, pitch, yaw, gripper]
        camera_matrix: 相机在世界坐标系下的4x4变换矩阵
    
    返回:
        camera_pose: 在相机坐标系下的位姿 [x, y, z, roll, pitch, yaw]
    """
    # 提取位姿部分（忽略可能的gripper值）
    if len(world_pose) == 7:
        world_pose_6d = world_pose[:6]
        gripper = world_pose[6]
    else:
        world_pose_6d = world_pose
        gripper = None
    
    # 如果camera_matrix不是4x4矩阵，则补全为4x4齐次变换矩阵
    if camera_matrix.shape == (3, 4):
        camera_matrix = np.concatenate([camera_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    elif camera_matrix.shape == (3, 3):
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = camera_matrix
    
    # 世界坐标系下的变换矩阵
    T_world_end = xyzrpy_to_matrix(*world_pose_6d)
    
    # 计算相机坐标系下的变换矩阵: T_camera_end = T_camera_world * T_world_end
    # 其中 T_camera_world = inv(T_world_camera)
    T_camera_world = np.linalg.inv(camera_matrix)
    T_camera_end = T_camera_world @ T_world_end
    
    # 转换回xyzrpy
    camera_pose_6d = matrix_to_xyzrpy(T_camera_end)
    
    # 如果需要，重新添加gripper值
    if gripper is not None:
        return np.concatenate([camera_pose_6d, [gripper]])
    else:
        return camera_pose_6d

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        logging.error(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper_all, left_arm_all = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper_all, right_arm_all = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        qpos = []
        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)  # joint
            state = state.astype(np.float32)
            qpos.append(state)
        
        qpos = np.array(qpos, dtype=np.float32)
        
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

        left_endpose_world, left_gripper_endpose = (
            root["/endpose/left_endpose"][:,:-1],
            root["/endpose/left_gripper"][()],
        )
        right_endpose_world, right_gripper_endpose = (
            root["/endpose/right_endpose"][:,:-1],
            root["/endpose/right_gripper"][()],
        )
        head_cam_extrinsic = root["/observation/head_camera/extrinsic_cv"][()]

        left_endpose_cam = []
        right_endpose_cam = []
        for i in range(left_endpose_world.shape[0]):
            left_6d_cam = world_to_camera_transform(left_endpose_world[i], head_cam_extrinsic[i])
            right_6d_cam = world_to_camera_transform(right_endpose_world[i], head_cam_extrinsic[i])
            left_6d_cam = np.concatenate([left_6d_cam, [left_gripper_endpose[i]]])
            right_6d_cam = np.concatenate([right_6d_cam, [right_gripper_endpose[i]]])
            left_endpose_cam.append(left_6d_cam)
            right_endpose_cam.append(right_6d_cam)
        left_endpose_cam = np.array(left_endpose_cam)
        right_endpose_cam = np.array(right_endpose_cam)
        # 拼接left_endpose_cam和right_endpose_cam，组成dual_endpose_cam
        dual_endpose_cam = np.concatenate([left_endpose_cam, right_endpose_cam], axis=1)
        
    return qpos, image_dict, dual_endpose_cam

def process_single_episode(episode_idx, hdf5_dir, instructions_dir, save_dir, task_name, task_level, segment_size, total_segments, total_frames):
    try:
        qpos, image_dict, dual_endpose_cam = load_hdf5(hdf5_dir + f"/episode{episode_idx}.hdf5")
        num_steps = qpos.shape[0]
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")

        sample_num = 100
        episode_info_list = []
        for sample_idx in range(sample_num):
            start_idx = np.random.randint(first_idx - 1, num_steps)
            sampled_traj_labels = dual_endpose_cam[start_idx:start_idx + total_frames]
            if len(sampled_traj_labels) < total_frames:
                sampled_traj_labels = np.concatenate(
                    [sampled_traj_labels, np.tile(sampled_traj_labels[-1:], (total_frames - len(sampled_traj_labels), 1))],
                    axis=0
                )
            LABEL_SIZE = segment_size
            NUM_LABEL  = total_segments

            def bucketize(delta, pos_eps=0.02, rot_eps=0.05, grip_eps=0.01):
                out = np.ones_like(delta, dtype=np.int64)
                idx = np.arange(delta.shape[-1]) % 7
                pos_mask  = idx < 3
                rot_mask  = (idx >= 3) & (idx < 6)
                grip_mask = idx == 6
                out[pos_mask  & (delta >  pos_eps)]  = 2
                out[pos_mask  & (delta < -pos_eps)]  = 0
                out[rot_mask & (delta >  rot_eps)]  = 2
                out[rot_mask & (delta < -rot_eps)]  = 0
                out[grip_mask & (delta >  grip_eps)] = 2
                out[grip_mask & (delta < -grip_eps)] = 0
                return out

            traj_label = []
            for i in range(NUM_LABEL):
                s0 = i * LABEL_SIZE
                s1 = s0 + LABEL_SIZE - 1
                delta = sampled_traj_labels[s1] - sampled_traj_labels[s0]
                traj_label.append(bucketize(delta))

            traj_label_txt = []
            txt_formal = ['<X_L_', '<Y_L_', '<Z_L_', '<ROLL_L_', '<PITCH_L_', '<YAW_L_', '<GR_L_',
                          '<X_R_', '<Y_R_', '<Z_R_', '<ROLL_R_', '<PITCH_R_', '<YAW_R_', '<GR_R_']
            for label in traj_label:
                for i, val in enumerate(label):
                    if val == 0:
                        traj_label_txt.append(txt_formal[i] + 'NEG>')
                    elif val == 1:
                        traj_label_txt.append(txt_formal[i] + 'ZERO>')
                    else:
                        traj_label_txt.append(txt_formal[i] + 'POS>')

            instruction_file = os.path.join(instructions_dir, f"episode{episode_idx}.json")
            with open(instruction_file, "r") as f:
                instructions = json.load(f)

            image_paths = []
            for cam_name in ["left_camera", "head_camera", "right_camera"]:
                raw_img = image_dict[cam_name][start_idx]
                img_bgr = to_bgr_image(raw_img)
                image_path_to_save = f"{save_dir}/images/{task_name}/{task_level}/episode_{episode_idx}"
                os.makedirs(image_path_to_save, exist_ok=True)
                image_path = os.path.join(image_path_to_save, f"{start_idx}_{cam_name}.jpg")
                ok = cv2.imwrite(image_path, img_bgr)
                if not ok:
                    raise IOError(f"Failed to write image to {image_path}")
                image_paths.append(image_path)

            special_tokens = [
                "<X_L_NEG>", "<X_L_ZERO>", "<X_L_POS>",
                "<Y_L_NEG>", "<Y_L_ZERO>", "<Y_L_POS>",
                "<Z_L_NEG>", "<Z_L_ZERO>", "<Z_L_POS>",
                "<ROLL_L_NEG>", "<ROLL_L_ZERO>", "<ROLL_L_POS>",
                "<PITCH_L_NEG>", "<PITCH_L_ZERO>", "<PITCH_L_POS>",
                "<YAW_L_NEG>", "<YAW_L_ZERO>", "<YAW_L_POS>",
                "<GR_L_NEG>", "<GR_L_ZERO>", "<GR_L_POS>",
                "<X_R_NEG>", "<X_R_ZERO>", "<X_R_POS>",
                "<Y_R_NEG>", "<Y_R_ZERO>", "<Y_R_POS>",
                "<Z_R_NEG>", "<Z_R_ZERO>", "<Z_R_POS>",
                "<ROLL_R_NEG>", "<ROLL_R_ZERO>", "<ROLL_R_POS>",
                "<PITCH_R_NEG>", "<PITCH_R_ZERO>", "<PITCH_R_POS>",
                "<YAW_R_NEG>", "<YAW_R_ZERO>", "<YAW_R_POS>",
                "<GR_R_NEG>", "<GR_R_ZERO>", "<GR_R_POS>",
            ]

            conversation = {
                "from": "human",
                "value": f"<image>\nThis is the image of the left wrist camera\n<image>\nThis is the image of the head camera\n<image>\nThis is the image of the right camera.\n"
                        f"instruction: {instructions['seen'][sample_idx]}\n"
                        f"Please predict the sequence of 32 trajectory labels that the robotic arm will execute next based on the image and instructions. Each trajectory label contains 14 tokens, and the format of one trajectory label is "
                        f"[x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l, x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]. "
                        f"Please output them in order, using special tokens to represent: { ' '.join(special_tokens) }"
            }
            conversation_gpt = {
                "from": "gpt",
                "value": f"{' '.join(traj_label_txt)}"
            }
            episode_info = {
                "image": image_paths,
                "conversations": [conversation, conversation_gpt]
            }
            episode_info_list.append(episode_info)
        print(f"Processed episode {episode_idx}.")
        logging.info(f"Processed episode {episode_idx}.")
        return episode_info_list
    except Exception as e:
        logging.error(f"Error processing episode {episode_idx}: {e}", exc_info=True)
        return []

def generate_vlm_data_with_sampling(hdf5_dir, instructions_dir, save_dir, task_name, task_level, episode_count=10, segment_size=8, total_segments=32, total_frames=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    episode_info_list = []
    logging.info(f"Processing {episode_count} episodes from {hdf5_dir}...")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_single_episode,
                episode_idx,
                hdf5_dir,
                instructions_dir,
                save_dir,
                task_name,
                task_level,
                segment_size,
                total_segments,
                total_frames
            )
            for episode_idx in range(episode_count)
        ]
        for f in concurrent.futures.as_completed(futures):
            episode_info_list.extend(f.result())

    json_data = episode_info_list
    json_save_path = os.path.join(save_dir, f"{task_name}_{task_level}_{episode_count}.json")
    with open(json_save_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    logging.info(f"Saved VLM data to {json_save_path}")

# 运行脚本
if __name__ == "__main__":
    root_data_dir = '/share/project/liyuanyuan/code/RoboTwin/data'
    save_dir = '/share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_all/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logger(save_dir)  # 初始化日志

    # clean和randomized目录名修正
    level_dir_map = {
        "clean": "aloha-agilex_clean_50",
        "randomized": "aloha-agilex_randomized_500"
    }

    # 遍历所有任务
    for task_name in os.listdir(root_data_dir):
        task_path = os.path.join(root_data_dir, task_name)
        if not os.path.isdir(task_path):
            continue
        # 遍历 clean 和 randomized
        for task_level in ["clean", "randomized"]:
            level_path = os.path.join(task_path, level_dir_map[task_level])
            hdf5_dir = os.path.join(level_path, "data")
            instructions_dir = os.path.join(level_path, "instructions")
            if not (os.path.isdir(hdf5_dir) and os.path.isdir(instructions_dir)):
                msg = f"Skip {task_name} {task_level}: missing data or instructions."
                print(msg)
                logging.warning(msg)
                continue

            # 自动检测 episode 文件数量
            hdf5_files = sorted(
                [f for f in os.listdir(hdf5_dir) if f.startswith("episode") and f.endswith(".hdf5")]
            )
            if len(hdf5_files) == 0:
                msg = f"No episode .hdf5 files found in {hdf5_dir}, skip."
                print(msg)
                logging.warning(msg)
                continue
            episode_count = len(hdf5_files)
            msg = f"Processing {task_name} {task_level}: {episode_count} episodes."
            print(msg)
            logging.info(msg)

            generate_vlm_data_with_sampling(
                hdf5_dir,
                instructions_dir,
                save_dir,
                task_name,
                task_level,
                episode_count=episode_count,
                segment_size=8,
                total_segments=32,
                total_frames=256,
            )
