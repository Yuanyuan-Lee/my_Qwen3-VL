import os
import json
import numpy as np
import cv2
import h5py
import transforms3d as t3d
import concurrent.futures
import logging

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
    if isinstance(img_data, (bytes, np.void)):
        buf = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
    else:
        if not isinstance(img_data, np.ndarray):
            img = np.asarray(img_data)
        else:
            img = img_data

    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
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
    roll, pitch, yaw = t3d.euler.mat2euler(rotation_matrix, 'rxyz')
    return np.array([translation[0], translation[1], translation[2], roll, pitch, yaw])

def world_to_camera_transform(world_pose, camera_matrix):
    """将世界坐标系下的位姿转换到相机坐标系下"""
    if len(world_pose) == 7:
        world_pose_6d = world_pose[:6]
        gripper = world_pose[6]
    else:
        world_pose_6d = world_pose
        gripper = None
    
    if camera_matrix.shape == (3, 4):
        camera_matrix = np.concatenate([camera_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    elif camera_matrix.shape == (3, 3):
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = camera_matrix
    
    T_world_end = xyzrpy_to_matrix(*world_pose_6d)
    T_camera_world = np.linalg.inv(camera_matrix)
    T_camera_end = T_camera_world @ T_world_end
    camera_pose_6d = matrix_to_xyzrpy(T_camera_end)
    
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
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
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
        dual_endpose_cam = np.concatenate([left_endpose_cam, right_endpose_cam], axis=1)
        
    return qpos, image_dict, dual_endpose_cam

def normalize_position(pos, axis_idx):
    """
    归一化位置到[0, 1]范围
    pos: 位置值
    axis_idx: 0=x1, 1=y1, 2=z1, 3=x2, 4=y2, 5=z2
    """
    if axis_idx == 0:  # x1: [-0.5, 0]
        return (pos + 0.5) / 0.5
    elif axis_idx == 3:  # x2: [0, 0.5]
        return pos / 0.5
    elif axis_idx in [1, 4]:  # y1, y2: [0, 0.5]
        return pos / 0.5
    elif axis_idx in [2, 5]:  # z1, z2: [0, 1]
        return pos / 1.0
    else:
        raise ValueError(f"Invalid axis_idx: {axis_idx}")

def denormalize_position(norm_pos, axis_idx):
    """
    反归一化位置
    """
    if axis_idx == 0:  # x1: [-0.5, 0]
        return norm_pos * 0.5 - 0.5
    elif axis_idx == 3:  # x2: [0, 0.5]
        return norm_pos * 0.5
    elif axis_idx in [1, 4]:  # y1, y2: [0, 0.5]
        return norm_pos * 0.5
    elif axis_idx in [2, 5]:  # z1, z2: [0, 1]
        return norm_pos * 1.0
    else:
        raise ValueError(f"Invalid axis_idx: {axis_idx}")

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

        if first_idx < 3:
            first_idx = 3

        sample_num = len(dual_endpose_cam)-first_idx
        episode_info_list = []
        
        for sample_idx in range(sample_num):
            start_idx = first_idx-1 + sample_idx
            sampled_traj_labels = dual_endpose_cam[start_idx:start_idx + total_frames]
            
            if len(sampled_traj_labels) < total_frames:
                sampled_traj_labels = np.concatenate(
                    [sampled_traj_labels, np.tile(sampled_traj_labels[-1:], (total_frames - len(sampled_traj_labels), 1))],
                    axis=0
                )
            
            # 提取xyz位置 (维度: 0,1,2,7,8,9)
            LABEL_SIZE = segment_size
            NUM_LABEL  = total_segments
            
            # 生成位置标签: 每LABEL_SIZE帧预测一次
            position_labels = []
            for i in range(NUM_LABEL):
                frame_idx = (i + 1) * LABEL_SIZE  # 预测第8, 16, 24...64帧
                if frame_idx < len(sampled_traj_labels):
                    endpose = sampled_traj_labels[frame_idx]
                    # 提取6个xyz位置并归一化
                    positions = np.array([
                        endpose[0],  # x1 (left)
                        endpose[1],  # y1
                        endpose[2],  # z1
                        endpose[7],  # x2 (right)
                        endpose[8],  # y2
                        endpose[9],  # z2
                    ])
                    # 归一化到[0, 1]
                    normalized_positions = np.array([
                        normalize_position(positions[i], i) for i in range(6)
                    ])
                    position_labels.append(normalized_positions)
                else:
                    # 如果超出范围,使用最后一帧
                    position_labels.append(position_labels[-1] if position_labels else np.zeros(6))
            
            # 将位置标签转换为字符串格式
            # 格式: "<pos>x1,y1,z1,x2,y2,z2</pos>_<pos>x1,y1,z1,x2,y2,z2</pos>_..."
            position_label_txt = "_".join([
                f"<pos>{','.join([f'{p:.6f}' for p in label])}</pos>"
                for label in position_labels
            ])

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

            endpose_txt = np.array2string(
                dual_endpose_cam[start_idx-2: start_idx+1], 
                separator=', ', 
                formatter={'float_kind': lambda x: f"{x:.6f}"}
            ).replace('\n', '')

            def build_human_conv(instr_text):
                return {
                    "from": "human",
                    "value": f"You are a Aloha-AgileX robot using end-effector control. The instruction is \"{instr_text}\".<image>\nThis is the left camera image of the current frame.\n<image>\nThis is the head camera image of the current frame.\n<image>\nThis is the right camera image of the current frame.\n"
                             f"The format of endpose is [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l, x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r] and the previous three (including current) frames' endposes are: {endpose_txt}.\n"
                             f"Please predict the next {total_segments} 3D gripper positions at frames {','.join([str((i+1)*segment_size) for i in range(total_segments)])}. "
                             f"Each position should contain [x_left, y_left, z_left, x_right, y_right, z_right]. "
                             f"The ranges are: x_left∈[-0.5,0], x_right∈[0,0.5], y_left,y_right∈[0,0.5], z_left,z_right∈[0,1]. "
                             f"Output format: <pos>x1,y1,z1,x2,y2,z2</pos>_<pos>x1,y1,z1,x2,y2,z2</pos>_..."
                }

            conversation_gpt = {
                "from": "gpt",
                "value": position_label_txt
            }

            seen_list = instructions.get('seen', [])
            if len(seen_list) == 0:
                logging.warning(f"No 'seen' instructions in {instruction_file}, skipping sample.")
                continue
            seen_idx = np.random.randint(0, len(seen_list))
            seen_instr = seen_list[seen_idx]

            episode_info_seen = {
                "image": image_paths,
                "conversations": [build_human_conv(seen_instr), conversation_gpt]
            }
            episode_info_list.append(episode_info_seen)

            unseen_list = instructions.get('unseen', [])
            if len(unseen_list) > 0:
                unseen_idx = np.random.randint(0, len(unseen_list))
                unseen_instr = unseen_list[unseen_idx]
                episode_info_unseen = {
                    "image": image_paths,
                    "conversations": [build_human_conv(unseen_instr), conversation_gpt]
                }
                episode_info_list.append(episode_info_unseen)
            else:
                logging.debug(f"No 'unseen' instructions in {instruction_file}; only 'seen' sample created.")
        
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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate VLM data from RoboTwin hdf5 datasets.")
    parser.add_argument("--root-data-dir", type=str, default='/share/project/liyuanyuan/code/RoboTwin/data',
                        help="Root directory containing task folders.")
    parser.add_argument("--save-dir", type=str, default='/share/project/liyuanyuan/code/Qwen3-VL/qwen3_vl_data_all/',
                        help="Directory to save processed data and images.")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated list of task folder names to process. If empty, process all tasks.")
    parser.add_argument("--task-file", type=str, default="",
                        help="Path to a text file containing one task name per line (overrides --tasks if provided).")
    parser.add_argument("--levels", type=str, default="clean,randomized",
                        help="Comma-separated levels to process among {clean,randomized}. Default: clean,randomized")
    parser.add_argument("--segment-size", type=int, default=8)
    parser.add_argument("--total-segments", type=int, default=8)
    parser.add_argument("--total-frames", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=8, help="Max threads for processing (ThreadPoolExecutor).")
    args = parser.parse_args()

    root_data_dir = args.root_data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setup_logger(save_dir)

    level_dir_map = {
        "clean": "aloha-agilex_clean_50",
        "randomized": "aloha-agilex_randomized_500"
    }

    if args.task_file:
        if not os.path.isfile(args.task_file):
            logging.error(f"Task file not found: {args.task_file}")
            sys.exit(1)
        with open(args.task_file, "r") as tf:
            requested_tasks = [line.strip() for line in tf if line.strip()]
    elif args.tasks:
        requested_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        requested_tasks = []

    requested_levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    valid_levels = {"clean", "randomized"}
    for lv in requested_levels:
        if lv not in valid_levels:
            logging.error(f"Invalid level '{lv}'. Valid options: clean, randomized")
            sys.exit(1)

    if requested_tasks:
        task_list = []
        missing_tasks = []
        for t in requested_tasks:
            tpath = os.path.join(root_data_dir, t)
            if os.path.isdir(tpath):
                task_list.append(t)
            else:
                missing_tasks.append(t)
        if missing_tasks:
            logging.warning(f"Requested tasks not found under {root_data_dir}: {missing_tasks}")
        if not task_list:
            logging.error("No valid tasks to process. Exiting.")
            sys.exit(1)
    else:
        task_list = [d for d in os.listdir(root_data_dir) if os.path.isdir(os.path.join(root_data_dir, d))]
        if not task_list:
            logging.error(f"No tasks found under {root_data_dir}. Exiting.")
            sys.exit(1)

    logging.info(f"Tasks to process: {task_list}")
    logging.info(f"Levels to process: {requested_levels}")

    for task_name in task_list:
        task_path = os.path.join(root_data_dir, task_name)
        if not os.path.isdir(task_path):
            logging.warning(f"Skip {task_name}: not a directory.")
            continue

        for task_level in requested_levels:
            level_key = task_level
            if level_key not in level_dir_map:
                logging.warning(f"Unknown level '{task_level}' for task {task_name}, skipping.")
                continue
            level_path = os.path.join(task_path, level_dir_map[level_key])
            hdf5_dir = os.path.join(level_path, "data")
            instructions_dir = os.path.join(level_path, "instructions")
            if not (os.path.isdir(hdf5_dir) and os.path.isdir(instructions_dir)):
                msg = f"Skip {task_name} {task_level}: missing data or instructions."
                print(msg)
                logging.warning(msg)
                continue

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
                segment_size=args.segment_size,
                total_segments=args.total_segments,
                total_frames=args.total_frames,
            )
