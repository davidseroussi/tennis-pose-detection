import numpy as np
from lib.preprocessing import get_ball_positions, get_min_distance_idx
from lib.pose import detectron_mapping


def pad_poses(poses, padding):
    return np.pad(poses, (padding, 0), mode='edge')


def pad_frames(frames, padding):
    return np.pad(frames, ((padding, 0), (0, 0), (0, 0), (0, 0)), mode='edge')


def get_keypoints(path_dataset):
    keypoints = np.load(path_dataset, allow_pickle=True)
    subject = list(keypoints['positions_2d'].item().keys())[0]
    action = list(keypoints['positions_2d'].item()[subject].keys())[0]
    keypoints = keypoints['positions_2d'].item()[subject][action][0]
    return keypoints


def get_padding_for_alignment(frames1, frames2, keypoints1, keypoints2, resize_factor1, resize_factor2):
    ball_positions1 = get_ball_positions(frames1)
    ball_positions2 = get_ball_positions(frames2)

    min_idx1 = get_min_distance_idx(
        ball_positions1,
        keypoints1[:, detectron_mapping['right_wrist']] * resize_factor1
    )
    min_idx2 = get_min_distance_idx(
        ball_positions2,
        keypoints2[:, detectron_mapping['right_wrist']] * resize_factor2
    )

    padding = abs(min_idx1 - min_idx2)

    return padding, min_idx1, min_idx2


if __name__ == "__main__":
    get_keypoints('datasets/data_cut_7.npz')
    get_keypoints('datasets/data_cut_5.npz')
