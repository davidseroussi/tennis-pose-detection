from lib.vector_operations import *
import numpy as np

detectron_mapping = {
    "nose":0,
    "left_eye":1,
    "right_eye":2,
    "left_ear":3,
    "right_ear":4,
    "left_shoulder":5,
    "right_shoulder":6,
    "left_elbow":7,
    "right_elbow":8,
    "left_wrist":9,
    "right_wrist":10,
    "left_hip":11,
    "right_hip":12,
    "left_knee":13,
    "right_knee":14,
    "left_ankle":15,
    "right_ankle":16
}

body_parts = {
    'nose': 9,
    'top_head': 10,
    'left_shoulder': 11,
    'right_shoulder': 14,
    'left_elbow': 12,
    'right_elbow': 15,
    'left_wrist': 13,
    'right_wrist': 16,
    'left_hip': 4,
    'right_hip': 1,
    'middle_hip':0,
    'middle_body': 7,
    'left_knee': 5,
    'right_knee': 2,
    'neck': 8,
    'left_ankle':6,
    'right_ankle':3
}

limbs = {
    'left_forearm': ('left_elbow', 'left_wrist'),
    'left_arm': ('left_shoulder', 'left_elbow'),
    'left_femur': ('left_hip', 'left_knee'),
    'left_tibia': ('left_knee', 'left_ankle'),
    'right_forearm': ('right_elbow', 'right_wrist'),
    'right_arm': ('right_shoulder', 'right_elbow'),
    'right_femur': ('right_hip', 'right_knee'),
    'right_tibia': ('right_knee', 'right_ankle'),
    'shoulders': ('left_shoulder', 'right_shoulder'),
    'hips': ('left_hip', 'right_hip'),
    'right_hip': ('right_hip', 'middle_hip'),
    'left_hip': ('left_hip', 'middle_hip'),
    'head': ('nose', 'top_head'),
    'neck': ('nose', 'neck'),
    'left_shoulder': ('left_shoulder', 'neck'),
    'right_shoulder': ('right_shoulder', 'neck'),
    'lower_back': ('middle_hip', 'middle_body'),
    'upper_back': ('neck', 'middle_body')
}

def angle_evolution(poses, limb1, limb2, axes=[0, 1, 2]):
    angles = []
    vec1s = []
    vec2s = []

    for pose in poses:
        angle, vec1, vec2 = pose.angle_between_limbs(limb1, limb2, axes)
        angles.append(angle)
        vec1s.append(vec1)
        vec2s.append(vec2)

    return angles, vec1s, vec2s

class Pose:
    def __init__(self, joints):
        self.joints = joints

    def point_of_part(self, part): 
        assert part in list(body_parts.keys())   
        return self.joints[body_parts[part]]

    def angle(self, joint1, joint_middle, joint2):
        point1 = self.point_of_part(joint1)
        point_middle = self.point_of_part(joint_middle)
        point2 = self.point_of_part(joint2)

        vec_part1 = point1 - point_middle
        vec_part2 = point2 - point_middle
        
        return angle_between(vec_part1, vec_part2)

    def points_of_limb(self, limb):
        return np.array([self.point_of_part(limb[0]), self.point_of_part(limb[1])])

    def angle_between_limbs(self, limb1, limb2, axes=[0, 1, 2]):
        """
        Only handles intersecting parts for now (forearm-arm, shoulders-arm etc)
        """

        assert limb1 in list(limbs.keys()), f'{limb1} is not registered'
        assert limb2 in list(limbs.keys()), f'{limb2} is not registered'

        limb1 = limbs[limb1]
        limb2 = limbs[limb2]

        points1 = self.points_of_limb(limb1)
        points2 = self.points_of_limb(limb2)

        vec_limb1 = points1[1] - points1[0]
        vec_limb2 = points2[1] - points2[0]

        intersection = get_intersection_3d(points1[0], vec_limb1, points2[0], vec_limb2)

        dist1 = [np.linalg.norm(point - intersection) for point in points1]
        dist2 = [np.linalg.norm(point - intersection) for point in points2]

        vec1 = points1[np.argmax(dist1)] - points1[np.argmin(dist1)]
        vec2 = points2[np.argmax(dist2)] - points2[np.argmin(dist2)]

        return angle_between(vec1[axes], vec2[axes]), vec1, vec2

if __name__ == "__main__":
    joints = np.load('joints.npy')
    pose = Pose(joints[0])
    angle = pose.angle_between_limbs('left_arm', 'left_forearm')
    print(angle)
    angle = pose.angle_between_limbs('left_forearm', 'left_arm')
    print(angle)
    