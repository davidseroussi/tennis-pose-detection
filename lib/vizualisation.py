import numpy as np
from ipywidgets import interact, Layout
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from lib.pose import Pose, angle_evolution, limbs, body_parts
import cv2
import os.path
import matplotlib.gridspec as gridspec

def segment(joints, part1, part2):
    idx1 = body_parts[part1]
    idx2 = body_parts[part2]
    
    start = joints[idx1]
    end = joints[idx2]
    
    return np.array([start[0], end[0]]), np.array([start[1], end[1]]), np.array([start[2], end[2]])

def update_frame_plot(implot, frame):
    implot.set_data(frame)

def update_pose_plot(lines_3d, pose):
    for i, (part1, part2) in enumerate(list(limbs.values())):
        _x, _y, _z = segment(pose.joints, part1, part2)
        lines_3d[i].set_ydata(_y)
        lines_3d[i].set_xdata(_x)
        lines_3d[i].set_3d_properties(_z, zdir='z')
        
def update_angle_plot(self, x):
    self.point.set_xdata([x])
    self.point.set_ydata(self.angles[x])
    self.vertical_line.set_xdata([x])
    self.txt.set_position((x + 5, self.angles[x]))
    self.txt.set_text(int(self.angles[x]))

class AnglePlot:
    def __init__(self, ax, angles, title=None):
        self.angles = angles
        self.points = []
        self.txts = []

        for i, angle_data in enumerate(self.angles):
            ax.plot(np.arange(len(angle_data)), angle_data, label=str(i))
            point, = ax.plot([0],[angle_data[0]], marker="o", zorder=3)
            self.points.append(point)
            self.txts.append(ax.text(5, angle_data[0], int(angle_data[0])))
        ax.legend()

        self.vertical_line = ax.axvline(color='k', alpha=0.2)

        if title is not None:
            ax.set_title(title)

    def update_cursor(self, idx):
        self.vertical_line.set_xdata([idx])
        for angle_data, point, txt in zip(self.angles, self.points, self.txts):
            try:
                point.set_xdata([idx])
                point.set_ydata(angle_data[idx])
                txt.set_position((idx + 5, angle_data[idx]))
                txt.set_text(int(angle_data[idx]))
            except IndexError as e:
                print('Index out of bounds')
                continue

class PosePlot:
    def __init__(self, ax, poses):
        self.ax = ax
        self.poses = poses
        self._init_pose_plot()

    def _init_pose_plot(self):
        radius = 1.7

        self.ax.set_xlim3d([-radius/3, radius/3])
        self.ax.set_zlim3d([-radius/3, radius/3])
        self.ax.set_ylim3d([-radius/3, radius/3])
        self.ax.dist = 7.5

        self.ax.set_zlabel('Z')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.view_init(elev=15., azim=-70)

        lines_3d = []

        first_pose = self.poses[0]

        for (part1, part2) in list(limbs.values()):
            c = 'red' if 'right' in part1 or 'right' in part2 else 'black'
            _x, _y, _z = segment(first_pose.joints, part1, part2)
            line_3d = self.ax.plot(_x, _y, _z, zdir='z', c=c)[0]
            lines_3d.append(line_3d)
            
        self.lines_3d = lines_3d
    
    def update_pose(self, idx):
        for i, (part1, part2) in enumerate(list(limbs.values())):
            _x, _y, _z = segment(self.poses[idx].joints, part1, part2)
            self.lines_3d[i].set_ydata(_y)
            self.lines_3d[i].set_xdata(_x)
            self.lines_3d[i].set_3d_properties(_z, zdir='z')

class VideoPlot:
    def __init__(self, frames):
        self.frames = frames
        self.frames_to_plot = self.frames.copy()

    def set_frames_range(self, frame_range):
        self.frames_to_plot = self.frames[frame_range[0]:frame_range[1]] if frame_range else self.frames.copy()

    def set_ax(self, ax):
        self.implot = ax.imshow(cv2.cvtColor(self.frames_to_plot[0], cv2.COLOR_BGR2RGB))

    def update_frame(self, idx):
        self.implot.set_data(cv2.cvtColor(self.frames_to_plot[idx], cv2.COLOR_BGR2RGB))

class PoseViewer:
    def __init__(self, poses, frames):
        self.poses = poses
        self.video_plot = VideoPlot(frames)

    def update_video_plot(self, idx):
        self.video_plot.update_frame(idx)
    
    def update_pose_plot(self, idx):
        self.pose_plot.update_pose(idx)
            
    def update_angle_plots(self, x):
        for angle_plot in self.angle_plots:
            angle_plot.update_cursor(x)
    
    def update(self, x):
        self.update_video_plot(x)
        self.update_pose_plot(x)
        self.update_angle_plots(x)
        self.fig.canvas.draw()

    def plot_pose_evolution(self, limbs_angles, frames_range=None):
        self.fig = plt.figure(figsize=(9, 3 + 3 * len(limbs_angles)))
        gs = gridspec.GridSpec(1 + len(limbs_angles), 2)

        self.poses_to_plot = self.poses[frames_range[0]:frames_range[1]] if frames_range else self.poses

        # init frame plot
        ax_frame = plt.subplot(gs[0, 0])
        self.video_plot.set_ax(ax_frame)
        self.video_plot.set_frames_range(frames_range)
        
        # init pose plot
        ax_pose = plt.subplot(gs[0, 1], projection='3d')
        self.pose_plot = PosePlot(ax_pose, self.poses_to_plot)

        # init angles plots
        self.angle_plots = []
        for i in range(len(limbs_angles)):
            ax_angle = plt.subplot(gs[i + 1, :])
            angle_between_limbs = limbs_angles[i]
            angles, _, _ = angle_evolution(self.poses_to_plot, angle_between_limbs[0], angle_between_limbs[1])
            angle_plot = AnglePlot(ax_angle, [angles], f'Angle between {angle_between_limbs[0]} and {angle_between_limbs[1]}')
            self.angle_plots.append(angle_plot)

        self.fig.tight_layout()
        self.fig.show()

        interact(self.update, x=widgets.IntSlider(min=0, max=len(self.poses_to_plot)-1, step=1, value=0))


class DualPoseViewer:
    def __init__(self, poses1, poses2, frames1, frames2):
        self.poses1 = poses1
        self.poses2 = poses2
        self.video_plot1 = VideoPlot(frames1)
        self.video_plot2 = VideoPlot(frames2)

    def update_video_plot(self, idx):
        self.video_plot1.update_frame(idx)
        self.video_plot2.update_frame(idx)
    
    def update_pose_plot(self, idx):
        self.pose_plot1.update_pose(idx)
        self.pose_plot2.update_pose(idx)

    def update_angle_plots(self, x):
        for angle_plot in self.angle_plots:
            angle_plot.update_cursor(x)
    
    def update(self, x):
        self.update_video_plot(x)
        self.update_pose_plot(x)
        self.update_angle_plots(x)
        self.fig.canvas.draw()

    def plot(self, limbs_angles, frames_range1=None, frames_range2=None):
        self.fig = plt.figure(figsize=(9, 6 + 3 * len(limbs_angles)))
        gs = gridspec.GridSpec(2 + len(limbs_angles), 2)

        self.poses_to_plot1 = self.poses1[frames_range1[0]:frames_range1[1]] if frames_range1 else self.poses1
        self.poses_to_plot2 = self.poses2[frames_range2[0]:frames_range2[1]] if frames_range2 else self.poses2

        # init frame plots
        ax_frame1 = plt.subplot(gs[0, 0])
        ax_frame2 = plt.subplot(gs[0, 1])

        self.video_plot1.set_ax(ax_frame1)
        self.video_plot1.set_frames_range(frames_range1)
        self.video_plot2.set_ax(ax_frame2)
        self.video_plot2.set_frames_range(frames_range2)
        
        # init pose plots
        ax_pose1 = plt.subplot(gs[1, 0], projection='3d')
        ax_pose2 = plt.subplot(gs[1, 1], projection='3d')

        self.pose_plot1 = PosePlot(ax_pose1, self.poses_to_plot1)
        self.pose_plot2 = PosePlot(ax_pose2, self.poses_to_plot2)

        # init angles plots
        self.angle_plots = []
        for i in range(len(limbs_angles)):
            ax_angle = plt.subplot(gs[i + 2, :])
            angle_between_limbs = limbs_angles[i]
            angles1, _, _ = angle_evolution(self.poses_to_plot1, angle_between_limbs[0], angle_between_limbs[1], axes=[1, 2])
            angles2, _, _ = angle_evolution(self.poses_to_plot2, angle_between_limbs[0], angle_between_limbs[1], axes=[1, 2])
            angle_plot = AnglePlot(ax_angle, [angles1, angles2], f'Angle between {angle_between_limbs[0]} and {angle_between_limbs[1]}')
            self.angle_plots.append(angle_plot)

        self.fig.tight_layout()
        self.fig.show()

        _max = max(len(self.poses_to_plot1)-1, len(self.poses_to_plot2)-1)
        interact(self.update, x=widgets.IntSlider(min=0, max=_max, step=1, value=0, layout=Layout(width='500px')))

