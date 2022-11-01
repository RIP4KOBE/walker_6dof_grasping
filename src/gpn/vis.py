"""Render volumes, point clouds, and grasp detections in rviz."""

import matplotlib.colors
import numpy as np
from sensor_msgs.msg import PointCloud2
import rospy
from rospy import Publisher
from visualization_msgs.msg import Marker, MarkerArray

from gpn.utils import ros_utils, workspace_lines
from gpn.utils.transform import Transform, Rotation


cmap = matplotlib.colors.LinearSegmentedColormap.from_list("redgreen", ["r", "g"])
DELETE_MARKER_MSG = Marker(action=Marker.DELETEALL)
DELETE_MARKER_ARRAY_MSG = MarkerArray(markers=[DELETE_MARKER_MSG])


def draw_workspace(size):
    scale = size * 0.005
    pose = Transform.identity()
    scale = [scale, 0.0, 0.0]
    color = [0, 0, 0]
    msg = _create_marker_msg(Marker.LINE_LIST, "task", pose, scale, color)
    msg.points = [ros_utils.to_point_msg(point) for point in workspace_lines(size)]
    pubs["workspace"].publish(msg)


def draw_tsdf(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["tsdf"].publish(msg)


def draw_points(points):
    msg = ros_utils.to_cloud_msg(points, frame="task")
    pubs["points"].publish(msg)


def draw_quality(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["quality"].publish(msg)


def draw_volume(vol, voxel_size, threshold=0.01):
    msg = _create_vol_msg(vol, voxel_size, threshold)
    pubs["debug"].publish(msg)


def draw_grasp(grasp, score, finger_depth):
    radius = 0.1 * finger_depth
    # w, d = grasp.width, finger_depth
    w, d = 0.0, 0.08
    color = cmap(float(score))

    markers = []

    msg = _create_grasp_marker_msg(grasp, score, finger_depth)
    msg.id = 0
    markers.append(msg)
    # # left finger1
    # pose = grasp.pose * Transform(Rotation.identity(), [-w / 2, 0.0 , d / 2])
    # scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 0
    # markers.append(msg)

    # # left finger2
    # pose = grasp.pose * Transform(Rotation.identity(), [-w / 2, d / 2,  d / 2])
    # scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 1
    # markers.append(msg)

    # # left finger3
    # pose = grasp.pose * Transform(Rotation.identity(), [-w / 2, -d / 2,  d / 2])
    # scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 2
    # markers.append(msg)

    # #thumb finger
    # pose = grasp.pose * Transform(Rotation.identity(), [w / 2, 0.0, d / 2])
    # scale = [radius, radius, d]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 4
    # markers.append(msg)

    # # wrist
    # pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    # scale = [radius, radius, d / 2]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 5
    # markers.append(msg)

    # # palm
    # pose = grasp.pose * Transform(
    #     Rotation.from_rotvec(np.pi / 2 * np.r_[0.0, 1.0, 0.0]), [0.0, 0.0, 0.0]
    # )
    # scale = [radius, radius, w]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 6
    # markers.append(msg)

    # # left finger4
    # pose = grasp.pose * Transform(
    # Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]), [-2 / w, 0.0, 0.0]
    # )
    # scale = [radius, radius, w]
    # msg = _create_marker_msg(Marker.CYLINDER, "task", pose, scale, color)
    # msg.id = 7
    # markers.append(msg)

    pubs["grasp"].publish(MarkerArray(markers=markers))


def draw_grasps(grasps, scores, finger_depth):
    markers = []
    for i, (grasp, score) in enumerate(zip(grasps, scores)):
        msg = _create_grasp_marker_msg(grasp, score, finger_depth)
        msg.id = i
        markers.append(msg)
    msg = MarkerArray(markers=markers)
    pubs["grasps"].publish(msg)


def clear():
    pubs["workspace"].publish(DELETE_MARKER_MSG)
    pubs["tsdf"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))
    pubs["points"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))
    clear_quality()
    pubs["grasp"].publish(DELETE_MARKER_ARRAY_MSG)
    clear_grasps()
    pubs["debug"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))


def clear_quality():
    pubs["quality"].publish(ros_utils.to_cloud_msg(np.array([]), frame="task"))


def clear_grasps():
    pubs["grasps"].publish(DELETE_MARKER_ARRAY_MSG)


def _create_publishers():
    pubs = dict()
    pubs["workspace"] = Publisher("/workspace", Marker, queue_size=1, latch=True)
    pubs["tsdf"] = Publisher("/tsdf", PointCloud2, queue_size=1, latch=True)
    pubs["points"] = Publisher("/points", PointCloud2, queue_size=1, latch=True)
    pubs["quality"] = Publisher("/quality", PointCloud2, queue_size=1, latch=True)
    pubs["grasp"] = Publisher("/grasp", MarkerArray, queue_size=1, latch=True)
    pubs["grasps"] = Publisher("/grasps", MarkerArray, queue_size=1000, latch=True)
    pubs["debug"] = Publisher("/debug", PointCloud2, queue_size=1, latch=True)
    return pubs


def _create_marker_msg(marker_type, frame, pose, scale, color):
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.type = marker_type
    msg.action = Marker.ADD
    msg.pose = ros_utils.to_pose_msg(pose)
    msg.scale = ros_utils.to_vector3_msg(scale)
    msg.color = ros_utils.to_color_msg(color)
    return msg


def _create_vol_msg(vol, voxel_size, threshold):
    vol = vol.squeeze()
    points = np.argwhere(vol > threshold) * voxel_size
    values = np.expand_dims(vol[vol > threshold], 1)
    return ros_utils.to_cloud_msg(points, values, frame="task")


def _create_grasp_marker_msg(grasp, score, finger_depth):
    radius = 0.1 * 0.05
    # w, d = grasp.width, finger_depth
    w, d = 0.055, 0.055
    scale = [radius, 0.0, 0.0]
    color = cmap(float(score))
    msg = _create_marker_msg(Marker.LINE_LIST, "task", grasp.pose, scale, color)
    msg.points = [ros_utils.to_point_msg(point) for point in _gripper_lines(w, d)]
    return msg


def _gripper_lines(width, depth):
    # vgn grasp gripper_local_frame_visualization
    # return [
    #     #wrist
    #     [0.0, 0.0, -depth],
    #     [0.0, 0.0, 0.0],

    #     #palm
    #     [0.0, -width / 2.0, 0.0],
    #     [0.0, width / 2.0, 0.0],

    #     #thumb
    #     #joint1
    #     [0.0, width / 2.0, 0.0],
    #     [0.0, width / 2.0, depth/4],
    #     #joint2
    #     [0.0, width / 2.0, depth/4],
    #     [0.0, width / 3.0, depth/2],


    #     #left_finger_baselink
    #     [width / 2.0, -width / 2.0, 0.0],
    #     [-width / 2.0, -width / 2.0, 0.0],

    #     #left_finger1
    #     #joint1
    #     [0.0, -width / 2.0, 0.0],
    #     [0.0, -width / 2.0, depth/2],
    #     #joint2
    #     [0.0, -width / 2.0, depth/2],
    #     [0.0, -width / 4.0, depth],

    #     #left_finger2
    #     #joint1
    #     [width / 2.0, -width / 2.0, 0.0],
    #     [width / 2.0, -width / 2.0, depth/2],
    #     #joint2
    #     [width / 2.0, -width / 2.0, depth/2],
    #     [width / 2.0, -width / 4.0, depth],

    #     #left_finger3
    #     #joint1
    #     [-width / 2.0, -width / 2.0, 0.0],
    #     [-width / 2.0, -width / 2.0, depth/2],
    #     #joint2
    #     [-width / 2.0, -width / 2.0, depth/2],
    #     [-width / 2.0, -width / 4.0, depth],

    # ]

    # walker grasp gripper_local_frame visualization
        return [
        #wrist
        [0.0, 0.0, -depth],
        [0.0, 0.0, 0.0],

        #palm
        [-width / 2.0, 0.0 , 0.0],
        [width / 2.0, 0.0 , 0.0],

        #thumb
        #joint1
        [width / 2.0, 0.0, 0.0],
        [width / 2.0, 0.0, depth/4],
        #joint2
        [width / 2.0, 0.0, depth/4],
        [width / 3.0, 0.0, depth/2],

        #left finger base_link
        [-width / 2.0, -width / 2.0, 0.0],
        [-width / 2.0, width / 2.0, 0.0],


        #left finger1
        #joint1
        [-width / 2.0, width / 2.0, 0.0],
        [-width / 2.0, width / 2.0, depth/2],
        #joint2
        [-width / 2.0, width / 2.0, depth/2],
        [-width / 4.0, width / 2.0, depth],     


        #left finger2
        #joint1
        [-width / 2.0, -width / 2.0, 0.0],
        [-width / 2.0, -width / 2.0, depth/2],
        #joint2
        [-width / 2.0, -width / 2.0, depth/2],
        [-width / 4.0, -width / 2.0, depth],

        #left finger3
        #joint1
        [-width / 2.0, 0.0, 0.0],
        [-width / 2.0, 0.0, depth/2],
        #joint2
        [-width / 2.0, 0.0, depth/2],
        [-width / 4.0, 0.0, depth],

    ]


pubs = _create_publishers()
