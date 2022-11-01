#!/usr/bin/env python
"""
Real-time Walker2 6-DOF Grasp Detection (dynamic hand-eye calibration).
"""
import argparse
from pathlib import Path
import time

import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped
from scipy.spatial.transform import Rotation as R
import torch

from gpn import vis
from gpn.grasp import Grasp
from gpn.detection import *
from gpn.perception import *
from gpn.utils import ros_utils
from gpn.utils.transform import Rotation, Transform


class GraspDetectionServer(object):
    def __init__(self, model_path):
    
    # define frames
        self.base_frame_id = "base_link"
        self.task_frame_id = "task"
        self.finger_depth = 0.06
        self.size = 6.0 * self.finger_depth
        self.tf_tree_cam1 = ros_utils.TransformTree()
        self.tf_tree_cam2 = ros_utils.TransformTree()
        

    # define camera parameters
        self.cam1_topic_name = "/rs1/depth/image_rect_raw"
        self.cam1_frame_id = "rs1_depth_optical_frame"
        self.cam1_intrinsic = CameraIntrinsic(848, 480, 428.005, 428.005, 431.945, 232.1087)

        self.cam2_topic_name = "/rs2/depth/image_rect_raw"
        self.cam2_frame_id = "rs2_depth_optical_frame"
        self.cam2_intrinsic = CameraIntrinsic(848, 480, 431.64,  431.64, 423.29, 242.33)

    # define trasnformation between TSDF(task) frame and table_marker frame
        self.T_tablemarker_task = Transform(Rotation.from_quat([-0.707, 0, 0,  0.707]), [0, 0, 0])#the T_offset between the coordinate tablemarker and the coordinate task,obtained by adjustment
        
    # construct the grasp planner
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_network(model_path, self.device)

    # initialize the workspace
        vis.clear()
        vis.draw_workspace(0.3)
     
    # subscribe camera imgs
        self.img = None
        self.cv_bridge = cv_bridge.CvBridge()

        rospy.Subscriber(self.cam1_topic_name, sensor_msgs.msg.Image, self.sensor_cb1)
        rospy.Subscriber(self.cam2_topic_name, sensor_msgs.msg.Image, self.sensor_cb2)

    # subscribe marker pose(real-time)
        rospy.Subscriber("/realsense/aruco_single_rs1/pose", PoseStamped, self.rs1Pose_cb)
        rospy.Subscriber("/realsense/aruco_single_rs2/pose", PoseStamped, self.rs2Pose_cb)
        # rospy.Subscriber("/mynteye/aruco_single_mynteye/pose", PoseStamped, self.torsoPose_cb)
        rospy.Subscriber("/torsocam/aruco_single/pose", PoseStamped, self.torsoPose_cb)

    # setup cb to detect grasps
        rospy.Timer(rospy.Duration(0.1), self.detect_grasps)

    # call back function
    def sensor_cb1(self, msg):
        self.cam1_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def sensor_cb2(self, msg):
        self.cam2_img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        
    def rs1Pose_cb(self, msg):
        self.T_cam1depth_tablemarker = Transform(Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]), [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.T_cam1depth_task = self.T_cam1depth_tablemarker * self.T_tablemarker_task
        # self.tf_tree_cam1.broadcast_static(self.T_cam1depth_task, self.cam1_frame_id, self.task_frame_id)#broard transform for visualization
    
    def rs2Pose_cb(self, msg):
        self.T_cam2depth_tablemarker = Transform(Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]), [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.T_cam2depth_task = self.T_cam2depth_tablemarker * self.T_tablemarker_task
        self.tf_tree_cam2.broadcast_static(self.T_cam2depth_task, self.cam2_frame_id, self.task_frame_id)

    def torsoPose_cb(self, msg):
        self.T_torsocamlink_wallmarker = Transform(Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]), [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # self.tf_tree_cam2.broadcast_static(self.T_torsocamlink_wallmarker, "torsocam_link", "marker_100")
    
    def detect_grasps(self, _):

    # TSDF construction
        self.tsdf = TSDFVolume(0.3, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)#initialize pc

    # TSDF intergration
        # cam1 intergration
        self.tsdf.integrate(self.cam1_img, self.cam1_intrinsic, self.T_cam1depth_task)
        self.high_res_tsdf.integrate(self.cam1_img, self.cam1_intrinsic, self.T_cam1depth_task)
        print("Cam1 tsdf Constructed")

        # cam2 intergration
        self.tsdf.integrate(self.cam2_img, self.cam2_intrinsic, self.T_cam2depth_task)
        self.high_res_tsdf.integrate(self.cam2_img, self.cam2_intrinsic, self.T_cam2depth_task)
        print("Cam2 tsdf Constructed")

        # extract grid   
        tsdf_vol = self.tsdf.get_grid()
        voxel_size = self.tsdf.voxel_size

    # TSDF visualization
        vis.draw_tsdf(tsdf_vol.squeeze(), voxel_size)

    # point cloud visualization
        vis.draw_points(np.asarray(self.high_res_tsdf.get_cloud().points))

    # plann grasps
        qual_vol, rot_vol, width_vol = predict(tsdf_vol, self.net, self.device)
        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
        vis.draw_quality(qual_vol, voxel_size, threshold=0.01)

    # select the grasp candidates with a rating of 0.80 or higher and visualize

        # select
        grasps, scores = select(qual_vol, rot_vol, width_vol, 0.82, 1)
        num_grasps = len(grasps)
        if num_grasps > 0:
            idx = np.random.choice(num_grasps, size=min(5, num_grasps), replace=False)
            grasps, scores = np.array(grasps)[idx], np.array(scores)[idx]
        grasps = [from_voxel_coordinates(g, voxel_size) for g in grasps]

        # visualize
        vis.clear_grasps()
        rospy.sleep(0.01)
        vis.draw_grasps(grasps, scores, 0.05)
        tic = time.time()
        print("Visualize", time.time() - tic)
        self.img = None

    # hand-eye calibration and pub msg

        # define grasp pose publisher
        grasp_pub = rospy.Publisher('walker_grasp_pose',Pose, queue_size=1)
        pregrasp_pub = rospy.Publisher('walker_pregrasp_pose',Pose, queue_size=1)
        
        for i, grasp in enumerate(grasps):

        # grasp pose hand-eye calibration and pub msgs

            # hand-eye calibration
            grasp = grasps[i]
            T_task_grasp = grasp.pose 
            T_grasp_offset = Transform(Rotation.identity(), [0.0, 0.0, 0.0])
            T_task_grasp_offset = T_task_grasp * T_grasp_offset
            grasp.pose = T_task_grasp_offset
            T_grasp_handeye_calibration = self.handeye_calibra(grasp)
            pose_torsobaselink_grasp = ros_utils.to_pose_msg(T_grasp_handeye_calibration)

            # pub msgs
            grasp_pub.publish(pose_torsobaselink_grasp)

        # pregrasp pose hand-eye calibration and pub msgs

            #hand-eye calibration 
            T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
            T_task_grasp = grasp.pose 
            T_task_pregrasp = T_task_grasp * T_grasp_pregrasp
            pregrasp = Grasp(T_task_pregrasp,0.05)
            T_pregrasp_handeye_calibration = self.handeye_calibra(pregrasp)
            pose_baselink_pregrasp = ros_utils.to_pose_msg(T_pregrasp_handeye_calibration)
    
            # pub msg
            pregrasp_pub.publish(pose_baselink_pregrasp)
            

    def handeye_calibra(self, grasp):

        T_task_grasp = grasp.pose

        # hand-eye calibration transform
        self.T_wallmarker_tablemarker = Transform(Rotation.from_quat([0.0568647, -0.104276, -0.613399, 0.780791]), [-0.373375, 0.372689, 0.0539117])

        # self.T_torsobaselink_torsocamlink = Transform(Rotation.from_quat([-0.4892347, 0.4489015, -0.54922587, 0.507432432]), [0.162719, 0.0406282, -0.062010])

        self.T_torsobaselink_torsocamlink = Transform(Rotation.from_quat([-0.019671, -0.00539124, 0.007838, 0.99976]), [0.11915, 0.03208, -0.05232])

        T_torsobaselink_task = self.T_torsobaselink_torsocamlink * self.T_torsocamlink_wallmarker * self.T_wallmarker_tablemarker * self.T_tablemarker_task
        self.tf_tree_cam2.broadcast_static(T_torsobaselink_task, "torso_base_link", "task")
        T_grasp_hankgrasp = Transform(Rotation.from_quat([0.471, 0.528, 0.528, -0.471]), [-0.000, 0.020, 0.006])

        # final hand-eye calibration 
        T_torsobaselink_hankgrasp = T_torsobaselink_task * T_task_grasp * T_grasp_hankgrasp

        return T_torsobaselink_hankgrasp

    def select_grasp(self, grasps, scores):

        # select the highest grasp
        heights = np.empty(len(grasps))
        for i, grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    rospy.init_node("walker_detection")
    GraspDetectionServer(args.model)
    rospy.spin()
