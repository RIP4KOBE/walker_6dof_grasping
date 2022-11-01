import argparse
from pathlib import Path
import pybullet_data

from mpi4py import MPI
import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
from gpn.utils import btsim
from gpn.utils.o3d_vis import create_arrow
from gpn.grasp import Grasp, Label
from gpn.io import *
from gpn.perception import *
from gpn.simulation import ClutterRemovalSim
from gpn.utils.transform import Rotation, Transform
import math
import time

OBJECT_COUNT_LAMBDA = 1
MAX_VIEWPOINT_COUNT = 10
GRASPS_PER_SCENE = 120

PI = math.pi

#Pybullet getJointState() function test
def test1():
    #scene initialization
    scene = "packed"
    object_set = "blocks"
    gui = True
    sim = ClutterRemovalSim(scene, object_set, gui)

    p = sim.world.p
    #load 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane100.urdf")

    urdf_path = Path("data/urdfs/walker/Hand_with_friction.urdf")
    R = Rotation.from_euler("xyz", [PI/6,PI/3,-PI/2])
    T_world_body = Transform(Rotation.identity() * R, [0.0, 0.0, 0.0]) 
    body = sim.world.load_urdf(urdf_path, T_world_body)

    angle = 0
    steps_to_take = 100
    numJoints = p.getNumJoints(body.uid)

    #close fingeres
    for i in range(1, steps_to_take):
        for j in range(0, numJoints):
            if j % 2 == 1:
                angle = 1.6 * i / steps_to_take
                print("angle", '\r'+ str(angle))
                p.setJointMotorControl2(bodyUniqueId=body.uid,
                                            jointIndex=j,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=angle,
                                            force=1)
            else:
                angle = 1.6 * i / steps_to_take
                print("angle", '\r'+str(angle))
                p.setJointMotorControl2(bodyUniqueId=body.uid,
                                            jointIndex=j,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=angle,
                                            force=1)
        sim.world.step()
        time.sleep(0.1)

    #read joint value
    joint_state1 = sim.world.p.getJointState(body.uid, 1)[0]
    print("joint_state1",joint_state1)
    joint_state2 = sim.world.p.getJointState(body.uid, 3)[0]
    print("joint_state2",joint_state2)

    contacts = sim.world.get_contacts(body)
    if not len(contacts) > 0:
        print("No collision detected")

    if gui:
        time.sleep(50)

#self.gripper.move_tcp_xyz() function test
def test2():
    scene = "packed"
    object_set = "blocks"
    seed = 3
    gui = True
    sim = ClutterRemovalSim(scene, object_set, gui)
    sim.gripper.reset(Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

    T_world_grasp = Transform(Rotation.identity(), [0.03, 0.03, 0.03])
    sim.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
    time.sleep(1)

    T_world_grasp = Transform(Rotation.identity(), [0.05, 0.05, 0.05])
    sim.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
    time.sleep(1)

    T_world_grasp = Transform(Rotation.identity(), [0.07, 0.07, 0.07])
    sim.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
    time.sleep(1)

    T_world_grasp = Transform(Rotation.identity(), [0.02, 0.02, 0.02])
    sim.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
    time.sleep(1)

    T_world_grasp = Transform(Rotation.identity(), [0.04, 0.04, 0.04])
    sim.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)

    if gui:
        time.sleep(50)

#filter packed scene model library
def test3():
    #set simulation parameters
    gui = True 
    world = btsim.BtWorld(gui)
    global_scaling = 1.15
    rng = np.random
    table_height =0.06

    #set model library path of packed scene
    urdf_root = Path("data/urdfs/")
    # object_set = "packed/object_test"
    object_set = "walker2_objet_model(pile+packed)"

    #add table 
    urdf_table = urdf_root / "setup" / "plane.urdf"
    pose_table = Transform(Rotation.identity(), [0.15, 0.15, table_height])
    world.load_urdf(urdf_table, pose_table, scale=0.6)

    #add walker2_hand
    urdf_walker2_hand = urdf_root / "walker" / "Hand_with_friction.urdf"
    R = Rotation.from_euler("xyz", [0,PI,PI])
    pose_hand = Transform(R, [0.1, 0.3, 0.12])
    world.load_urdf(urdf_walker2_hand, pose_hand)

    #display models in order
    root = urdf_root / object_set
    object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
    counter = 0

    for urdf in object_urdfs:
        print(urdf)
        x = rng.uniform(0.08, 0.22)
        y = rng.uniform(0.08, 0.22)
        z = 1.0
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
        pose = Transform(rotation, np.r_[x, y, z])
        scale = rng.uniform(0.8, 1.0)
        body = world.load_urdf(urdf, pose, scale=global_scaling * scale)
        lower, upper = world.p.getAABB(body.uid)
        z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
        body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
        world.step()
        time.sleep(5)
        world.remove_body(body)
        counter += 1
        print(counter)

def main():
    scene = "packed"
    #scene = "pile"
    object_set = "blocks"
    gui = True
    sim = ClutterRemovalSim(scene, object_set, gui)
    sim.gripper.reset(Transform(Rotation.identity(), [0.0, 0.0, 0.0]))

    T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.1, 0.1, 0.0])
    #sim.gripper.move_tcp_xyz(T_grasp_pregrasp_world)
    # sim.gripper.move(0.0)
    # while True:
    #     pass
    finger_depth = sim.gripper.finger_depth

    #pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
    sim.reset(object_count)
    sim.save_state()

    # render synthetic depth images
    n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1
    depth_imgs, extrinsics = render_images(sim, n)

    # reconstrct point cloud using a subset of the images
    tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
    pc = tsdf.get_cloud()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    pc = pc.crop(bounding_box)


    if pc.is_empty():
        print("Point cloud empty, skipping scene")

    # Save point cloud file
    o3d.io.write_point_cloud("/home/walker/Documents/gpg/build/runtime.pcd", pc)

    for _ in range(GRASPS_PER_SCENE):
        # sample and evaluate a grasp point
        point, normal, grasp_depth = sample_grasp_point(pc, finger_depth)
        start = point
        end = point - normal * 0.05
        #print("start, end", start, end)
        arrow = create_arrow(start, end)
        lineset = vis_normal_pts(point, normal,grasp_depth)
        o3d.visualization.draw_geometries([pc, arrow, lineset])
        #o3d.visualization.draw_geometries([arrow])
        #print(point, normal)
        grasp, label = evaluate_grasp_point(sim, point, normal)



        # store the sample
        #write_grasp(args.root, scene_id, grasp, label)



def setup_mpi():
    workers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return workers, rank


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

def vis_normal_pts(point, normal, grasp_depth):
    import math
    #grasp_depth = 0.05
    points = [point - normal * 0.05, point + normal * grasp_depth]
    #print("line length", math.abs(grasp_depth))
    lines = [[0, 1]]
    colors = [[1, 0, 0]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
    #draw_geometries([line_set])


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1 and point[2] > 0.1  # make sure the normal is poitning upwards
    finger_depth = 0.06
    grasp_depth = np.random.uniform(-2*eps * finger_depth, (1.0 + eps) * finger_depth)
    #grasp_depth = finger_depth
    point = point #+ normal * grasp_depth
    return point, normal, grasp_depth

def normal_to_rotation(normal):
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    return R

def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)

        outcome, width = sim.execute_grasp(candidate, remove=False)
        print(outcome)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))



def place_table(height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("root", type=Path)
    # parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    # parser.add_argument("--object-set", type=str, default="blocks")
    # parser.add_argument("--num-grasps", type=int, default=10000)
    # parser.add_argument("--sim-gui", action="store_true")
    # args = parser.parse_args()
    # main()
    # test2()
    test3()