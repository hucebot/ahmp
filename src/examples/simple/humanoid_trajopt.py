import time

import numpy as np
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *
from terrain.terrain_grid import TerrainGrid

from robots.g1.G1Wrapper import G1
from robots.talos.TalosWrapper import Talos

from visualiser.visualiser import TrajoptVisualiser

import imageio

VIS = True
DT = 0.05

robot = Talos()
# robot = G1()
q = robot.go_neutral()

contacts_dict = {
    "l_foot": robot.left_foot_frames,
    "r_foot": robot.right_foot_frames,
    # "l_foot": robot.left_foot_line_frames,
    # "r_foot": robot.right_foot_line_frames,
    # "l_foot": robot.left_foot_point_frames,
    # "r_foot": robot.right_foot_point_frames,
    "l_gripper": robot.left_gripper_frames,
    # "r_gripper": robot.right_gripper_frames,
}

contact_scheduler = ContactScheduler(
    robot.model, dt=DT, contact_frame_dict=contacts_dict
)

chimney_mu = 0.9
# chimney_y = 0.5
chimney_y = 0.35

x_start = 0.2
stair_dx = 0.1
stair_dz = 0.1

rail_y = 0.45
rail_z = 0.9
rail_w = 0.01

contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper"], 0.3)
contact_scheduler.add_phase(["l_gripper"], 0.8)
contact_scheduler.add_phase(["l_foot", "r_foot"], 0.5)

frame_contact_seq = contact_scheduler.contact_sequence_fnames

contact_frame_names = (
    robot.left_gripper_frames
    + robot.right_gripper_frames
    + robot.left_foot_frames
    + robot.right_foot_frames
    # + robot.left_foot_line_frames
    # + robot.right_foot_line_frames
)

stages = []

for contact_phase_fnames in frame_contact_seq:
    stage_node = Node(
        robot.model.nv,
        contact_phase_fnames=contact_phase_fnames,
        contact_fnames=contact_frame_names,
        node_type="vault",
    )

    dyn_const = WholeBodyDynamics()
    stage_node.dynamics_type = dyn_const.name

    stage_node.constraints_list.extend(
        [
            dyn_const,
            TimeConstraint(min_dt=DT, max_dt=DT, total_time=None),
            SemiEulerIntegration(),
            # EulerIntegration(),
            FrictionConstraints(mu=0.8),
            VaultContactConstraints(0.7, 0.3, 0.55),
            # HandrailsContactConstraints(rail_y=rail_y, rail_z=rail_z, rail_w=rail_w),
            # ChimneyFrictionConstraints(y=chimney_y, mu=chimney_mu),
            # ChimneyContactConstraints(y=chimney_y),
            # StairsContactConstraints(x_start=x_start, dx=stair_dx, dz=stair_dz),
        ]
    )

    # stage_node.costs_list.extend([ConfigurationCost(np.zeros(robot.model.nv - 6), np.eye(robot.model.nv - 6) * 1e-8)])
    stage_node.costs_list.extend(
        [
            JointAccelerationCost(
                np.zeros(robot.model.nv - 6), np.eye(robot.model.nv - 6) * 1e-7
            )
        ]
    )

    stages.append(stage_node)


opti = NLTrajOpt(model=robot.model, nodes=stages, dt=DT)

opti.set_initial_pose(q)
qf = np.copy(q)
qf[0] = 1.0
# qf[2] = 0.2 + (1. // 0.3) * 0.1
# qf[2] += 0.1
# qf[5] = 0.383
# qf[3] = 0.924
opti.set_target_pose(qf)
# opti.set_base_target(qf)
# opti.set_base_xy_target(qf)
# opti.set_base_xyz_target(qf)

result = opti.solve(max_iter=100, tol=1e-3, parallel=True, print_level=5)

# NLTrajOpt.save_solution(result, "talos_simple", "src/examples/simple")

K = len(result["nodes"])
dts = [result["nodes"][k]["dt"] for k in range(K)]
state_trajectory = [result["nodes"][k]["q"] for k in range(K)]
forces = [result["nodes"][k]["forces"] for k in range(K)]

if VIS:
    tvis = TrajoptVisualiser(robot)

    interp_states = state_trajectory
    tvis.display_robot_q(robot, state_trajectory[0])

    # Visualise the terrain
    # tvis.load_terrain(terrain)
    tvis.load_vault_obstacle(0.65, 0.35, 0.55)
    # tvis.load_chimney_walls(0.5)
    # tvis.load_stairs(x_start, stair_dx, stair_dz)
    # tvis.load_handrails(rail_y, rail_z, rail_w)

    time.sleep(1)
    while True:
        for i in range(len(interp_states)):
            time.sleep(dts[i])
            tvis.display_robot_q(robot, interp_states[i])
            # tvis.update_forces(robot, forces[i], 0.01)
            img = tvis.vis.captureImage()

        tvis.update_forces(robot, {}, 0.01)
        time.sleep(2)
