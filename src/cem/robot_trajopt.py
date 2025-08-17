import time

import numpy as np
import pinocchio as pin

from nltrajopt.trajectory_optimization import NLTrajOpt
from nltrajopt.contact_scheduler import ContactScheduler
from nltrajopt.node import Node
from nltrajopt.constraint_models import *
from nltrajopt.cost_models import *

from robots.talos.TalosWrapper import Talos
from robots.g1.G1Wrapper import G1
from robots.go2.Go2Wrapper import Go2


def calc_n_violations(t_x, t_lb, t_ub, tol):
    x = np.array(t_x)
    # Turn None to -inf and inf respectively
    lb = np.array([item if item is not None else -np.inf for item in t_lb])
    ub = np.array([item if item is not None else np.inf for item in t_ub])
    violations = np.logical_or(x < lb - tol, x > ub + tol)
    return np.sum(violations)


def cem_to_trajopt(robot, params, xd, xc):
    contacts_dict = {}
    # if params.scenario == "vault":
    #     contacts_dict = {
    #         "l_foot": robot.left_foot_frames,
    #         "r_foot": robot.right_foot_frames,
    #         "l_gripper": robot.left_gripper_frames,
    #         "r_gripper": robot.right_gripper_frames,
    #     }
    if params.scenario == "chimney" or "handrails":
        contacts_dict = {
            "l_foot": robot.left_foot_line_frames,
            "r_foot": robot.right_foot_line_frames,
            "l_gripper": robot.left_gripper_frames,
            "r_gripper": robot.right_gripper_frames,
        }

    contact_scheduler = ContactScheduler(
        robot.model, dt=params.dt, contact_frame_dict=contacts_dict
    )

    # if params.scenario == "vault":
    #     contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
    #     contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper"], 0.2)
    if params.scenario == "chimney":
        contact_scheduler.add_phase(["l_foot", "r_foot"], 0.5)
        contact_scheduler.add_phase([], 0.2)
        contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)
    elif params.scenario == "handrails":
        contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
    else:
        print("Error: Scenario not specified")
        exit()

    for i in range(params.dim_discrete):
        contact_ees = []

        if xd[i] & 0b0001:
            contact_ees.extend(["l_foot"])
        if xd[i] & 0b0010:
            contact_ees.extend(["r_foot"])
        if xd[i] & 0b0100:
            contact_ees.extend(["l_gripper"])
        if xd[i] & 0b1000:
            contact_ees.extend(["r_gripper"])

        contact_scheduler.add_phase(contact_ees, xc[i])

    if params.scenario == "vault" or params.scenario == "handrails":
        contact_scheduler.add_phase(["l_foot", "r_foot"], 0.3)
    elif params.scenario == "chimney":
        contact_scheduler.add_phase(["l_foot", "r_foot", "l_gripper", "r_gripper"], 0.2)

    return contact_scheduler.contact_sequence_fnames

def solve_trajopt(input):
    xd = input[0].copy()
    xc = input[1].copy()
    params = input[2]

    robot = None
    if params.robot == "talos":
        robot = Talos()
    elif params.robot == "g1":
        robot = G1()
    elif params.robot == "go2":
        robot = Go2()
    else:
        print("Error, robot type not supported")
        exit()

    q = robot.go_neutral()

    if params.trajopt_type == "whole_body":
        constraints = [WholeBodyDynamics()]
    elif params.trajopt_type == "centroidal":
        constraints = [CentroidalDynamics()]
    else:
        print("Error: Valid trajopt_type parameters are: whole_body, centroidal")
        exit()

    contacts_frame_names = []
    if params.scenario == "vault":
        contact_frame_names = (
            robot.left_gripper_frames
            + robot.right_gripper_frames
            + robot.left_foot_frames
            + robot.right_foot_frames
        )
    elif params.scenario == "chimney" or params.scenario == "handrails":
        contact_frame_names = (
            robot.left_gripper_frames
            + robot.right_gripper_frames
            + robot.left_foot_line_frames
            + robot.right_foot_line_frames
        )

    stages = []

    frame_contact_seq = cem_to_trajopt(robot, params, xd, xc)
    for contact_phase_fnames in frame_contact_seq:
        stage_node = Node(
            robot.model.nv,
            contact_phase_fnames=contact_phase_fnames,
            contact_fnames=contact_frame_names,
            node_type=params.scenario,
        )

        dyn_const = None
        if params.trajopt_type == "whole_body":
            dyn_const = WholeBodyDynamics()
        elif params.trajopt_type == "centroidal":
            dyn_const = CentroidalDynamics()
        stage_node.dynamics_type = dyn_const.name

        stage_node.constraints_list.extend(
            [
                dyn_const,
                TimeConstraint(min_dt=params.dt, max_dt=params.dt, total_time=None),
                SemiEulerIntegration(),
            ]
        )

        if params.scenario == "vault":
            stage_node.constraints_list.extend(
                [
                    FrictionConstraints(params.vault_mu),
                    VaultContactConstraints(params.vault_z, params.vault_x_min, params.vault_x_max),
                ]
            )
        elif params.scenario == "chimney":
            stage_node.constraints_list.extend(
                [
                    ChimneyFrictionConstraints(y=params.chimney_y, mu=params.chimney_mu),
                    ChimneyContactConstraints(y=params.chimney_y),
                ]
            )
        elif params.scenario == "handrails":
            stage_node.constraints_list.extend(
                [
                    FrictionConstraints(params.rail_mu),
                    HandrailsContactConstraints(rail_y=params.rail_y, rail_z=params.rail_z, rail_w=params.rail_w),
                ]
            )

        # stage_node.costs_list.extend([JointAccelerationCost(np.zeros(robot.model.nv - 6), np.eye(robot.model.nv - 6) * 1e-7)])

        stages.append(stage_node)

    opti = NLTrajOpt(model=robot.model, nodes=stages, dt=params.dt)

    opti.set_initial_pose(q)
    qf = np.copy(q)
    if params.scenario == "vault" or params.scenario == "handrails":
        qf[0:2] = np.array(params.base_target_xy)
        opti.set_target_pose(qf)
    elif params.scenario == "chimney":
        qf[2] += np.array(params.base_target_dz)
        opti.set_base_target(qf)

    result = opti.solve(params.max_trajopt_iter, params.viol_tol, params.parallel, params.print_level)

    n_g_viol = calc_n_violations(result["info"]["g"], opti.clb, opti.cub, params.viol_tol)
    result["n_viol"] = n_g_viol

    return result
