import numpy as np
import pinocchio as pin
import cyipopt


from nltrajopt.wbtrajopt.node import Node
from nltrajopt.wbtrajopt.node_functions import *


class WBTrajOpt:
    def __init__(self, model_, dt_, contact_phases_):
        self.model = model_
        self.data = self.model.createData()

        self.dt = dt_
        self.K = contact_phases_.shape[0]
        self.q_final = None
        self.R_q = np.eye(self.model.nv) * 10e-9 * 0
        self.R_vq = np.eye(self.model.nv) * 10e-3 * 0
        self.R_aq = np.eye(self.model.nv) * 10e-6 * 0
        self.R_forces = np.eye(3) * 10e-4 * 0

        self.nodes: list[Node] = []
        self.vars_dim = 0
        self.cons_dim = 0
        for k in range(self.K):
            contacts = []
            if contact_phases_[k] == 0:
                contacts = ["left", "right"]
            if contact_phases_[k] == 1:
                contacts = ["left"]
            if contact_phases_[k] == 2:
                contacts = ["right"]

            node = Node(
                k,
                v_id=self.vars_dim,
                c_id=self.cons_dim,
                nv=model_.nv,
                dt=dt_,
                contact_phase=contacts,
            )
            self.nodes.append(node)
            self.vars_dim += node.x_dim
            self.cons_dim += node.c_dim

        self.x0 = np.zeros((self.vars_dim,))
        self.c = np.zeros((self.cons_dim,))
        self.grad = np.zeros((self.cons_dim, self.vars_dim))

        self.lb = [None] * self.vars_dim
        self.ub = [None] * self.vars_dim
        self.clb = [0] * self.cons_dim
        self.cub = [0] * self.cons_dim

        for node in self.nodes:
            nd_vars_bounds(node, self.lb, self.ub, self.model)
            nd_cons_bounds(node, self.clb, self.cub, self.model)

        self.row_ids = []
        self.col_ids = []

        for k in range(self.K - 1):
            const_grad_ids(self.nodes[k], self.nodes[k + 1], self.row_ids, self.col_ids)

    def set_initial_pose(self, q_):
        q_tan = q_pin2tan(q_)
        self.lb[self.nodes[0].q_id] = q_tan
        self.ub[self.nodes[0].q_id] = q_tan
        self.lb[self.nodes[0].vq_id] = np.zeros((self.nodes[0].nv,))
        self.ub[self.nodes[0].vq_id] = np.zeros((self.nodes[0].nv,))
        # self.lb[self.nodes[0].aq_id] = np.zeros((self.nodes[0].nv,))
        # self.ub[self.nodes[0].aq_id] = np.zeros((self.nodes[0].nv,))

        g = self.model.gravity.linear[2]
        mass = pin.computeTotalMass(self.model, self.data)

        for node in self.nodes:
            self.x0[node.q_id] = q_tan
            N_contact_frames = len(node.contact_frames)
            for joint in list(node.force_frames.keys()):
                for frame in node.force_frames[joint]:
                    if frame in node.contact_frames:
                        self.x0[node.forces_ids[frame].start + 2] = (
                            80  # -mass * g / N_contact_frames
                        )

    def set_target_pose(self, q_):
        q_tan = q_pin2tan(q_)

        self.q_final = q_tan

        self.lb[self.nodes[-1].q_id] = q_tan
        self.ub[self.nodes[-1].q_id] = q_tan
        self.lb[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
        self.ub[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
        # self.lb[self.nodes[-1].aq_id] = np.zeros((self.nodes[0].nv,))
        # self.ub[self.nodes[-1].aq_id] = np.zeros((self.nodes[0].nv,))

        q0 = q_tan2pin(self.x0[self.nodes[0].q_id])

        for node in self.nodes:
            self.x0[node.q_id] = q_pin2tan(
                pin.interpolate(self.model, q0, q_, node.k / self.K)
            )

    def objective(self, ww):
        obj = 0

        if self.q_final is not None:
            obj += 0.5 * np.sum(
                (ww[n.q_id] - self.q_final).T @ self.R_q @ (ww[n.q_id] - self.q_final)
                for n in self.nodes[:-1]
            )
        obj += 0.5 * np.sum(
            ww[n.vq_id].T @ self.R_vq @ ww[n.vq_id] for n in self.nodes[:-1]
        )
        obj += 0.5 * np.sum(
            ww[n.aq_id].T @ self.R_aq @ ww[n.aq_id] for n in self.nodes[:-1]
        )
        for node in self.nodes:
            for joint in list(node.force_frames.keys()):
                for frame in node.force_frames[joint]:
                    obj += (
                        0.5
                        * ww[node.forces_ids[frame]]
                        @ self.R_forces
                        @ ww[node.forces_ids[frame]]
                    )
        return obj

    def gradient(self, ww):
        grad = np.zeros_like(ww)

        for node in self.nodes:
            if self.q_final is not None:
                grad[node.vq_id] = (
                    (ww[node.q_id] - self.q_final).T
                    @ self.R_q
                    @ (ww[node.q_id] - self.q_final)
                )

            grad[node.vq_id] = ww[node.vq_id].T @ self.R_vq
            grad[node.aq_id] = ww[node.aq_id].T @ self.R_aq

            for joint in list(node.force_frames.keys()):
                for frame in node.force_frames[joint]:
                    grad[node.forces_ids[frame]] = (
                        self.R_forces @ ww[node.forces_ids[frame]]
                    )
        return grad

    def constraints(self, ww):
        c = np.zeros((self.cons_dim,))

        for k in range(self.K - 1):
            const(self.nodes[k], self.nodes[k + 1], ww, c, self.model, self.data)

        return c

    def jac_test(self, ww):
        jac = np.zeros((self.cons_dim, self.vars_dim))

        for k in range(self.K - 1):
            const_grad(self.nodes[k], self.nodes[k + 1], ww, jac, self.model, self.data)

        return jac

    def jacobian(self, ww):
        jac = np.zeros((self.cons_dim, self.vars_dim))

        for k in range(self.K - 1):
            const_grad(self.nodes[k], self.nodes[k + 1], ww, jac, self.model, self.data)

        rows, cols = self.jacobianstructure()

        return jac[rows, cols]

    def jacobianstructure(self):
        return (np.array(self.row_ids), np.array(self.col_ids))

    def decode_solution(self):
        qs = []
        forces = []
        for node in self.nodes:
            qs.append(q_tan2pin(self.sol[node.q_id]))
            f_dict = {}
            for joint in list(node.force_frames.keys()):
                for frame in node.force_frames[joint]:
                    frame_id = self.model.getFrameId(frame)
                    f_dict.update({frame_id: self.sol[node.forces_ids[frame]]})
            forces.append(f_dict)

        return qs, forces

    def savetofile(self):
        states = self.sol
        cons = self.constraints(self.sol)

        # Write to file
        with open("data/trajopt/nodes.txt", "w") as f:
            f.write("qs\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in states[node.q_id]) + "\n")

            f.write("\n\nvqs\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in states[node.vq_id]) + "\n")

            f.write("\n\naqs\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in states[node.aq_id]) + "\n")

            f.write("\n\nc_qs\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in cons[node.c_q_id]) + "\n")

            f.write("\n\nc_vqs\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in cons[node.c_vq_id]) + "\n")

            f.write("\n\nc_taus\n")
            for node in self.nodes:
                f.write(" ".join(f"{x:8.2f}" for x in cons[node.c_tau_id]) + "\n")

            f.write("\n\nc_contact\n")
            # Collect frames and their corresponding nodes
            frame_to_nodes = {}

            for node in self.nodes:
                for frame in node.contact_frames:
                    if frame not in frame_to_nodes:
                        frame_to_nodes[frame] = {}
                    frame_to_nodes[frame][node] = cons[node.c_cont_ids[frame]]

            # Write to file
            for frame, nodes in frame_to_nodes.items():
                f.write("\t" + f"{frame}\n")  # Write frame
                for node, values in nodes.items():
                    f.write(
                        "\t" + " ".join(f"{x:8.2f}" for x in values) + "\n"
                    )  # Indent values

            f.write("\n\nc_contact\n")
            # Collect frames and their corresponding nodes
            frame_to_nodes = {}

            for node in self.nodes:
                for joint in list(node.contacts.keys()):
                    for frame in node.contacts[joint]:
                        if frame not in frame_to_nodes:
                            frame_to_nodes[frame] = {}
                        frame_to_nodes[frame][node] = states[node.forces_ids[frame]]

            # Write to file
            for frame, nodes in frame_to_nodes.items():
                f.write("\t" + f"{frame}\n")  # Write frame
                for node, values in nodes.items():
                    f.write(
                        "\t" + " ".join(f"{x:8.2f}" for x in values) + "\n"
                    )  # Indent values
