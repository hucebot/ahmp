from nltrajopt.constraint_models.abstract_constraint import *

LEFT_GRIPPER_FNAMES = [
    "gripper_left_inner_single_link",
    "left_hand_point_contact",
]

RIGHT_GRIPPER_FNAMES = [
    "gripper_right_inner_single_link",
    "right_hand_point_contact",
]

class VaultContactConstraints(AbstractConstraint):
    """Contact position and velocity constraints"""

    def __init__(self, z: float, x_min: float, x_max: float):
        """
        Args:
            z: Height of vault object
            x_min: Where the vault begins
            x_min: Where the vault ends
        """
        self.z = z
        self.x_min = x_min
        self.x_max = x_max

    @property
    def name(self) -> str:
        return "vault_contact"

    def compute_constraints(self, node_curr: Node, node_next, state_vars, c, model, data):
        """Compute contact position and velocity constraints"""
        q = q_tan2pin(state_vars[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for frame in node_curr.contact_fnames:
            frame_id = model.getFrameId(frame)

            if frame not in node_curr.contact_phase_fnames:
                # Height constraint: p_z ≥ ground
                ee_z = data.oMf[frame_id].translation[2]
                h = 0.0
                c[node_curr.c_z_ids[frame]] = np.array([0.0, 0.0,  ee_z - h])
            else:
                # Position constraint: p_contact - p_frame = 0
                c[node_curr.c_contact_kinematics_ids[frame]] = data.oMf[frame_id].translation - state_vars[node_curr.contact_pos_ids[frame]]

                ee_x = state_vars[node_curr.contact_pos_ids[frame]][0]
                ee_y = state_vars[node_curr.contact_pos_ids[frame]][1]
                ee_z = state_vars[node_curr.contact_pos_ids[frame]][2]
                h = self.z if ee_x > self.x_min and ee_x < self.x_max else 0.0
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    c[node_curr.c_z_ids[frame]] = [ee_x, 0.0,  ee_z - h]
                else:
                    # foot_x_c = (-ee_x - self.x_min) * (ee_x - self.x_max)
                    c[node_curr.c_z_ids[frame]] = [0.0, 0.0,  ee_z - h]

                # Velocity constraint: p_next - p_curr = 0 (if next node exists)
                if node_next and frame in node_next.contact_phase_fnames:
                    c[node_curr.c_vel_ids[frame]] = (
                        state_vars[node_next.contact_pos_ids[frame]] - state_vars[node_curr.contact_pos_ids[frame]]
                    )

    def compute_jacobians(self, node_curr: Node, node_next, w, jac, model, data):
        """Compute contact Jacobians"""

        q = q_tan2pin(w[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for frame in node_curr.contact_fnames:
            frame_id = model.getFrameId(frame)
            J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J[:, :6] = J[:, :6] @ pin.Jexp6(w[node_curr.q_id][:6])  # Convert to tangent space

            if frame not in node_curr.contact_phase_fnames:
                jac_x = np.zeros(J[2, :].shape)
                deriv = np.vstack((jac_x, jac_x, J[2, :]))
                jac[node_curr.c_z_ids[frame], node_curr.q_id] = deriv

            else:
                # Position constraint Jacobians
                jac[node_curr.c_contact_kinematics_ids[frame], node_curr.q_id] = J[:3, :]
                jac[
                    node_curr.c_contact_kinematics_ids[frame],
                    node_curr.contact_pos_ids[frame],
                ] = -np.eye(3)

                # Height constraint Jacobian
                dx = [0.0, 0.0, 0.0]
                dy = [0.0, 0.0, 0.0]
                dz = [0.0, 0.0, 1.0]

                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    dx = [1.0, 0.0, 0.0]
                    dy = [0.0, 0.0, 0.0]

                deriv = np.vstack((dx, dy, dz))
                jac[node_curr.c_z_ids[frame], node_curr.contact_pos_ids[frame]] = deriv

                # Velocity constraint Jacobians
                if node_next and frame in node_next.contact_phase_fnames:
                    jac[node_curr.c_vel_ids[frame], node_next.contact_pos_ids[frame]] = np.eye(3)
                    jac[node_curr.c_vel_ids[frame], node_curr.contact_pos_ids[frame]] = -np.eye(3)

    def get_structure_ids(self, node_curr, node_next, row_ids, col_ids):
        """Contact constraints sparsity pattern"""

        for frame in node_curr.contact_fnames:

            if frame not in node_curr.contact_phase_fnames:
                extend_ids_lists(row_ids, col_ids, node_curr.c_z_ids[frame], node_curr.q_id)
            else:
                extend_ids_lists(
                    row_ids,
                    col_ids,
                    node_curr.c_contact_kinematics_ids[frame],
                    node_curr.q_id,
                )
                extend_ids_lists(
                    row_ids,
                    col_ids,
                    node_curr.c_contact_kinematics_ids[frame],
                    node_curr.contact_pos_ids[frame],
                )
                extend_ids_lists(
                    row_ids,
                    col_ids,
                    node_curr.c_z_ids[frame],
                    node_curr.contact_pos_ids[frame],
                )

                if node_next is not None and frame in node_next.contact_phase_fnames:
                    extend_ids_lists(
                        row_ids,
                        col_ids,
                        node_curr.c_vel_ids[frame],
                        node_next.contact_pos_ids[frame],
                    )
                    extend_ids_lists(
                        row_ids,
                        col_ids,
                        node_curr.c_vel_ids[frame],
                        node_curr.contact_pos_ids[frame],
                    )

    def get_bounds(
        self,
        node: "Node",
        lb: np.ndarray,
        ub: np.ndarray,
        clb: np.ndarray,
        cub: np.ndarray,
        model: pin.Model,
    ) -> None:
        """Contact bounds"""

        for frame in node.contact_fnames:
            if frame not in node.contact_phase_fnames:
                clb[node.c_z_ids[frame]] = [None, None, 0.0]
                cub[node.c_z_ids[frame]] = [None, None, None]
            else:
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    clb[node.c_z_ids[frame]] = [self.x_min, None, 0.0]
                    cub[node.c_z_ids[frame]] = [self.x_max, None, 0.0]
                else:
                    clb[node.c_z_ids[frame]] = [None, None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, None, 0.0]
