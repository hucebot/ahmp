from nltrajopt.constraint_models.abstract_constraint import *

LEFT_FOOT_FNAMES = [
    "left_mm",
    "left_mp",
    "left_pp",
    "left_pm",
    "left_lp",
    "left_lm",
    "left_foot_upper_left",
    "left_foot_upper_right",
    "left_foot_lower_left",
    "left_foot_lower_right",
    "left_foot_line_contact_upper",
    "left_foot_line_contact_lower",
]

RIGHT_FOOT_FNAMES = [
    "right_mp",
    "right_mm",
    "right_pp",
    "right_pm",
    "right_lp",
    "right_lm",
    "right_foot_upper_left",
    "right_foot_upper_right",
    "right_foot_lower_left",
    "right_foot_lower_right",
    "right_foot_line_contact_upper",
    "right_foot_line_contact_lower",
]

LEFT_GRIPPER_FNAMES = [
    "gripper_left_inner_single_link",
    "left_hand_point_contact",
]

RIGHT_GRIPPER_FNAMES = [
    "gripper_right_inner_single_link",
    "right_hand_point_contact",
]

class HandrailsContactConstraints(AbstractConstraint):
    def __init__(self, rail_y: float, rail_z: float, rail_w: float):
        """
        Args:
            y: Where the chimney wall begins
            k: The node index after which the feet must stick to the ground
        """
        self.rail_y = rail_y
        self.rail_z = rail_z
        self.rail_w = rail_w

    @property
    def name(self) -> str:
        return "chimney_contact"

    def compute_constraints(self, node_curr: Node, node_next, state_vars, c, model, data):
        """Compute contact position and velocity constraints"""
        q = q_tan2pin(state_vars[node_curr.q_id])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for frame in node_curr.contact_fnames:
            frame_id = model.getFrameId(frame)

            if frame not in node_curr.contact_phase_fnames:
                # Height constraint: p_z ≥ ground
                ee_y = data.oMf[frame_id].translation[1]
                ee_z = data.oMf[frame_id].translation[2]
                val = []
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    val = [0.0, ee_z]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    val = [0.0, ee_z]
                else:
                    print("Error")
                    exit()

                c[node_curr.c_z_ids[frame]] = val
            else:
                # Position constraint: p_contact - p_frame = 0
                c[node_curr.c_contact_kinematics_ids[frame]] = data.oMf[frame_id].translation - state_vars[node_curr.contact_pos_ids[frame]]

                ee_y = state_vars[node_curr.contact_pos_ids[frame]][1]
                ee_z = state_vars[node_curr.contact_pos_ids[frame]][2]
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    val = [ee_y, ee_z]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    val = [0.0, ee_z]
                else:
                    print("Error")
                    exit()
                c[node_curr.c_z_ids[frame]] = val

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
                deriv = None
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    deriv = [0.0 * J[1, :], J[2, :]]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    deriv = [J[1, :], J[2, :]]
                else:
                    print("MISTAKE", frame)
                    exit()
                jac[node_curr.c_z_ids[frame], node_curr.q_id] = deriv
            else:
                # Position constraint Jacobians
                jac[node_curr.c_contact_kinematics_ids[frame], node_curr.q_id] = J[:3, :]
                jac[
                    node_curr.c_contact_kinematics_ids[frame],
                    node_curr.contact_pos_ids[frame],
                ] = -np.eye(3)

                # Height constraint Jacobian
                jac[node_curr.c_z_ids[frame], node_curr.contact_pos_ids[frame]] = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

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
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    clb[node.c_z_ids[frame]] = [None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, None]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    clb[node.c_z_ids[frame]] = [None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, None]
            else:
                if frame in LEFT_GRIPPER_FNAMES:
                    clb[node.c_z_ids[frame]] = [self.rail_y, self.rail_z]
                    cub[node.c_z_ids[frame]] = [self.rail_y + self.rail_w, self.rail_z]
                elif frame in RIGHT_GRIPPER_FNAMES:
                    clb[node.c_z_ids[frame]] = [-self.rail_y - self.rail_w, self.rail_z]
                    cub[node.c_z_ids[frame]] = [-self.rail_y + self.rail_w, self.rail_z]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    clb[node.c_z_ids[frame]] = [None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, 0.0]
