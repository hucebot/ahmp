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

class ChimneyFrictionConstraints(AbstractConstraint):
    """Friction cone constraints for chimney environment (pyramid approximation)"""

    def __init__(self, y: float, mu: float = 0.5, max_force: float = -1.0):
        """
        Args:
            y: y where the chimney walls begin
            mu: Friction coefficient
            bounds_fn: Optional function to compute dynamic bounds
        """
        self.y = y
        self.mu = mu
        self.max_force = max_force

    @property
    def name(self) -> str:
        return "chimney friction"

    def compute_constraints(self, node_curr, node_next, state_vars, c, model, data):
        """Compute friction cone constraints (4 linear inequalities per contact + 1 max force)"""
        q = q_tan2pin(state_vars[node_curr.q_id])

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        max_force = 4.0 * pin.computeTotalMass(model) * np.linalg.norm(model.gravity.vector) if self.max_force <= 0.0 else self.max_force
        for frame in node_curr.contact_phase_fnames:
            frame_id = model.getFrameId(frame)
            ee_y = state_vars[node_curr.contact_pos_ids[frame]][1]
            F_world = (data.oMf[frame_id].rotation @ state_vars[node_curr.forces_ids[frame]]).reshape((3, 1))

            n = None
            t1 = None
            t2 = None
            eps = 1e-5
            if  ee_y <= -self.y + eps or ee_y >= self.y - eps:
                if frame in LEFT_GRIPPER_FNAMES + LEFT_FOOT_FNAMES:
                    n = np.array([[0.0, -1.0, 0.0]]).T
                    t1 = np.array([[1.0, 0.0, 0.0]]).T
                    t2 = np.array([[0.0, 0.0, 1.0]]).T
                elif frame in RIGHT_GRIPPER_FNAMES + RIGHT_FOOT_FNAMES:
                    n = np.array([[0.0, 1.0, 0.0]]).T
                    t1 = np.array([[1.0, 0.0, 0.0]]).T
                    t2 = np.array([[0.0, 0.0, -1.0]]).T
            else:
                n = np.array([[0.0, 0.0, 1.0]]).T
                t1 = np.array([[1.0, 0.0, 0.0]]).T
                t2 = np.array([[0.0, 1.0, 0.0]]).T

            c[node_curr.c_friction_ids[frame]] = [
                (F_world.T @ (n * self.mu - t1))[0, 0],  # mu_Fz - Fx
                (F_world.T @ (n * self.mu + t1))[0, 0],  # mu_Fz + Fx
                (F_world.T @ (n * self.mu - t2))[0, 0],  # mu_Fz - Fy
                (F_world.T @ (n * self.mu + t2))[0, 0],  # mu_Fz + Fy
                (F_world.T @ n)[0, 0],  # Fz>=0
                max_force - ((F_world.T @ F_world) ** (0.5))[0, 0],  # ||F|| ≤ max_force
            ]

    def compute_jacobians(self, node_curr, node_next, w, jac, model, data):
        """Compute friction cone Jacobians"""
        q = q_tan2pin(w[node_curr.q_id])

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        for frame in node_curr.contact_phase_fnames:
            frame_id = model.getFrameId(frame)
            ee_y = w[node_curr.contact_pos_ids[frame]][1]
            R = data.oMf[frame_id].rotation
            Jf = pin.computeFrameJacobian(model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
            Jf[:, :6] = Jf[:, :6] @ pin.Jexp6(w[node_curr.q_id][:6])

            f = w[node_curr.forces_ids[frame]].reshape((3, 1))
            F_world = R @ f

            n = None
            t1 = None
            t2 = None
            eps = 1e-5
            if  ee_y <= -self.y + eps or ee_y >= self.y - eps:
                if frame in LEFT_GRIPPER_FNAMES + LEFT_FOOT_FNAMES:
                    n = np.array([[0.0, -1.0, 0.0]]).T
                    t1 = np.array([[1.0, 0.0, 0.0]]).T
                    t2 = np.array([[0.0, 0.0, 1.0]]).T
                elif frame in RIGHT_GRIPPER_FNAMES + RIGHT_FOOT_FNAMES:
                    n = np.array([[0.0, 1.0, 0.0]]).T
                    t1 = np.array([[1.0, 0.0, 0.0]]).T
                    t2 = np.array([[0.0, 0.0, -1.0]]).T
            else:
                n = np.array([[0.0, 0.0, 1.0]]).T
                t1 = np.array([[1.0, 0.0, 0.0]]).T
                t2 = np.array([[0.0, 1.0, 0.0]]).T

            tmp = -R @ hat(f) @ Jf[3:, :]

            jac[node_curr.c_friction_ids[frame].start + 0, node_curr.forces_ids[frame]] = (n * self.mu - t1).T @ R
            jac[node_curr.c_friction_ids[frame].start + 1, node_curr.forces_ids[frame]] = (n * self.mu + t1).T @ R
            jac[node_curr.c_friction_ids[frame].start + 2, node_curr.forces_ids[frame]] = (n * self.mu - t2).T @ R
            jac[node_curr.c_friction_ids[frame].start + 3, node_curr.forces_ids[frame]] = (n * self.mu + t2).T @ R
            jac[node_curr.c_friction_ids[frame].start + 4, node_curr.forces_ids[frame]] = n.T @ R
            jac[node_curr.c_friction_ids[frame].start + 5, node_curr.forces_ids[frame]] = (
                -((F_world) / ((F_world.T @ F_world) ** (0.5))).T @ R
            )

            jac[node_curr.c_friction_ids[frame].start + 0, node_curr.q_id] = (n * self.mu - t1).T @ tmp
            jac[node_curr.c_friction_ids[frame].start + 1, node_curr.q_id] = (n * self.mu + t1).T @ tmp
            jac[node_curr.c_friction_ids[frame].start + 2, node_curr.q_id] = (n * self.mu - t2).T @ tmp
            jac[node_curr.c_friction_ids[frame].start + 3, node_curr.q_id] = (n * self.mu + t2).T @ tmp
            jac[node_curr.c_friction_ids[frame].start + 4, node_curr.q_id] = n.T @ tmp
            jac[node_curr.c_friction_ids[frame].start + 5, node_curr.q_id] = -((F_world) / ((F_world.T @ F_world) ** (0.5))).T @ tmp

    def get_structure_ids(self, node_curr, node_next, row_ids, col_ids):
        """Friction cone sparsity pattern"""

        for frame in node_curr.contact_phase_fnames:
            extend_ids_lists(
                row_ids,
                col_ids,
                node_curr.c_friction_ids[frame],
                node_curr.forces_ids[frame],
            )
            extend_ids_lists(row_ids, col_ids, node_curr.c_friction_ids[frame], node_curr.q_id)

    def get_bounds(
        self,
        node: "Node",
        lb: np.ndarray,
        ub: np.ndarray,
        clb: np.ndarray,
        cub: np.ndarray,
        model: pin.Model,
    ) -> None:
        """Friction bounds (0 ≤ constraint ≤ ∞)"""
        for frame in node.contact_phase_fnames:
            cub[node.c_friction_ids[frame]] = [None] * 6

class ChimneyContactConstraints(AbstractConstraint):
    """Contact position and velocity constraints"""

    def __init__(self, y: float, k: int=10):
        """
        Args:
            y: Where the chimney wall begins
            k: The node index after which the feet must stick to the ground
        """
        self.k = k
        self.z = 0
        self.y = y

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
                val = 0.0
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    val = ee_y
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    val = ee_z - self.z if node_curr.k < self.k else ee_y
                else:
                    print("Error")
                    exit()

                c[node_curr.c_z_ids[frame]] = val
            else:
                # Position constraint: p_contact - p_frame = 0
                c[node_curr.c_contact_kinematics_ids[frame]] = data.oMf[frame_id].translation - state_vars[node_curr.contact_pos_ids[frame]]

                ee_y = state_vars[node_curr.contact_pos_ids[frame]][1]
                ee_z = state_vars[node_curr.contact_pos_ids[frame]][2]
                val = 0.0
                if frame in LEFT_GRIPPER_FNAMES:
                    val = ee_y - self.y
                elif frame in RIGHT_GRIPPER_FNAMES:
                    val = ee_y + self.y
                elif frame in LEFT_FOOT_FNAMES:
                    val = ee_z - self.z if node_curr.k < self.k else ee_y - self.y
                elif frame in RIGHT_FOOT_FNAMES:
                    val = ee_z - self.z if node_curr.k < self.k else ee_y + self.y
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
                    deriv = J[1, :]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    deriv = J[2, :] if node_curr.k < self.k else J[1, :]
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
                deriv = np.zeros((1, 3))
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    deriv[0][1] = 1.0
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    if node_curr.k < self.k:
                        deriv[0][2] = 1.0
                    else:
                        deriv[0][1] = 1.0

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
                if frame in LEFT_GRIPPER_FNAMES + RIGHT_GRIPPER_FNAMES:
                    clb[node.c_z_ids[frame]] = [-self.y]
                    cub[node.c_z_ids[frame]] = [self.y]
                elif frame in LEFT_FOOT_FNAMES + RIGHT_FOOT_FNAMES:
                    if node.k < self.k:
                        clb[node.c_z_ids[frame]] = [0.0]
                        cub[node.c_z_ids[frame]] = [None]
                    else:
                        clb[node.c_z_ids[frame]] = [-self.y]
                        cub[node.c_z_ids[frame]] = [self.y]
            else:
                clb[node.c_z_ids[frame]] = [0.0]
                cub[node.c_z_ids[frame]] = [0.0]
