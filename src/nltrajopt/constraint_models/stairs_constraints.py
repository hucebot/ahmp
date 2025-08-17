from nltrajopt.constraint_models.abstract_constraint import *

def h(ee_x):
    return h

class StairsContactConstraints(AbstractConstraint):
    """Contact position and velocity constraints"""

    def __init__(self,x_start: float, dx: float, dz: float):
        """
        Args:
            x_start: where stairs start
            dx: Stair length
            dz: Stair height
        """
        self.x_start = x_start
        self.dz = dz
        self.dx = dx

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
                ee_x = data.oMf[frame_id].translation[0]
                ee_z = data.oMf[frame_id].translation[2]
                h = 0.0
                if ee_x >= self.x_start:
                    h = (((ee_x - self.x_start) // self.dx) + 1) * self.dz
                c[node_curr.c_z_ids[frame]] = [ee_x, ee_z - h]
            else:
                # Position constraint: p_contact - p_frame = 0
                c[node_curr.c_contact_kinematics_ids[frame]] = data.oMf[frame_id].translation - state_vars[node_curr.contact_pos_ids[frame]]

                ee_x = state_vars[node_curr.contact_pos_ids[frame]][0]
                ee_z = state_vars[node_curr.contact_pos_ids[frame]][2]
                h = 0.0
                if ee_x >= self.x_start:
                    h = (((ee_x - self.x_start) // self.dx) + 1) * self.dz
                c[node_curr.c_z_ids[frame]] = [ee_x, ee_z - h]

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
                jac[node_curr.c_z_ids[frame], node_curr.q_id] = [J[0, :], J[2, :]]
            else:
                # Position constraint Jacobians
                jac[node_curr.c_contact_kinematics_ids[frame], node_curr.q_id] = J[:3, :]
                jac[
                    node_curr.c_contact_kinematics_ids[frame],
                    node_curr.contact_pos_ids[frame],
                ] = -np.eye(3)

                # Height constraint Jacobian
                jac[node_curr.c_z_ids[frame], node_curr.contact_pos_ids[frame]] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

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
                if node.k < 15:
                    clb[node.c_z_ids[frame]] = [None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, None]
                else:
                    clb[node.c_z_ids[frame]] = [self.x_start, 0.0]
                    cub[node.c_z_ids[frame]] = [None, None]
            else:
                if node.k < 15:
                    clb[node.c_z_ids[frame]] = [None, 0.0]
                    cub[node.c_z_ids[frame]] = [None, 0.0]
                else:
                    clb[node.c_z_ids[frame]] = [self.x_start, 0.0]
                    cub[node.c_z_ids[frame]] = [None, 0.0]
