import numpy as np
import pinocchio as pin
import time
from datetime import datetime
from pathlib import Path
import json
import cyipopt
from typing import List, Dict, Tuple

from nltrajopt.centrajopt.node import Node
from nltrajopt.centrajopt.node_functions import (
    q_pin2tan,
    q_tan2pin,
    nd_vars_bounds,
    nd_cons_bounds,
    extend_ids_lists,
    const_grad_ids,
    const,
    const_grad,
)


class CenTrajOpt:
    def __init__(self, model: pin.Model, dt: float, contact_phases: np.ndarray):
        """Initialize trajectory optimization problem.

        Args:
            model: Pinocchio robot model
            dt: Initial time step guess
            contact_phases: Array specifying contact states for each phase
        """
        self.model = model
        self.data = model.createData()
        self.dt = dt
        self.K = contact_phases.shape[0]

        # Cost function weights
        self.R_q = np.eye(model.nv - 6) * 1e-6 * 0
        self.R_vq = np.eye(model.nv) * 1e-6 * 0
        self.R_aq = np.eye(model.nv) * 1e-3 * 0
        self.R_forces = np.eye(3) * 1e-9 * 0

        # Reference and target configurations
        self.q_ref = None

        # Initialize nodes and problem dimensions
        self.nodes = self._initialize_nodes(contact_phases)
        self.vars_dim, self.cons_dim = self._get_problem_dimensions()

        # Initialize optimization problem structures
        self.x0 = np.zeros(self.vars_dim)
        self.lb, self.ub = self._initialize_variable_bounds()
        self.clb, self.cub = self._initialize_constraint_bounds()

        # Initialize sparsity pattern
        self.row_ids, self.col_ids = self._initialize_sparsity_pattern()

    def _initialize_nodes(self, contact_phases: np.ndarray) -> List[Node]:
        """Create node objects for each contact phase."""
        nodes = []
        var_offset = 0
        con_offset = 1  # Starting after the time constraint

        for k in range(self.K):
            contacts = self._get_contacts_for_phase(contact_phases[k])
            node = Node(
                k,
                v_id=var_offset,
                c_id=con_offset,
                nv=self.model.nv,
                dt=self.dt,
                contact_phase=contacts,
            )
            nodes.append(node)
            var_offset += node.x_dim
            con_offset += node.c_dim

        return nodes

    def _get_contacts_for_phase(self, phase: int) -> List[str]:
        """Map contact phase number to contact state."""
        if phase == 0:
            return ["left", "right"]
        if phase == 1:
            return ["left"]
        if phase == 2:
            return ["right"]
        return []

    def _get_problem_dimensions(self) -> Tuple[int, int]:
        """Calculate total variables and constraints dimensions."""
        vars_dim = sum(node.x_dim for node in self.nodes)
        cons_dim = 1 + sum(node.c_dim for node in self.nodes)  # +1 for time constraint
        return vars_dim, cons_dim

    def _initialize_variable_bounds(self) -> Tuple[List, List]:
        """Initialize variable bounds."""
        lb = [None] * self.vars_dim
        ub = [None] * self.vars_dim

        for node in self.nodes:
            self.x0[node.dt_id] = self.dt  # Initialize time step
            nd_vars_bounds(node, lb, ub, self.model)

        return lb, ub

    def _initialize_constraint_bounds(self) -> Tuple[List, List]:
        """Initialize constraint bounds."""
        clb = [0] * self.cons_dim
        cub = [0] * self.cons_dim
        clb[0] = 2.0
        cub[0] = 2.0

        for node in self.nodes:
            nd_cons_bounds(node, clb, cub, self.model)

        return clb, cub

    def _initialize_sparsity_pattern(self) -> Tuple[List, List]:
        """Initialize Jacobian sparsity pattern."""
        row_ids, col_ids = [], []
        c_tf_id = slice(0, 1)

        for k in range(self.K - 1):
            extend_ids_lists(row_ids, col_ids, c_tf_id, self.nodes[k].dt_id)
            const_grad_ids(self.nodes[k], self.nodes[k + 1], row_ids, col_ids)

        extend_ids_lists(row_ids, col_ids, c_tf_id, self.nodes[-1].dt_id)
        const_grad_ids(self.nodes[-1], None, row_ids, col_ids)

        return row_ids, col_ids

    def set_initial_pose(self, q: np.ndarray):
        """Set initial configuration and initialize forces."""
        q_tan = q_pin2tan(q)
        self.q_ref = q_tan

        # Set bounds for first node
        first_node = self.nodes[0]
        self.lb[first_node.q_id] = self.ub[first_node.q_id] = q_tan
        self.lb[first_node.vq_id] = self.ub[first_node.vq_id] = np.zeros(self.model.nv)

        # Initialize forces based on gravity compensation
        g = self.model.gravity.linear[2]
        mass = pin.computeTotalMass(self.model, self.data)

        for node in self.nodes:
            self.x0[node.q_id] = q_tan
            n_contacts = len(node.contact_frames)
            for frame in node.contact_frames:
                self.x0[node.forces_ids[frame].start + 2] = -mass * g / n_contacts

    def set_target_pose(self, q: np.ndarray):
        """Set target configuration and initialize trajectory guess."""
        q_tan = q_pin2tan(q)

        # Set bounds for last node
        last_node = self.nodes[-1]
        self.lb[last_node.q_id] = self.ub[last_node.q_id] = q_tan
        self.lb[last_node.vq_id] = self.ub[last_node.vq_id] = np.zeros(self.model.nv)

        # Initialize trajectory guess with linear interpolation
        q0 = q_tan2pin(self.x0[self.nodes[0].q_id])
        for node in self.nodes:
            self.x0[node.q_id] = q_pin2tan(
                pin.interpolate(self.model, q0, q, node.k / self.K)
            )

    def objective(self, w: np.ndarray) -> float:
        """Compute objective function value."""
        obj = 0.0

        # Configuration tracking cost
        if self.q_ref is not None:
            obj += 0.5 * sum(
                (w[n.q_id.start + 6 : n.q_id.stop] - self.q_ref[6:]).T
                @ self.R_q
                @ (w[n.q_id.start + 6 : n.q_id.stop] - self.q_ref[6:])
                for n in self.nodes[:-1]
            )

        # Velocity and acceleration costs
        obj += 0.5 * sum(w[n.vq_id].T @ self.R_vq @ w[n.vq_id] for n in self.nodes[:-1])
        obj += 0.5 * sum(w[n.aq_id].T @ self.R_aq @ w[n.aq_id] for n in self.nodes[:-1])

        # Contact force costs
        for node in self.nodes:
            for frame in node.contact_frames:
                obj += (
                    0.5
                    * w[node.forces_ids[frame]]
                    @ self.R_forces
                    @ w[node.forces_ids[frame]]
                )

        return obj

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Compute objective function gradient."""
        grad = np.zeros_like(w)

        for node in self.nodes:
            # Configuration gradient
            if self.q_ref is not None:
                grad[node.q_id.start + 6 : node.q_id.stop] = (
                    w[node.q_id.start + 6 : node.q_id.stop] - self.q_ref[6:]
                ).T @ self.R_q

            # Velocity and acceleration gradients
            grad[node.vq_id] = w[node.vq_id].T @ self.R_vq
            grad[node.aq_id] = w[node.aq_id].T @ self.R_aq

            # Contact force gradients
            for frame in node.contact_frames:
                grad[node.forces_ids[frame]] = self.R_forces @ w[node.forces_ids[frame]]

        return grad

    def constraints(self, w: np.ndarray) -> np.ndarray:
        """Compute constraint values."""
        c = np.zeros(self.cons_dim)

        # Time constraint (sum of time steps)
        c[0] = sum(w[node.dt_id] for node in self.nodes)

        # Dynamics constraints
        for k in range(self.K - 1):
            const(self.nodes[k], self.nodes[k + 1], w, c, self.model, self.data)
        const(self.nodes[-1], None, w, c, self.model, self.data)

        return c

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get Jacobian sparsity pattern."""
        return np.array(self.row_ids), np.array(self.col_ids)

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        """Compute constraint Jacobian."""
        jac = np.zeros((self.cons_dim, self.vars_dim))

        # Time constraint derivatives
        for node in self.nodes:
            jac[0, node.dt_id] = 1

        # Dynamics constraint derivatives
        for k in range(self.K - 1):
            const_grad(self.nodes[k], self.nodes[k + 1], w, jac, self.model, self.data)
        const_grad(self.nodes[-1], None, w, jac, self.model, self.data)

        # Return only non-zero elements
        rows, cols = self.jacobianstructure()
        return jac[rows, cols]

    def intermediate(
        self,
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials,
    ):
        self.iter_count = iter_count

    def decode_solution(self) -> Tuple[List, List, List[Dict]]:
        """Extract solution into interpretable formats.

        Returns:
            Tuple of (time steps, configurations, contact forces)
        """
        dts = []
        qs = []
        forces = []

        for node in self.nodes:
            dts.append(self.sol[node.dt_id.start])
            qs.append(q_tan2pin(self.sol[node.q_id]))

            frame_forces = {}
            for frame in node.contact_frames:
                frame_id = self.model.getFrameId(frame)
                frame_forces[frame_id] = self.sol[node.forces_ids[frame]]

            forces.append(frame_forces)

        return dts, qs, forces

    def save_solution_report(
        self,
        n_iterations: int = None,
        convergence_time: float = None,
        status: str = "",
        message: str = "",
        save_dir: str = "data/trajopt",
    ) -> dict:
        """
        Save complete solution report with all requested metrics.

        Args:
            status: Optimization status ("success", "failed", "error")
            n_iterations: Number of iterations completed
            convergence_time: Time until convergence in seconds
            message: Optional status message
            save_dir: Directory to save the report

        Returns:
            Dictionary containing all saved metrics
        """
        if not hasattr(self, "sol"):
            raise ValueError("No solution available - run optimization first")

        # Prepare contact gait description
        contact_gait = []
        for node in self.nodes:
            contact_gait.append(node.contact_phase)

        # Prepare forces data
        forces_data = {}
        for node in self.nodes:
            for frame in node.contact_frames:
                # frame_id = self.model.getFrameId(frame)
                if frame not in forces_data:
                    forces_data[frame] = []
                forces_data[frame].append(
                    {"node": node.k, "force": self.sol[node.forces_ids[frame]].tolist()}
                )

        # Compile complete report
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": (
                    self.model.name
                    if hasattr(self.model, "name")
                    else str(type(self.model))
                ),
                "status": status,
                "message": message,
                "convergence_time_sec": convergence_time,
                "iterations": n_iterations,
            },
            "parameters": {
                "num_nodes": self.K,
                "contact_gait": contact_gait,
                "initial_dt": float(self.dt),
            },
            "solution": {
                "dts": [float(self.sol[n.dt_id]) for n in self.nodes],
                "qs": [self.sol[n.q_id].tolist() for n in self.nodes],
                "vqs": [self.sol[n.vq_id].tolist() for n in self.nodes],
                "aqs": [self.sol[n.aq_id].tolist() for n in self.nodes],
                "forces": forces_data,
            },
        }

        # Save to file
        timestamp = datetime.now().strftime("%d%m_%H%M%S")
        filename = f"trajopt_solution_{timestamp}.json"
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
