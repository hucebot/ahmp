import pinocchio as pin
import cyipopt
import numpy as np
import time
import os
from datetime import datetime
import json
from typing import List, Dict, Optional, Tuple, Callable

from nltrajopt.se3tangent import *
from nltrajopt.constraint_models import *
from nltrajopt.node import *


class NLTrajOpt:
    def __init__(
        self,
        model: pin.Model,
        nodes: List[Node],
        dt: float,
    ):
        """
        Initialize trajectory optimization problem.

        Args:
            model: Pinocchio robot model
            dt: Initial time step guess
            contact_phases: List of contact states for each phase (e.g., [["left"], ["right"]])
            dynamics_type: "centroidal" or "whole_body"
            friction_coeff: Friction coefficient μ
            ground_height: Contact surface height
        """
        self.model = model
        self.data = model.createData()
        self.dt = dt
        self.K = len(nodes)

        self.nodes = nodes

        # Reference and target configurations
        self.base_zori_ref = None
        self.q_ref = None

        # Initialize nodes
        self._initialize_nodes()

        # Problem dimensions
        self.vars_dim = sum(node.x_dim for node in self.nodes)
        self.cons_dim = 1 + sum(node.c_dim for node in self.nodes)  # +1 for time constraint

        # Initialize optimization structures
        self.x0 = np.zeros(self.vars_dim)
        self.lb = [None] * self.vars_dim
        self.ub = [None] * self.vars_dim
        self.clb = [0] * self.cons_dim
        self.cub = [0] * self.cons_dim

        # Set default bounds
        self._initialize_bounds()

        # Initialize sparsity pattern
        self.row_ids, self.col_ids = self._initialize_sparsity_pattern()

        # Initialize this
        self.iter_count = 0

    def _initialize_nodes(self):
        k = 0
        var_offset = 0
        con_offset = 1  # total time constraint

        for node in self.nodes:
            node.init_node_ids(var_offset, con_offset, k)
            k += 1
            var_offset += node.x_dim
            con_offset += node.c_dim

    def _initialize_sparsity_pattern(self) -> Tuple[List[int], List[int]]:
        """Build Jacobian sparsity pattern"""
        row_ids, col_ids = [], []

        # Time constraint (depends on all dt variables)
        for node in self.nodes:
            row_ids.append(0)
            col_ids.append(node.dt_id.start)

        # Other constraints
        for k in range(self.K - 1):
            for constraint in self.nodes[k].constraints_list:
                constraint.get_structure_ids(self.nodes[k], self.nodes[k + 1], row_ids, col_ids)

        # Final node
        for constraint in self.nodes[-1].constraints_list:
            constraint.get_structure_ids(self.nodes[-1], None, row_ids, col_ids)

        return row_ids, col_ids

    def _initialize_bounds(self):
        """Set default variable and constraint bounds"""
        for node in self.nodes:
            for constraint in node.constraints_list:
                constraint.get_bounds(
                    node=node,
                    model=self.model,
                    lb=self.lb,
                    ub=self.ub,
                    clb=self.clb,
                    cub=self.cub,
                )

    def set_initial_pose(self, q: np.ndarray, v: Optional[np.ndarray] = None):
        """
        Set initial configuration and initialize forces.

        Args:
            q: Initial configuration in pinocchio format (quaternion for floating base)
            v: Optional initial velocity (zero if None)
        """
        q_tan = q_pin2tan(q)
        v = np.zeros(self.model.nv) if v is None else v

        self.q_ref = q_tan

        # Set first node state
        first_node = self.nodes[0]
        self.x0[first_node.q_id] = q_tan
        self.x0[first_node.vq_id] = v

        # Fix initial state bounds
        self.lb[first_node.q_id] = self.ub[first_node.q_id] = q_tan
        self.lb[first_node.vq_id] = self.ub[first_node.vq_id] = v
        # self.lb[first_node.aq_id] = self.ub[first_node.aq_id] = np.zeros(self.model.nv)

        # Initialize forces based on gravity compensation
        g = np.linalg.norm(self.model.gravity.linear)
        mass = pin.computeTotalMass(self.model, self.data)

        for node in self.nodes:
            self.x0[node.q_id] = q_tan
            n_contacts = len(node.contact_phase_fnames)
            for frame in node.contact_phase_fnames:
                self.x0[node.forces_ids[frame].start + 2] = mass * g / n_contacts
                # self.x0[node.forces_ids[frame].start + 2] = 1

    def set_target_pose(self, q: np.ndarray, v: Optional[np.ndarray] = None):
        """
        Set target configuration and initialize trajectory guess.

        Args:
            q: Target configuration in pinocchio format
            v: Optional target velocity (zero if None)
        """
        q_tan = q_pin2tan(q)
        v = np.zeros(self.model.nv) if v is None else v

        # Set last node state
        last_node = self.nodes[-1]
        self.x0[last_node.q_id] = q_tan
        self.x0[last_node.vq_id] = v

        # Fix target state bounds
        self.lb[last_node.q_id] = self.ub[last_node.q_id] = q_tan
        self.lb[last_node.vq_id] = self.ub[last_node.vq_id] = v
        self.lb[last_node.aq_id] = self.ub[last_node.aq_id] = np.zeros(self.model.nv)

        # Initialize trajectory guess with linear interpolation
        # q0 = q_tan2pin(self.x0[self.nodes[0].q_id])
        # for node in self.nodes:
        #     self.x0[node.q_id] = q_pin2tan(pin.interpolate(self.model, q0, q, node.k / self.K))

    def set_base_target(self, q_):
        """
        Only takes into account base position and orientation,
        does not constrain joint poisitions
        """
        q_tan = q_pin2tan(q_)

        self.q_final = q_tan
        j_pos_lower = self.model.lowerPositionLimit[7:]
        j_pos_upper = self.model.upperPositionLimit[7:]

        self.lb[self.nodes[-1].q_id] = np.hstack((q_tan[0:6], j_pos_lower))
        self.ub[self.nodes[-1].q_id] = np.hstack((q_tan[0:6], j_pos_upper))
        self.lb[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
        self.ub[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))

        q0 = q_tan2pin(self.x0[self.nodes[0].q_id])

        for node in self.nodes:
            self.x0[node.q_id] = q_pin2tan(
                pin.interpolate(self.model, q0, q_, node.k / self.K)
            )

    # def set_base_xy_target(self, q_):
    #     """
    #     Only takes into account base xy position
    #     """
    #     q_tan = q_pin2tan(q_)
    #
    #     self.q_final = q_tan
    #     j_pos_lower = self.model.lowerPositionLimit[7:]
    #     j_pos_upper = self.model.upperPositionLimit[7:]
    #
    #     self.lb[self.nodes[-1].q_id] = np.hstack((q_tan[0:2], np.full(1, -np.inf), q_tan[3:6], j_pos_lower))
    #     self.ub[self.nodes[-1].q_id] = np.hstack((q_tan[0:2], np.full(1, np.inf), q_tan[3:6], j_pos_upper))
    #     self.lb[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
    #     self.ub[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
    #
    #     q0 = q_tan2pin(self.x0[self.nodes[0].q_id])
    #
    #     for node in self.nodes:
    #         self.x0[node.q_id] = q_pin2tan(
    #             pin.interpolate(self.model, q0, q_, node.k / self.K)
    #         )

    def set_base_xyz_target(self, q_):
        """
        Only takes into account base xy position
        """
        q_tan = q_pin2tan(q_)

        self.q_final = q_tan
        j_pos_lower = self.model.lowerPositionLimit[7:]
        j_pos_upper = self.model.upperPositionLimit[7:]

        self.lb[self.nodes[-1].q_id] = np.hstack((q_tan[0:3], np.full(3, -np.inf), j_pos_lower))
        self.ub[self.nodes[-1].q_id] = np.hstack((q_tan[0:3], np.full(3, np.inf), j_pos_upper))
        self.lb[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))
        self.ub[self.nodes[-1].vq_id] = np.zeros((self.nodes[0].nv,))

        q0 = q_tan2pin(self.x0[self.nodes[0].q_id])

        for node in self.nodes:
            self.x0[node.q_id] = q_pin2tan(
                pin.interpolate(self.model, q0, q_, node.k / self.K)
            )


    def objective(self, w: np.ndarray) -> float:
        """Compute objective function value."""
        obj = 0.0

        for k in range(self.K):
            for cost in self.nodes[k].costs_list:
                if k == self.K - 1:
                    obj += cost.obj(w, self.nodes[k], None)
                else:
                    obj += cost.obj(w, self.nodes[k], self.nodes[k + 1])

        return obj

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Compute objective function gradient."""
        grad = np.zeros_like(w)
        z_ax = np.array([[0, 0, 1]]).T

        for k in range(self.K):
            for cost in self.nodes[k].costs_list:
                if k == self.K - 1:
                    cost.grad(w, grad, self.nodes[k], None)
                else:
                    cost.grad(w, grad, self.nodes[k], self.nodes[k + 1])

        return grad

    def constraints(self, w: np.ndarray) -> np.ndarray:
        """Compute all constraint values"""

        t0 = time.time()
        c = np.zeros(self.cons_dim)

        # Apply all constraints
        for k in range(self.K):
            next_node = None
            if k != self.K - 1:
                next_node = self.nodes[k + 1]

            for constraint in self.nodes[k].constraints_list:
                constraint.compute_constraints(self.nodes[k], next_node, w, c, self.model, self.data)

        # print(time.time() - t0)

        return c

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get Jacobian sparsity pattern."""
        return np.array(self.row_ids), np.array(self.col_ids)

    def jac_test(self, w):
        jac = np.zeros((self.cons_dim, self.vars_dim))

        # Apply all constraint Jacobians
        for k in range(self.K):
            next_node = None
            if k != self.K - 1:
                next_node = self.nodes[k + 1]

            for constraint in self.nodes[k].constraints_list:
                constraint.compute_jacobians(self.nodes[k], next_node, w, jac, self.model, self.data)

        return jac

    def jacobian(self, w: np.ndarray) -> np.ndarray:
        """Compute constraint Jacobian"""
        jac = np.zeros((self.cons_dim, self.vars_dim))

        t0 = time.time()

        # Apply all constraint Jacobians
        for k in range(self.K):
            next_node = None
            if k != self.K - 1:
                next_node = self.nodes[k + 1]

            for constraint in self.nodes[k].constraints_list:
                constraint.compute_jacobians(self.nodes[k], next_node, w, jac, self.model, self.data)

        # Return only non-zero elements
        rows, cols = self.jacobianstructure()

        return jac[rows, cols]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.iter_count = iter_count

    def solve(self, max_iter: int = 1000, tol: float = 1e-4, parallel=True, print_level=3) -> Dict:
        """Solve the optimization problem"""

        # Initialize solver (using IPOPT through cyipopt)
        nlp = cyipopt.Problem(
            n=self.vars_dim,
            m=self.cons_dim,
            problem_obj=self,
            lb=self.lb,
            ub=self.ub,
            cl=self.clb,
            cu=self.cub,
        )

        nlp.add_option("max_iter", max_iter)
        nlp.add_option("print_level", print_level)
        if parallel:
            nlp.add_option("linear_solver", "ma97")
        nlp.add_option("tol", tol)

        # Solve
        t0 = time.time()
        self.sol, self.info = nlp.solve(self.x0)
        solve_time = time.time() - t0

        # Return solution
        return {
            "model": self.model.name,
            "solve_time": solve_time,
            "iter_count": self.iter_count,
            "info": self.info,
            "solution": self.sol,
            "nodes": self._decode_solution(self.sol),
        }

    def _decode_solution(self, sol: np.ndarray) -> List[Dict]:
        """Convert solution vector to interpretable format"""
        results = []
        for node in self.nodes:
            result = {
                "dt": float(sol[node.dt_id]),
                "q": q_tan2pin(sol[node.q_id]),
                "v": sol[node.vq_id],
                "a": sol[node.aq_id],
                "forces": {},
                "contact_positions": {},
            }

            for frame in node.contact_phase_fnames:
                result["forces"][frame] = sol[node.forces_ids[frame]]
                result["contact_positions"][frame] = sol[node.contact_pos_ids[frame]]

            results.append(result)
        return results

    def save_solution(solution_dict: dict, save_name, save_dir: str = "trajopt_solutions"):

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{save_name}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)

        # Prepare data for JSON serialization
        save_data = {
            "info": {
                "model": solution_dict["model"],
                "solve_time": solution_dict["solve_time"],
                "iterations": solution_dict["iter_count"],
                "notes": "",
            },
            "solution": {"nodes": []},
        }

        # Convert numpy arrays to lists for JSON serialization
        for node in solution_dict["nodes"]:
            node_data = {
                "dt": float(node["dt"]),
                "q": node["q"].tolist(),
                "v": node["v"].tolist(),
                "a": node["a"].tolist(),
                "forces": {k: v.tolist() for k, v in node["forces"].items()},
                "contact_positions": {k: v.tolist() for k, v in node["contact_positions"].items()},
            }
            save_data["solution"]["nodes"].append(node_data)

        # Save to file
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"Solution saved to {filepath}")
        return filepath

    def load_solution(filepath: str) -> dict:
        """
        Load a saved trajectory optimization solution from a JSON file.

        Args:
            filepath: Path to the solution JSON file

        Returns:
            Dictionary containing the solution in the same format as NLTrajOpt.solve()
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct the solution dictionary
        solution_dict = {
            "info": {
                "solve_time": data["info"]["solve_time"],
                "iter": data["info"]["iterations"],
            },
            "solution": None,  # Will be filled below
            "nodes": [],
        }

        # Convert lists back to numpy arrays
        for node_data in data["solution"]["nodes"]:
            node = {
                "dt": node_data["dt"],
                "q": np.array(node_data["q"]),
                "v": np.array(node_data["v"]),
                "a": np.array(node_data["a"]),
                "forces": {k: np.array(v) for k, v in node_data["forces"].items()},
            }
            solution_dict["nodes"].append(node)

        solution_dict["solution"] = solution_dict["nodes"]  # For backward compatibility

        return solution_dict
