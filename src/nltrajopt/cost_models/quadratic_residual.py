import numpy as np
import pinocchio as pin
from .abstract_cost import AbstractCostFunction


class ConfigurationCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.q_id.start + 6 : node.q_id.stop].reshape(-1, 1)
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.q_id.start + 6 : node.q_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.q_id.start + 6 : node.q_id.stop] += jac


class JointVelocityCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.vq_id.start + 6 : node.vq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.vq_id.start + 6 : node.vq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.vq_id.start + 6 : node.vq_id.stop] += jac


class JointAccelerationCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        var = opt_vect[node.aq_id.start + 6 : node.aq_id.stop].reshape(-1, 1)
        res = self.compute_residual(var)
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        var = opt_vect[node.aq_id.start + 6 : node.aq_id.stop].reshape((-1, 1))
        res = self.compute_residual(var)
        jac = self.compute_gradient(res)
        cost_grad[node.aq_id.start + 6 : node.aq_id.stop] += jac


class BaseOrientationCost(AbstractCostFunction):
    def __init__(self, ref: np.array, weight: np.array):
        super().__init__(ref, weight)
        self.z_axis = np.array([[0, 0, 1]]).T

    def obj(self, opt_vect, node, next_node):
        theta = opt_vect[node.q_id.start + 3 : node.q_id.start + 6]
        res = pin.exp3(theta) @ self.z_axis - self.ref
        cost = self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        theta = opt_vect[node.q_id.start + 3 : node.q_id.start + 6]
        R = pin.exp3(theta)
        res = R @ self.z_axis - self.ref

        J_r = pin.Jexp3(theta)
        J_res = -R @ pin.skew(self.z_axis) @ J_r
        gr = self.compute_gradient(res) @ J_res
        cost_grad[node.q_id.start + 3 : node.q_id.start + 6] += gr.reshape((-1,))


class ForceCost(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        cost = 0
        for frame in node.contact_phase_fnames:
            f = opt_vect[node.forces_ids[frame]]
            cost += self.compute_cost(f)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        for frame in node.contact_phase_fnames:
            f = opt_vect[node.forces_ids[frame]]
            jac = self.compute_gradient(f)
            cost_grad[node.forces_ids[frame]] = jac


class DeltaForce(AbstractCostFunction):
    def obj(self, opt_vect, node, next_node):
        if next_node is None:
            return 0
        cost = 0
        for frame in node.contact_phase_fnames:
            self.ref = opt_vect[node.forces_ids[frame]]
            if frame in next_node.contact_phase_fnames:
                fk1 = opt_vect[next_node.forces_ids[frame]]
            else:
                fk1 = np.zeros((3,))
            res = self.compute_residual(fk1)
            cost += self.compute_cost(res)
        return cost

    def grad(self, opt_vect, cost_grad, node, next_node):
        if next_node is None:
            return
        for frame in node.contact_phase_fnames:
            self.ref = opt_vect[node.forces_ids[frame]]
            if frame in next_node.contact_phase_fnames:
                fk1 = opt_vect[next_node.forces_ids[frame]]
            else:
                fk1 = np.zeros((3,))
            res = self.compute_residual(fk1)
            jac = self.compute_gradient(res)
            cost_grad[node.forces_ids[frame]] -= jac
            if frame in next_node.contact_phase_fnames:
                cost_grad[next_node.forces_ids[frame]] += jac
