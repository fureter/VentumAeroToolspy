import copy
import abc

import numpy as np
from matplotlib import pyplot as plt


class ParameterEstimator(abc.ABC):

    def __init__(self, p_0, sensor_variance, estimate_variance):
        self.p_0 = p_0
        self.p = copy.deepcopy(self.p_0)
        self.R = sensor_variance
        self.invR = np.linalg.inv(sensor_variance)
        self.Q = estimate_variance

    def estimate(self, t, x0, y_true, u, max_iter=100, dp=1E-5, unconverge_tol=1E4, converge_tol=1E-4):
        iteration = 0
        while iteration < max_iter:
            print('iteration: {iteration}'.format(iteration=iteration+1))
            print('Integrating State Space')
            x_est = self.state_space_model(t, x0, u, self.p)
            print('Calculating Observables')
            y_est = self.observation_model(t, x_est)
            print('Estimating Grad of y_est')
            grad_y_est = self.calc_estimate_gradient(t, x_est, u, dp)

            grad_J_p = np.zeros(self.p.shape[0])
            H_J_p = np.zeros([self.p.shape[0], self.p.shape[0]])
            print('Estimating Grad J_p and H_J_p')
            for i in range(y_true.shape[1]):
                grad_J_p += -grad_y_est[i, :, :].T @ self.invR @ (y_true[:, i] - y_est[:, i])
                H_J_p += grad_y_est[i, :, :].T @ self.invR @ grad_y_est[i, :, :]
            print('Inverting H_J_p')
            inv_H_J_p = np.linalg.inv(H_J_p)
            print('Updating Estimate of P')
            p_change = inv_H_J_p @ grad_J_p
            self.p = self.p - p_change
            print('Change in P: %s' % p_change)
            if np.linalg.norm(p_change,ord=2) > unconverge_tol:
                print('Failed to converge')
                break
            elif np.linalg.norm(p_change,ord=2) < converge_tol:
                print('Converged in %s Iterations' % iteration)
                break
            iteration += 1

    def calc_estimate_gradient(self, t, x_est, u, dp):
        grad_y = np.zeros([len(t), len(x_est), len(self.p)])
        wiggle = np.zeros(self.p.shape[0])
        for i in range(len(self.p)):
            wiggle[i] = 1
            x_est_plus = self.state_space_model(t, x_est[:, 0], u, self.p + dp * wiggle)
            y_est_plus = self.observation_model(t, x_est_plus)
            y_est = self.observation_model(t, x_est)
            grad_y[:, :, i] = (y_est_plus - y_est).T/dp
            wiggle[i] = 0


        return grad_y

    def cost_function(self, t, x0, u, y_true):
        x_est = self.state_space_model(t, x0, u, self.p)
        y_est = self.observation_model(t, x_est)
        cost = 0
        for i in range(y_true.shape[1]):
                cost += 0.5 * (y_true[:, i] - y_est[:, i]).T @ self.invR @ (y_true[:, i] - y_est[:, i])
        return cost

    @abc.abstractmethod
    def state_space_model(self, time, x0, u, p):
        raise NotImplementedError

    @abc.abstractmethod
    def observation_model(self, t, x):
        raise NotImplementedError

    def plot_cost_function_isocontour(self, t, x0, u, y_true, ind_param1, ind_param2, param1_test_points,
                                      param2_test_points, xlabel, ylabel):
        p1, p2 = np.meshgrid(param1_test_points, param2_test_points)
        cost_surface = np.zeros(p1.shape)
        for i in range(p1.shape[0]):
            for j in range(p1.shape[1]):
                p_orig = copy.deepcopy(self.p)
                self.p[ind_param1] = p1[i, j]
                self.p[ind_param2] = p2[i, j]
                cost_surface[i, j] = self.cost_function(t, x0, u, y_true)
                self.p = p_orig
        plt.figure(figsize=(7,6), dpi=128)
        plt.contourf(p1, p2, cost_surface)
        cbar = plt.colorbar()
        cbar.set_label('Cost')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Cost Contour')
        plt.grid()
        plt.show()

