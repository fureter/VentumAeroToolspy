import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from analysis.parameter_estimator import ParameterEstimator


class SimplifiedLongitudinal(ParameterEstimator):
    """
    Documentation below written in LaTeX:

    Parameters\space from\space Aircraft: m, I_{yy}, T_{max}, S, c, b, K, \rho, \delta_{zT} \newline
    Parameters\space to\space Estimate: cl_0, cl_{alpha}, cl_{\delta e}, cd_0, cd_{alpha}, cm_0, cm_{alpha}, cm_{\delta e}, cm_q \newline

    Equations\space of\space Motion: \newline
        State\space Space: \bar{x} = [U, W, Q, \theta, X_E, Z_E] \newline
        Control\space Vector: \bar{u} = [\delta e, n] \newline \newline

        m(\dot{U} + QW) = T - D\cos{\alpha} + L\cos{\alpha} - mg\sin{\theta} \newline
        m(\dot{Q} + QU) = -L\cos{\alpha} - D\sin{\alpha} + mg\cos{\theta} \newline
        I_{yy}*\dot{Q} = M_{A} - T\delta_{zT} \newline
        \dot{\Theta} = Q \newline
        L = \frac{1}{2}\rho V^2SC_L \newline
        C_L = cl_0 + cl_{\alpha} \alpha + cl_{\delta e}\delta e \newline
        C_D = cd_0 + cd_{\alpha} \alpha + KC_L^2 \newline
        C_M = cm_0 + cm_{\alpha} \alpha + c_{m_{\delta e}} \delta e + c_{m\hat{q}}\hat{q} \newline
        T = nT_{max}, \space where\space n = (0-100) \%
    """
    def __init__(self, p_0, sensor_variance, estimate_variance, aircraft_param):
        super().__init__(p_0, sensor_variance, estimate_variance)
        self.aircraft_param = aircraft_param

    def state_space_model(self, time, x0, u_in, p):
        m = self.aircraft_param['m']
        c = self.aircraft_param['c']
        S = self.aircraft_param['S']
        K = self.aircraft_param['K']
        rho = self.aircraft_param['rho']
        T_max = self.aircraft_param['T_max']
        delta_zT = self.aircraft_param['delta_zT']
        Iyy = self.aircraft_param['Iyy']
        t_span = time[-1] - time[0]
        dt_u = t_span / u_in.shape[0]

        def func(t, y):
            ind = int(t / dt_u)
            if ind == u_in.shape[0]:
                ind -= 1

            u = y[0]
            w = y[1]
            q = y[2]
            pitch = y[3]
            v = np.linalg.norm([u, w], ord=2)
            alpha = np.arctan2(w, u)
            qh = q * c/v

            cl = p[0] + p[1] * alpha + p[2] * u_in[ind,0]
            cd = p[3] + p[4] * alpha + K * cl**2
            cm = p[5] + p[6] * alpha + p[7] * u_in[ind, 0] + p[8] * qh

            q_inf = 0.5 * rho*v**2 * S

            lift = q_inf * cl
            drag = q_inf * cd
            moment = q_inf * cm * c

            c_a = np.cos(alpha)
            s_a = np.sin(alpha)
            c_p = np.cos(pitch)
            s_p = np.sin(pitch)

            return np.array([
                -q * w + (lift * s_a - drag * c_a + T_max*u_in[ind, 1]) / m - 9.80665 * s_p,
                q * u + (-lift * c_a - drag * s_a) / m + 9.80665 * c_p,
                (moment - T_max*u_in[ind, 1] * delta_zT) / Iyy,
                q,
                u * c_p - w * s_p,
                u * s_p + w * c_p
            ])
        sol = solve_ivp(func,[0, t_span], x0, t_eval=time)
        return sol.y

    def observation_model(self, t, x):
        return x

def simplified_longitudinal_example_elevator_pulse():
    p_true = np.array([
        0.895,  # CL_0
        5.01,  # CL_alpha
        0.722,  # CL_de
        0.177,  # CD_0
        0.232,  # CD_alpha
        -0.046,  # CM_0
        -1.087,  # CM_alpha
        -1.88,  # CM_de
        -7.055,  # CM_q
    ])

    p_0 = np.array([
        0.895 + 0.2,  # CL_0
        5.01 - 1.6,  # CL_alpha
        0.722 - 0.2,  # CL_de
        0.177 + 0.05,  # CD_0
        0.232 + 0.01,  # CD_alpha
        -0.046 - 0.01,  # CM_0
        -1.087 - 0.2,  # CM_alpha
        -1.88 + 0.4,  # CM_de
        -7.055 + 0.8,  # CM_q
    ])
    sensor_variance = np.eye(6)
    sensor_variance[0, 0] = 0.4 ** 2
    sensor_variance[1, 1] = 0.4 ** 2
    sensor_variance[2, 2] = 0.05 ** 2
    sensor_variance[3, 3] = np.deg2rad(4.0) ** 2
    sensor_variance[4, 4] = 10.0 ** 2
    sensor_variance[5, 5] = 10.0 ** 2
    estimate_variance = np.eye(p_0.shape[0]) * 0.01  # currently unused
    aircraft_param = {'m': 7484.4, 'Iyy': 84309, 'T_max': 13878, 'S': 32.8, 'c': 2.29, 'K': 1 / (np.pi * 6.25 * 0.9),
                      'rho': 0.9, 'delta_zT': -0.378}

    long_perf_true = SimplifiedLongitudinal(p_true, sensor_variance, estimate_variance, aircraft_param)
    long_perf_est = SimplifiedLongitudinal(p_0, sensor_variance, estimate_variance, aircraft_param)

    time = np.linspace(0, 50, int(100))
    x0 = np.array([
        69.371,
        2.0529,
        0.0,
        -0.0171,
        0.0,
        0.0
    ])

    # steady state control inputs
    u = np.zeros([100, 2])
    u[:, 1] = np.ones(100)
    u[:, 0] = np.ones(100) * np.deg2rad(-1.4)
    u[10:20, 0] = np.ones(10) * np.deg2rad(-5.4)

    plt.figure(figsize=(3,2))
    plt.plot(u[:,0])
    plt.title('Elevator Control Input')
    plt.ylabel('de [rad]')
    plt.grid()
    plt.show()

    y_true = long_perf_true.state_space_model(time, x0, u, p=p_true)
    y_true_noise = copy.deepcopy(y_true)

    y_true_noise[0, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[0, 0]), size=y_true_noise[0, :].shape)
    y_true_noise[1, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[1, 1]), size=y_true_noise[1, :].shape)
    y_true_noise[2, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[2, 2]), size=y_true_noise[2, :].shape)
    y_true_noise[3, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[3, 3]), size=y_true_noise[3, :].shape)
    y_true_noise[4, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[4, 4]), size=y_true_noise[4, :].shape)
    y_true_noise[5, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[5, 5]), size=y_true_noise[5, :].shape)

    y_initial = long_perf_est.state_space_model(time, x0, u, p=p_0)

    print('beginning estimation')
    long_perf_est.estimate(time, x0, y_true_noise, u, max_iter=100, dp=1E-5)
    print('P before: %s' % long_perf_est.p_0)
    print('P after: %s' % long_perf_est.p)
    print('P Delta: %s' % (long_perf_est.p - long_perf_est.p_0))
    print('Initial Error: %s' % (p_true - long_perf_est.p_0))
    print('End Error: %s' % (p_true - long_perf_est.p))
    print('Finished estimation')
    y_est_update = long_perf_est.state_space_model(time, x0, u, p=long_perf_est.p)
    plt.figure(figsize=(16, 9))
    for ind, value_type in zip(range(6), ['U [m/s]', 'W [m/s]', 'Q [rad/s]', 'Theta [rad]', 'X [m]', 'Z [m]']):
        plt.subplot(3, 2, ind + 1)
        plt.plot(time, y_true[ind, :])
        plt.plot(time, y_true_noise[ind, :])
        plt.plot(time, y_initial[ind, :], '--')
        plt.plot(time, y_est_update[ind, :], '-.')
        plt.title(value_type)
        plt.grid()
    plt.figlegend(['True', 'Measured', 'Initial Guess', 'Updated Estimate'])
    plt.suptitle('Elevator Singlet Example')
    plt.show()

def simplified_longitudinal_example_elevator_doublet():
    p_true = np.array([
        0.895,  # CL_0
        5.01,  # CL_alpha
        0.722,  # CL_de
        0.177,  # CD_0
        0.232,  # CD_alpha
        -0.046,  # CM_0
        -1.087,  # CM_alpha
        -1.88,  # CM_de
        -7.055,  # CM_q
    ])

    p_0 = np.array([
        0.895 + 0.2,  # CL_0
        5.01 - 1.6,  # CL_alpha
        0.722 - 0.2,  # CL_de
        0.177 + 0.05,  # CD_0
        0.232 + 0.01,  # CD_alpha
        -0.046 - 0.01,  # CM_0
        -1.087 - 0.2,  # CM_alpha
        -1.88 + 0.4,  # CM_de
        -7.055 + 0.8,  # CM_q
    ])
    sensor_variance = np.eye(6)
    sensor_variance[0, 0] = 0.4 ** 2
    sensor_variance[1, 1] = 0.4 ** 2
    sensor_variance[2, 2] = 0.05 ** 2
    sensor_variance[3, 3] = np.deg2rad(4.0) ** 2
    sensor_variance[4, 4] = 10.0 ** 2
    sensor_variance[5, 5] = 10.0 ** 2
    estimate_variance = np.eye(p_0.shape[0]) * 0.01  # currently unused
    aircraft_param = {'m': 7484.4, 'Iyy': 84309, 'T_max': 13878, 'S': 32.8, 'c': 2.29,
                      'K': 1 / (np.pi * 6.25 * 0.9),
                      'rho': 0.9, 'delta_zT': -0.378}

    long_perf_true = SimplifiedLongitudinal(p_true, sensor_variance, estimate_variance, aircraft_param)
    long_perf_est = SimplifiedLongitudinal(p_0, sensor_variance, estimate_variance, aircraft_param)

    time = np.linspace(0, 50, int(100))
    x0 = np.array([
        69.371,
        2.0529,
        0.0,
        -0.0171,
        0.0,
        0.0
    ])

    # steady state control inputs
    u = np.zeros([100, 2])
    u[:, 1] = np.ones(100)
    u[:, 0] = np.ones(100) * np.deg2rad(-1.4)
    u[10:20, 0] = np.ones(10) * np.deg2rad(-5.4)
    u[20:30, 0] = np.ones(10) * np.deg2rad(5.4)

    plt.figure(figsize=(3,2))
    plt.plot(u[:,0])
    plt.title('Elevator Control Input')
    plt.ylabel('de [rad]')
    plt.grid()
    plt.show()

    y_true = long_perf_true.state_space_model(time, x0, u, p=p_true)
    y_true_noise = copy.deepcopy(y_true)

    y_true_noise[0, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[0, 0]),
                                           size=y_true_noise[0, :].shape)
    y_true_noise[1, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[1, 1]),
                                           size=y_true_noise[1, :].shape)
    y_true_noise[2, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[2, 2]),
                                           size=y_true_noise[2, :].shape)
    y_true_noise[3, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[3, 3]),
                                           size=y_true_noise[3, :].shape)
    y_true_noise[4, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[4, 4]),
                                           size=y_true_noise[4, :].shape)
    y_true_noise[5, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[5, 5]),
                                           size=y_true_noise[5, :].shape)

    y_initial = long_perf_est.state_space_model(time, x0, u, p=p_0)

    print('beginning estimation')
    long_perf_est.estimate(time, x0, y_true_noise, u, max_iter=100, dp=1E-5)
    print('P before: %s' % long_perf_est.p_0)
    print('P after: %s' % long_perf_est.p)
    print('P Delta: %s' % (long_perf_est.p - long_perf_est.p_0))
    print('Initial Error: %s' % (p_true - long_perf_est.p_0))
    print('End Error: %s' % (p_true - long_perf_est.p))
    print('Finished estimation')
    y_est_update = long_perf_est.state_space_model(time, x0, u, p=long_perf_est.p)
    plt.figure(figsize=(16, 9))
    for ind, value_type in zip(range(6), ['U [m/s]', 'W [m/s]', 'Q [rad/s]', 'Theta [rad]', 'X [m]', 'Z [m]']):
        plt.subplot(3, 2, ind + 1)
        plt.plot(time, y_true[ind, :])
        plt.plot(time, y_true_noise[ind, :])
        plt.plot(time, y_initial[ind, :], '--')
        plt.plot(time, y_est_update[ind, :], '-.')
        plt.title(value_type)
        plt.grid()
    plt.figlegend(['True', 'Measured', 'Initial Guess', 'Updated Estimate'])
    plt.suptitle('Elevator Doublet Example')
    # plt.show()

    sweep_range = 0.25
    sweep_points = 81
    long_perf_est.plot_cost_function_isocontour(time, x0, u, y_true, ind_param1=1, ind_param2=4,
                                                param1_test_points=np.linspace(5.01 - sweep_range, 5.01 + sweep_range, sweep_points,
                                                                               endpoint=True),
                                                param2_test_points=np.linspace(0.232 - sweep_range, 0.232 + sweep_range, sweep_points,
                                                                               endpoint=True),
                                                xlabel=r'CL_alpha', ylabel=r'CD_alpha', )


def simplified_longitudinal_example_elevator_multimaneuver():
    p_true = np.array([
        0.895,  # CL_0
        5.01,  # CL_alpha
        0.722,  # CL_de
        0.177,  # CD_0
        0.232,  # CD_alpha
        -0.046,  # CM_0
        -1.087,  # CM_alpha
        -1.88,  # CM_de
        -7.055,  # CM_q
    ])

    p_0 = np.array([
        0.895 + 0.2,  # CL_0
        5.01 - 1.6,  # CL_alpha
        0.722 - 0.2,  # CL_de
        0.177 + 0.05,  # CD_0
        0.232 + 0.01,  # CD_alpha
        -0.046 - 0.01,  # CM_0
        -1.087 - 0.2,  # CM_alpha
        -1.88 + 0.4,  # CM_de
        -7.055 + 0.8,  # CM_q
    ])
    sensor_variance = np.eye(6)
    sensor_variance[0, 0] = 0.4 ** 2
    sensor_variance[1, 1] = 0.4 ** 2
    sensor_variance[2, 2] = 0.05 ** 2
    sensor_variance[3, 3] = np.deg2rad(4.0) ** 2
    sensor_variance[4, 4] = 10.0 ** 2
    sensor_variance[5, 5] = 10.0 ** 2
    estimate_variance = np.eye(p_0.shape[0]) * 0.01  # currently unused
    aircraft_param = {'m': 7484.4, 'Iyy': 84309, 'T_max': 13878, 'S': 32.8, 'c': 2.29,
                      'K': 1 / (np.pi * 6.25 * 0.9),
                      'rho': 0.9, 'delta_zT': -0.378}

    long_perf_true = SimplifiedLongitudinal(p_true, sensor_variance, estimate_variance, aircraft_param)
    long_perf_est = SimplifiedLongitudinal(p_0, sensor_variance, estimate_variance, aircraft_param)
    num_samples = 2000
    time = np.linspace(0, 200, int(num_samples))
    x0 = np.array([
        69.371,
        2.0529,
        0.0,
        -0.0171,
        0.0,
        0.0
    ])

    # steady state control inputs
    u = np.zeros([num_samples, 2])
    u[:, 1] = np.ones(num_samples)
    u[:, 0] = np.ones(num_samples) * np.deg2rad(-1.4)
    u[10:40, 0] = np.ones(30) * np.deg2rad(-10.4)
    u[40:60, 0] = np.ones(20) * np.deg2rad(10.4)
    u[60:70, 0] = np.ones(10) * np.deg2rad(-10.4)
    u[70:80, 0] = np.ones(10) * np.deg2rad(10.4)

    u[250:275, 0] = np.ones(25) * np.deg2rad(-30)
    u[300:325, 0] = np.ones(25) * np.deg2rad(30)

    plt.figure(figsize=(3,2))
    plt.plot(time, u[:,0])
    plt.title('Elevator Control Input')
    plt.ylabel('de [rad]')
    plt.grid()
    plt.show()

    y_true = long_perf_true.state_space_model(time, x0, u, p=p_true)
    y_true_noise = copy.deepcopy(y_true)

    y_true_noise[0, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[0, 0]),
                                           size=y_true_noise[0, :].shape)
    y_true_noise[1, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[1, 1]),
                                           size=y_true_noise[1, :].shape)
    y_true_noise[2, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[2, 2]),
                                           size=y_true_noise[2, :].shape)
    y_true_noise[3, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[3, 3]),
                                           size=y_true_noise[3, :].shape)
    y_true_noise[4, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[4, 4]),
                                           size=y_true_noise[4, :].shape)
    y_true_noise[5, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[5, 5]),
                                           size=y_true_noise[5, :].shape)

    y_initial = long_perf_est.state_space_model(time, x0, u, p=p_0)

    print('beginning estimation')
    long_perf_est.estimate(time, x0, y_true_noise, u, max_iter=500, dp=1E-5)
    print('P before: %s' % long_perf_est.p_0)
    print('P after: %s' % long_perf_est.p)
    print('P Delta: %s' % (long_perf_est.p - long_perf_est.p_0))
    print('Initial Error: %s' % (p_true - long_perf_est.p_0))
    print('End Error: %s' % (p_true - long_perf_est.p))
    print('Finished estimation')
    y_est_update = long_perf_est.state_space_model(time, x0, u, p=long_perf_est.p)
    plt.figure(figsize=(16, 9))
    for ind, value_type in zip(range(6), ['U [m/s]', 'W [m/s]', 'Q [rad/s]', 'Theta [rad]', 'X [m]', 'Z [m]']):
        plt.subplot(3, 2, ind + 1)
        plt.plot(time, y_true[ind, :])
        plt.plot(time, y_true_noise[ind, :])
        plt.plot(time, y_initial[ind, :], '--')
        plt.plot(time, y_est_update[ind, :], '-.')
        plt.title(value_type)
        plt.grid()
    plt.figlegend(['True', 'Measured', 'Initial Guess', 'Updated Estimate'])
    plt.suptitle('Elevator Doublet Example')
    plt.show()

    sweep_range = 0.25
    sweep_points = 81
    long_perf_est.plot_cost_function_isocontour(time, x0, u, y_true, ind_param1=1, ind_param2=4,
                                                param1_test_points=np.linspace(5.01 - sweep_range, 5.01 + sweep_range, sweep_points,
                                                                               endpoint=True),
                                                param2_test_points=np.linspace(0.232 - sweep_range, 0.232 + sweep_range, sweep_points,
                                                                               endpoint=True),
                                                xlabel=r'CL_alpha', ylabel=r'CD_alpha', )

def simplified_longitudinal_example_unobservable():
    p_true = np.array([
        0.895,  # CL_0
        5.01,  # CL_alpha
        0.722,  # CL_de
        0.177,  # CD_0
        0.232,  # CD_alpha
        -0.046,  # CM_0
        -1.087,  # CM_alpha
        -1.88,  # CM_de
        -7.055,  # CM_q
    ])

    p_0 = np.array([
        0.895 + 0.2,  # CL_0
        5.01 - 1.6,  # CL_alpha
        0.722 - 0.2,  # CL_de
        0.177 + 0.05,  # CD_0
        0.232 + 0.01,  # CD_alpha
        -0.046 - 0.01,  # CM_0
        -1.087 - 0.2,  # CM_alpha
        -1.88 + 0.4,  # CM_de
        -7.055 + 0.8,  # CM_q
    ])
    sensor_variance = np.eye(6)
    sensor_variance[0, 0] = 0.4 ** 2
    sensor_variance[1, 1] = 0.4 ** 2
    sensor_variance[2, 2] = 0.05 ** 2
    sensor_variance[3, 3] = np.deg2rad(4.0) ** 2
    sensor_variance[4, 4] = 10.0 ** 2
    sensor_variance[5, 5] = 10.0 ** 2
    estimate_variance = np.eye(p_0.shape[0]) * 0.01  # currently unused
    aircraft_param = {'m': 7484.4, 'Iyy': 84309, 'T_max': 13878, 'S': 32.8, 'c': 2.29, 'K': 1 / (np.pi * 6.25 * 0.9),
                      'rho': 0.9, 'delta_zT': -0.378}

    long_perf_true = SimplifiedLongitudinal(p_true, sensor_variance, estimate_variance, aircraft_param)
    long_perf_est = SimplifiedLongitudinal(p_0, sensor_variance, estimate_variance, aircraft_param)

    time = np.linspace(0, 50, int(100))
    x0 = np.array([
        69.371,
        2.0529,
        0.0,
        -0.0171,
        0.0,
        0.0
    ])

    # steady state control inputs
    u = np.zeros([100, 2])
    u[:, 1] = np.ones(100)
    u[:, 0] = np.ones(100) * np.deg2rad(-1.4)

    plt.figure(figsize=(3,2))
    plt.plot(u[:,0])
    plt.title('Constant Control Input')
    plt.ylabel('de [rad]')
    plt.grid()
    plt.show()

    y_true = long_perf_true.state_space_model(time, x0, u, p=p_true)
    y_true_noise = copy.deepcopy(y_true)

    y_true_noise[0, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[0, 0]), size=y_true_noise[0, :].shape)
    y_true_noise[1, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[1, 1]), size=y_true_noise[1, :].shape)
    y_true_noise[2, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[2, 2]), size=y_true_noise[2, :].shape)
    y_true_noise[3, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[3, 3]), size=y_true_noise[3, :].shape)
    y_true_noise[4, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[4, 4]), size=y_true_noise[4, :].shape)
    y_true_noise[5, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[5, 5]), size=y_true_noise[5, :].shape)

    y_initial = long_perf_est.state_space_model(time, x0, u, p=p_0)

    print('beginning estimation')
    long_perf_est.estimate(time, x0, y_true_noise, u, max_iter=20, dp=1E-5, unconverge_tol=1E2)
    print('P before: %s' % long_perf_est.p_0)
    print('P after: %s' % long_perf_est.p)
    print('P Delta: %s' % (long_perf_est.p - long_perf_est.p_0))
    print('Initial Error: %s' % (p_true - long_perf_est.p_0))
    print('End Error: %s' % (p_true - long_perf_est.p))
    print('Finished estimation')
    y_est_update = long_perf_est.state_space_model(time, x0, u, p=long_perf_est.p)
    plt.figure(figsize=(16, 9))
    for ind, value_type in zip(range(6), ['U [m/s]', 'W [m/s]', 'Q [rad/s]', 'Theta [rad]', 'X [m]', 'Z [m]']):
        plt.subplot(3, 2, ind + 1)
        plt.plot(time, y_true[ind, :])
        plt.plot(time, y_true_noise[ind, :])
        plt.plot(time, y_initial[ind, :], '--')
        plt.plot(time, y_est_update[ind, :], '-.')
        plt.title(value_type)
        plt.grid()
    plt.figlegend(['True', 'Measured', 'Initial Guess', 'Updated Estimate'])
    plt.suptitle('Elevator Unobservable Example')
    plt.show()


def simplified_longitudinal_example_thrust_3211():
    p_true = np.array([
        0.895,  # CL_0
        5.01,  # CL_alpha
        0.722,  # CL_de
        0.177,  # CD_0
        0.232,  # CD_alpha
        -0.046,  # CM_0
        -1.087,  # CM_alpha
        -1.88,  # CM_de
        -7.055,  # CM_q
    ])

    p_0 = np.array([
        0.895 + 0.2,  # CL_0
        5.01 - 1.6,  # CL_alpha
        0.722 - 0.2,  # CL_de
        0.177 + 0.05,  # CD_0
        0.232 + 0.01,  # CD_alpha
        -0.046 - 0.01,  # CM_0
        -1.087 - 0.2,  # CM_alpha
        -1.88 + 0.4,  # CM_de
        -7.055 + 0.8,  # CM_q
    ])
    sensor_variance = np.eye(6)
    sensor_variance[0, 0] = 0.4 ** 2
    sensor_variance[1, 1] = 0.4 ** 2
    sensor_variance[2, 2] = 0.05 ** 2
    sensor_variance[3, 3] = np.deg2rad(4.0) ** 2
    sensor_variance[4, 4] = 10.0 ** 2
    sensor_variance[5, 5] = 10.0 ** 2
    estimate_variance = np.eye(p_0.shape[0]) * 0.01  # currently unused
    aircraft_param = {'m': 7484.4, 'Iyy': 84309, 'T_max': 13878, 'S': 32.8, 'c': 2.29, 'K': 1 / (np.pi * 6.25 * 0.9),
                      'rho': 0.9, 'delta_zT': -0.378}

    long_perf_true = SimplifiedLongitudinal(p_true, sensor_variance, estimate_variance, aircraft_param)
    long_perf_est = SimplifiedLongitudinal(p_0, sensor_variance, estimate_variance, aircraft_param)

    time = np.linspace(0, 100, int(200))
    x0 = np.array([
        69.371,
        2.0529,
        0.0,
        -0.0171,
        0.0,
        0.0
    ])

    # steady state control inputs
    u = np.zeros([200, 2])
    u[:, 1] = np.ones(200)
    u[20:50, 1] = np.ones(30)*1.25
    u[50:70, 1] = np.ones(20)*0.75
    u[70:80, 1] = np.ones(10)*1.25
    u[80:90, 1] = np.ones(10)*0.75
    u[:, 0] = np.ones(200) * np.deg2rad(-1.4)
    plt.figure(figsize=(3,2))
    plt.plot(time, u[:,1])
    plt.title('Thrust Control Input')
    plt.ylabel('n [percent]')
    plt.grid()
    plt.show()

    y_true = long_perf_true.state_space_model(time, x0, u, p=p_true)
    y_true_noise = copy.deepcopy(y_true)

    y_true_noise[0, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[0, 0]), size=y_true_noise[0, :].shape)
    y_true_noise[1, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[1, 1]), size=y_true_noise[1, :].shape)
    y_true_noise[2, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[2, 2]), size=y_true_noise[2, :].shape)
    y_true_noise[3, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[3, 3]), size=y_true_noise[3, :].shape)
    y_true_noise[4, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[4, 4]), size=y_true_noise[4, :].shape)
    y_true_noise[5, :] += np.random.normal(loc=0, scale=np.sqrt(sensor_variance[5, 5]), size=y_true_noise[5, :].shape)

    y_initial = long_perf_est.state_space_model(time, x0, u, p=p_0)
    print('beginning estimation')
    long_perf_est.estimate(time, x0, y_true_noise, u, max_iter=100, dp=1E-5)
    print('P before: %s' % long_perf_est.p_0)
    print('P after: %s' % long_perf_est.p)
    print('P Delta: %s' % (long_perf_est.p - long_perf_est.p_0))
    print('Initial Error: %s' % (p_true - long_perf_est.p_0))
    print('End Error: %s' % (p_true - long_perf_est.p))
    print('Finished estimation')
    y_est_update = long_perf_est.state_space_model(time, x0, u, p=long_perf_est.p)
    plt.figure(figsize=(16, 9))
    for ind, value_type in zip(range(6), ['U [m/s]', 'W [m/s]', 'Q [rad/s]', 'Theta [rad]', 'X [m]', 'Z [m]']):
        plt.subplot(3, 2, ind + 1)
        plt.plot(time, y_true[ind, :])
        plt.plot(time, y_true_noise[ind, :])
        plt.plot(time, y_initial[ind, :], '--')
        plt.plot(time, y_est_update[ind, :], '-.')
        plt.title(value_type)
        plt.grid()
    plt.suptitle('Thrust 3211 Example')
    plt.figlegend(['True', 'Measured', 'Initial Guess', 'Updated Estimate'])
    plt.show()