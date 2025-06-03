import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import rfft, fftfreq, fftshift


class OpenFOAMForceCoeffDataStruct(object):
    pass


def plot_coefficents(time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw, title='', save_dir=None, bins=1, show=True):
    """

    :param np.ndarray time: Array of time values for plotting.
    :param np.ndarray cd: Array of cd coeff values for plotting.
    :param np.ndarray cl: Array of cl coeff values for plotting.
    :param np.ndarray cs: Array of cs coeff values for plotting.
    :param np.ndarray cm_roll: Array of cm_roll coeff values for plotting.
    :param np.ndarray cm_pitch: Array of cm_pitch coeff values for plotting.
    :param np.ndarray cm_yaw: Array of cm_yaw coeff values for plotting.
    :param str title: Title for the plots/output file.
    :param str save_dir: Directory to save plots to, if None images will not be created.
    :param int bins: Number of bins for running average of the data.
    """
    fig1, (ax1, ax2) = plt.subplots(2, 1)

    avg_cl = moving_avg(cl, bins)
    avg_cd = moving_avg(cd, bins)
    avg_cs = moving_avg(cs, bins)

    avg_cr = moving_avg(cm_roll, bins)
    avg_cp = moving_avg(cm_pitch, bins)
    avg_cy = moving_avg(cm_yaw, bins)

    excised_time = time[bins:]

    ax1.plot(excised_time, avg_cl)
    ax1.plot(excised_time, avg_cd)
    ax1.plot(excised_time, avg_cs)
    ax1.legend(['Cl', 'Cd', 'Cs'])
    ax1.set_title('Force Coeffs: %s' % title)

    ax2.plot(excised_time, avg_cr)
    ax2.plot(excised_time, avg_cp)
    ax2.plot(excised_time, avg_cy)
    ax2.legend(['CmRoll', 'CmPitch', 'CmYaw'])
    ax2.set_title('Moment Coeffs: %s' % title)

    fig1.set_size_inches(20, (20 * 9 / 16.0))

    if save_dir is not None:
        if title == '':
            title = 'tmp'
        fig1.savefig(fname=os.path.join(save_dir, '%s.png' % title), backend='Cairo')

    fig2, (ax3, ax4) = plt.subplots(2, 1)
    ax3.plot(excised_time, diff(excised_time, avg_cl))
    ax3.plot(excised_time, diff(excised_time, avg_cd))
    ax3.plot(excised_time, diff(excised_time, avg_cs))
    ax3.legend(['d(Cl)/dt', 'd(Cd)/dt', 'd(Cs)/dt'])
    ax3.set_title('Force Coeffs Diff: %s' % title)

    ax4.plot(excised_time, diff(excised_time, avg_cr))
    ax4.plot(excised_time, diff(excised_time, avg_cp))
    ax4.plot(excised_time, diff(excised_time, avg_cy))
    ax4.legend(['d(CmRoll)/dt', 'd(CmPitch)/dt', 'd(CmYaw)/dt'])
    ax4.set_title('Moment Coeffs Diff: %s' % title)

    fig2.set_size_inches(20, (20 * 9 / 16.0))

    if save_dir is not None:
        if title == '':
            title = 'tmp'
        fig2.savefig(fname=os.path.join(save_dir, '%s_diff.png' % title), backend='Cairo')
    if show:
        plt.show()


def print_stats(val, excise, variable_name):
    print('=' * 80)
    print('%s average: %s' % (variable_name, np.average(val[excise:])))
    print('%s stdev: %s' % (variable_name, np.std(val[excise:])))
    print('%s min: %s' % (variable_name, np.min(val[excise:])))
    print('%s max: %s' % (variable_name, np.max(val[excise:])))
    print('=' * 80)


def moving_avg(val, bins):
    bin_arr = np.zeros(bins)
    ret_val = np.zeros(len(val))
    for idx in range(0, len(val)):
        bin_arr[idx % bins] = val[idx]
        if idx < bins:
            ret_val[idx] = val[idx]
        else:
            ret_val[idx] = np.average(bin_arr)

    return ret_val[bins:]


def plot_fft(val, save_dir=None, title='', sampling_rate=1e-3):
    xf = fftfreq(len(val), d=sampling_rate)
    xf = fftshift(xf)
    y = rfft(val, len(val))

    plt.semilogy(xf, np.abs(y))
    plt.title('Force Coeffs Diff: %s' % title)

    if save_dir is not None:
        if title == '':
            title = 'tmp'
        plt.savefig(fname=os.path.join(save_dir, '%s_diff.png' % title), backend='Cairo')
    plt.show()


def diff(time, val):
    differencne = np.zeros(len(val))

    for idx in range(1, len(time) - 1):
        differencne[idx] = (val[idx + 1] - val[idx - 1]) / (time[idx + 1] - time[idx - 1])

    return differencne

def get_average_sampling_rate(time):
    diff = time[1:-1] - time[0:-2]
    return np.average(diff)

def format_force_data_as_np_arr(data):
    keys = sorted(data.keys())
    data_len = len(data)

    time = np.zeros(data_len)
    cd = np.zeros(data_len)
    cl = np.zeros(data_len)
    cs = np.zeros(data_len)
    cm_roll = np.zeros(data_len)
    cm_pitch = np.zeros(data_len)
    cm_yaw = np.zeros(data_len)

    idx = 0
    for key in keys:
        time[idx] = float(key)
        cd[idx] = float(data[key]['Cd'])
        cl[idx] = float(data[key]['Cl'])
        cs[idx] = float(data[key]['Cs'])
        cm_roll[idx] = float(data[key]['CmRoll'])
        cm_pitch[idx] = float(data[key]['CmPitch'])
        cm_yaw[idx] = float(data[key]['CmYaw'])
        idx += 1

    return time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw


def parse_force_coeff(directory, logger, excise=0):
    files = find_files(directory, 'dat')
    # Data is contained within a dictionary with the time used as the key, this is primarily to protect against sim
    # restarts which could produce duplicate data for time steps
    data = dict()
    data_keys = {'time': 0, 'Cd': 1, 'Cs': 2, 'Cl': 3, 'CmRoll': 4, 'CmPitch': 5, 'CmYaw': 6, 'Cd(f)': 7, 'Cd(r)': 8,
                 'Cs(f)': 9, 'Cs(r)': 10, 'Cl(f)': 11, 'Cl(r)': 12}
    for file in files:
        logger.info('Opening File: %s ' % file)
        with open(file) as data_file:
            for line in data_file.readlines():
                if '#' not in line:
                    items = line.split('\t')
                    # logger.info('time: %s' % items[data_keys['time']])
                    if float(items[data_keys['time']]) >= excise:
                        data[float(items[data_keys['time']])] = {'Cd': float(items[data_keys['Cd']]),
                                                                 'Cs': float(items[data_keys['Cs']]),
                                                                 'Cl': float(items[data_keys['Cl']]),
                                                                 'CmRoll': float(items[data_keys['CmRoll']]),
                                                                 'CmPitch': float(items[data_keys['CmPitch']]),
                                                                 'CmYaw': float(items[data_keys['CmYaw']]),
                                                                 'Cd(f)': float(items[data_keys['Cd(f)']]),
                                                                 'Cd(r)': float(items[data_keys['Cd(r)']]),
                                                                 'Cs(f)': float(items[data_keys['Cs(f)']]),
                                                                 'Cs(r)': float(items[data_keys['Cs(r)']]),
                                                                 'Cl(f)': float(items[data_keys['Cl(f)']]),
                                                                 'Cl(r)': float(items[data_keys['Cl(r)']]),
                                                                 }

    return data


def find_files(directory, file_ext):
    files = list()
    for path in Path(directory).rglob('*.%s' % file_ext):
        files.append(path)
    return files


def find_directory(top_directory, directory_name):
    directory_paths = list()
    for direct in Path(top_directory).rglob('*%s*' % directory_name):
        if os.path.isdir(direct):
            directory_paths.append(direct)
    return directory_paths


def aoa_aircraft_analysis(top_level_directory, logger, output_directory, force_coeff_widget_name, averaging_window=400):
    directories_to_parse = find_directory(top_level_directory, 'AOA_0Flap')
    collected_data = dict()
    for directory in directories_to_parse:
        folder_name = os.path.split(directory)[1]
        aoa = folder_name[:folder_name.find('AOA')]
        logger.debug(aoa)
        if 'm' in aoa:
            aoa = -int(aoa[1:])
        else:
            aoa = int(aoa)
        if os.path.exists(os.path.join(directory, 'postProcessing', force_coeff_widget_name)):
            data = parse_force_coeff(os.path.join(directory, 'postProcessing', force_coeff_widget_name),
                                     logger, excise=400)
            (time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw) = format_force_data_as_np_arr(data)
            plot_coefficents(time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw, title=folder_name,
                             save_dir=output_directory,
                             show=False, bins=1)
            collected_data[aoa] = {'cl': cl, 'cd': cd, 'cs': cs, 'cm_roll': cm_roll, 'cm_pitch': cm_pitch,
                                   'cm_yaw': cm_yaw}
    aoa_list = list()
    cl = list()
    cd = list()
    cl_cd = list()
    cs = list()
    cm_roll = list()
    cm_pitch = list()
    cm_yaw = list()

    for aoa in sorted(collected_data.keys()):
        aoa_list.append(aoa)
        cl.append(np.average(collected_data[aoa]['cl'][-averaging_window:]))
        cd.append(np.average(collected_data[aoa]['cd'][-averaging_window:]))
        cs.append(np.average(collected_data[aoa]['cs'][-averaging_window:]))
        cm_roll.append(np.average(collected_data[aoa]['cm_roll'][-averaging_window:]))
        cm_yaw.append(np.average(collected_data[aoa]['cm_yaw'][-averaging_window:]))
        cm_pitch.append(np.average(collected_data[aoa]['cm_pitch'][-averaging_window:]))

        cl_cd.append(cl[-1] / cd[-1])

    plot_aoa_analysis(aoa_list, cl, cd, cl_cd, cs, cm_roll, cm_pitch, cm_yaw, output_directory)


def plot_aoa_analysis(aoa, cl, cd, cl_cd, cs, cm_roll, cm_pitch, cm_yaw, save_dir):
    plt.figure(figsize=(16, 9), dpi=160)
    plt.plot(cd, cl)
    plt.title('Cl vs Cd')
    plt.savefig(fname=os.path.join(save_dir, 'cl_vs_cd.png'), backend='svg')

    plt.figure(figsize=(16, 9), dpi=160)
    plt.plot(aoa, cl)
    plt.title('Cl vs AoA')
    plt.savefig(fname=os.path.join(save_dir, 'cl_vs_aoa.png'), backend='svg')

    plt.figure(figsize=(16, 9), dpi=160)
    plt.plot(aoa, cd)
    plt.title('Cd vs AoA')
    plt.savefig(fname=os.path.join(save_dir, 'cd_vs_aoa.png'), backend='svg')

    plt.figure(figsize=(16, 9), dpi=160)
    plt.plot(aoa, cl_cd)
    plt.title('Cl/Cd vs AoA')
    plt.savefig(fname=os.path.join(save_dir, 'cl_cd_vs_aoa.png'), backend='svg')

    plt.figure(figsize=(16, 9), dpi=160)
    plt.plot(aoa, cm_roll)
    plt.plot(aoa, cm_pitch)
    plt.plot(aoa, cm_yaw)
    plt.title('Moment Coeff vs AoA')
    plt.legend(['cm_roll', 'cm_pitch', 'cm_yaw'])
    plt.savefig(fname=os.path.join(save_dir, 'moment_coeff_vs_aoa.png'), backend='svg')


def reynold_aircraft_analysis(top_level_directory, logger, output_directory, force_coeff_widget_name, s, rho, l_ref,
                              averaging_window=400):
    directories_to_parse = find_directory(top_level_directory, '_RE')
    collected_data = dict()
    for directory in directories_to_parse:
        folder_name = os.path.split(directory)[1]
        re = float(folder_name.split('_')[1])
        logger.debug(re)
        if os.path.exists(os.path.join(directory, 'postProcessing', force_coeff_widget_name)):
            data = parse_force_coeff(os.path.join(directory, 'postProcessing', force_coeff_widget_name),
                                     logger, excise=400)
            (time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw) = format_force_data_as_np_arr(data)
            plot_coefficents(time, cd, cl, cs, cm_roll, cm_pitch, cm_yaw, title=folder_name,
                             save_dir=output_directory,
                             show=False, bins=1)
            collected_data[re] = {'cl': cl, 'cd': cd, 'cs': cs, 'cm_roll': cm_roll, 'cm_pitch': cm_pitch,
                                   'cm_yaw': cm_yaw}
    re_list = list()
    cl = list()
    lift = list()
    cd = list()
    drag = list()
    lift_drag = list()
    cl_cd = list()
    cs = list()
    cm_roll = list()
    cm_pitch = list()
    cm_yaw = list()

    vel = list()

    for re in sorted(collected_data.keys()):
        re_list.append(re)
        print(re)
        u = (re * 1.5E-5) / l_ref
        vel.append(u)
        q = 0.5 * s * rho * u ** 2
        cl.append(np.average(collected_data[re]['cl'][-averaging_window:]))
        print(cl[-1])
        lift.append(cl[-1] * q / 9.81)
        cd.append(np.average(collected_data[re]['cd'][-averaging_window:]))
        drag.append(cd[-1] * q / 9.81)
        print(cd[-1])
        cs.append(np.average(collected_data[re]['cs'][-averaging_window:]))
        cm_roll.append(np.average(collected_data[re]['cm_roll'][-averaging_window:]))
        cm_yaw.append(np.average(collected_data[re]['cm_yaw'][-averaging_window:]))
        cm_pitch.append(np.average(collected_data[re]['cm_pitch'][-averaging_window:]))

        cl_cd.append(cl[-1] / cd[-1])

    plot_reynolds_analysis(re_list, cl, cd, cl_cd, cs, cm_roll, cm_pitch, cm_yaw, lift, drag, rho, s, l_ref,
                           output_directory)


def plot_reynolds_analysis(re, cl, cd, cl_cd, cs, cm_roll, cm_pitch, cm_yaw, lift, drag, rho, s, l_ref, save_dir):
    def rey_to_vel(rey):
        return (rey * 1.5E-5)/l_ref
    def vel_to_rey(vel):
        return vel * l_ref / 1.5E-5

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, cl)
    plt.title('Cl vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.xlabel('Reynolds Number')
    plt.ylabel('Cl')
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.savefig(fname=os.path.join(save_dir, 'cl_vs_reynolds.png'), backend='svg')

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, cd)
    plt.title('Cd vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.xlabel('Reynolds Number')
    plt.ylabel('Cd')
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.savefig(fname=os.path.join(save_dir, 'cd_vs_reynolds.png'), backend='svg')

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, cl_cd)
    plt.title('Cl/Cd vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.xlabel('Reynolds Number')
    plt.ylabel('Cl/Cd')
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.savefig(fname=os.path.join(save_dir, 'cl_cd_vs_reynolds.png'), backend='svg')

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, cm_roll)
    plt.plot(re, cm_pitch)
    plt.plot(re, cm_yaw)
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.title('Moment Coeff vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.legend(['cm_roll', 'cm_pitch', 'cm_yaw'])
    plt.xlabel('Reynolds Number')
    plt.ylabel('Cm')
    plt.savefig(fname=os.path.join(save_dir, 'moment_coeff_vs_reynolds.png'), backend='svg')

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, lift)
    plt.title('Lift vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.xlabel('Reynolds Number')
    plt.ylabel('Lift [kg]')
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.savefig(fname=os.path.join(save_dir, 'lift_vs_reynolds.png'), backend='svg')

    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot(re, drag)
    plt.title('Drag vs Reynolds Number\nrho: %skg/m³, S: %sm², L_ref: %sm' % (rho, s, l_ref))
    plt.xlabel('Reynolds Number')
    plt.ylabel('Drag [kg]')
    plt.grid(visible=True)
    ax2 = ax.secondary_xaxis('top', functions=(rey_to_vel, vel_to_rey))
    ax2.set_xlabel('Velocity [m/s]')
    plt.savefig(fname=os.path.join(save_dir, 'drag_vs_reynolds.png'), backend='svg')


def get_time_steps(directory):
    items = os.listdir(directory)
    times = list()
    for item in items:
        if item not in ['system', 'constant', '0.orig'] and Path(os.path.join(directory,item)).is_dir():
            times.append(float(item))
    return sorted(times)

def load_from_directory(variable, directory, grid_size, scalar=True, skip_zero=True):
    times = get_time_steps(directory)

    data = np.zeros([len(times), grid_size])
    for t_ind, time in enumerate(times):
        if skip_zero and t_ind == 0:
            continue
        file_path = os.path.join(directory, str(time), variable)
        with open(file_path, 'r') as csvfile:
            tmp_data = list()
            found_start1 = False
            found_start2 = False
            end = False
            for line in csvfile:
                if not end:
                    if 'internalField' in line:
                        found_start1 = True
                    if '(' in line and found_start1 and not found_start2:
                        found_start2 = True
                        continue
                    if ')' in line and '(' not in line and found_start2:
                        end = True
                        continue
                    if found_start2:
                        if scalar:
                            tmp_data.append(float(line))
                        else:
                            tmp_data.append(float(line[1:line.index(' ')+1]))
            data[t_ind, :] = np.array(tmp_data)

    return data