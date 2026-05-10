# %matplotlib widget
import os
import sys
import numpy as np
import matplotlib

# matplotlib.rcParams['font.family'] = 'Times New Roman'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import math
from IPython.display import HTML
from tqdm import tqdm
import random
from numpy.lib import recfunctions as rfn
import multiprocessing

from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.interpolate import interpn

# import cupy as cp
import time
import numba as nb

# 画图部分
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
from functools import partial
import matplotlib.cm as cm

# ROC曲线部分
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

# 插值
import torch
import torch.cuda as cuda
from scipy.interpolate import griddata

# 系数拟合部分
from scipy.optimize import curve_fit
from sklearn import svm


# 概率比较部分
from scipy.special import gammaln
import torch.distributions as dist

# 位置重建函数
import sys

sys.path.append("..")
import position_rec as pos

import DE
from DE import DESimulation
import importlib

# from thundersvm import SVC
index = sys.argv[1]
e_num_sim = 5


def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r) ** (1.0 / g)


def dense_muon_gen(modified_merged_muon_array, interp_num, event_time, dead_time_ratio):
    """生成dense_muon_points数据, 为保证速度，从muon track的首尾线性插值，并且能量均匀分布"""

    # 定义每个元素的类型
    element_types = [
        ("eventId", "<i4"),
        ("energy", "<f8"),
        ("xd", "<f8"),
        ("yd", "<f8"),
        ("zd", "<f8"),
        ("muon_time", "<f8"),
        ("num_e", "<u4"),
        ("num_e_delayed", "<u4"),
    ]
    # 定义muon模拟数量
    muon_sim_num = modified_merged_muon_array["eventId"][-1] + 1
    dense_muon_points = np.zeros(int(muon_sim_num * interp_num), dtype=element_types)
    # 获取modified_merged_muon_array中每个eventId刷新坐标
    unique_elements, start_indices = np.unique(
        modified_merged_muon_array["eventId"], return_index=True
    )
    end_indices = np.concatenate(
        [
            start_indices[1:] - 1,
            np.array(
                [
                    len(modified_merged_muon_array) - 1,
                ]
            ),
        ]
    )

    start_point, end_point = (
        modified_merged_muon_array[start_indices],
        modified_merged_muon_array[end_indices],
    )
    gap_x = -start_point["xd"] + end_point["xd"]
    gap_y = -start_point["yd"] + end_point["yd"]
    gap_z = -start_point["zd"] + end_point["zd"]

    # 创建长度为100的0-1之间差值数组
    arr = np.tile(np.arange(0, 100 / 99, 1 / 99), (muon_sim_num, 1))
    x_dense, y_dense, z_dense = (
        arr * gap_x[:, np.newaxis],
        arr * gap_y[:, np.newaxis],
        arr * gap_z[:, np.newaxis],
    )
    x_dense, y_dense, z_dense = (
        x_dense + start_point["xd"][:, np.newaxis],
        y_dense + start_point["yd"][:, np.newaxis],
        z_dense + start_point["zd"][:, np.newaxis],
    )

    dense_muon_points["xd"], dense_muon_points["yd"], dense_muon_points["zd"] = (
        x_dense.flatten(),
        y_dense.flatten(),
        z_dense.flatten(),
    )
    dense_muon_points["eventId"] = (np.arange(muon_sim_num * interp_num) / 100).astype(
        "uint32"
    )

    # 计算每个muon沉积能量总和及电子总和

    # 提取 eventId 和 electron_num 列
    event_ids = modified_merged_muon_array["eventId"]
    electron_nums = modified_merged_muon_array["electronNum"]
    muon_step_energy = modified_merged_muon_array["energy"]

    # 使用 numpy.unique 获取唯一 eventId 和对应的索引
    unique_event_ids, event_id_indices = np.unique(event_ids, return_inverse=True)

    # 使用 numpy.bincount 计算每个相同 eventId 的 electron_num 总和
    muon_electron_sum = np.bincount(event_id_indices, weights=electron_nums)
    muon_electron_energy = np.bincount(event_id_indices, weights=muon_step_energy)
    muon_electron_sum.astype("uint32")

    # 补全dense_muon_points中的energy，num_e以及num_e_delayed
    average_electron = (muon_electron_sum / 100).astype("uint32")
    average_energy = (muon_electron_energy / 100).astype("uint32")

    dense_muon_points["energy"] = average_energy[dense_muon_points["eventId"]]
    dense_muon_points["num_e"] = average_electron[dense_muon_points["eventId"]]
    dense_muon_points["muon_time"] = event_time[dense_muon_points["eventId"]]

    # 定义残留比例
    delay_ratio = ((dense_muon_points["zd"] - 100) / (-1.7) / 1000 + 0.02) / 100
    delay_ratio[delay_ratio < 0.001] = 0.001
    dense_muon_points["num_e_delayed"] = (
        dense_muon_points["num_e"] * delay_ratio * dead_time_ratio
    )

    return dense_muon_points


file_load_once = 10
param = 0

cevns_sim = DESimulation("DE_param_30_" + str(e_num_sim) + "e.yaml")

# test_cevns_sim.pe_gain_std = 2.4e6

muon_file_list = []
for i in range(file_load_once):
    # print(i)
    file_index = param * file_load_once + i
    filename = (
        cevns_sim.load_folder_path
        + "muon_track/muon_track."
        + str(file_index)
        + (".npy")
    )
    # print(filename)
    muon_file_list.append(filename)

cevns_sim.generate_muon_points(muon_file_list)
cevns_sim.generate_CEVNS_points()
cevns_sim.generate_CEVNS_pattern()
cevns_sim.CEVNS_position_recon()
cevns_sim.generate_CEVNS_recon_pattern()
cevns_sim.CEVNS_sp_cor()

cevns_points = cevns_sim.cevns_points[1]


def calculate_poisson_probabilities_log(data_array, mean_array):
    # 计算每个元素的对数概率
    log_probabilities = (
        -mean_array + data_array * np.log(mean_array) - gammaln(data_array + 1)
    )
    return log_probabilities


# 定义cevns点的三参数
cevns_info = np.zeros(
    len(cevns_points),
    dtype=np.dtype(
        [
            ("area", "<f8"),
            ("st_cor", "<f8"),
            ("pattern", "<f8"),
            ("st_cor_new", "<f8"),
            ("e_num", "uint32"),
        ]
    ),
)
cevns_info["area"] = np.sum(cevns_points["pe_by_area"], axis=1)
cevns_info["st_cor"] = np.log(cevns_points["st_cor"])
# cevns_info['st_cor_new'] = np.log(cevns_points['st_cor_new'])
cevns_info["pattern"] = np.sum(
    calculate_poisson_probabilities_log(
        cevns_points["pe_by_area"][:, :64], cevns_points["recons_light_pattern"][:, :64]
    ),
    axis=1,
)

k = np.load("0824_20000_sigma.npy")
[k_st, k_area, b] = k

sigma_se = 190e-9


def cevns_waveform_gen(pe_info, sigma_se, z):

    pe_num = len(pe_info)

    Dl = 12
    vd = 0.174 * 1000000

    # cevns_z = np.random.uniform(0,24)
    cevns_z = z
    t = cevns_z / vd
    # sigma_d = math.sqrt((2 * Dl * t) / (vd ** 2) + sigma_0 ** 2)
    sigma_t = np.sqrt((2 * Dl * t) / (vd**2))
    sigma = sigma_t
    pe_e_time = np.zeros(len(pe_info))

    for e_id in np.unique(pe_info["e_id"]):
        e_time = np.random.normal(0, sigma)
        mask = pe_info["e_id"] == e_id
        pe_e_time[mask] = e_time
    # print("#########", pe_e_time)

    se_signal_delay = np.random.normal(0, sigma_se, pe_num)
    pe_time = pe_e_time + se_signal_delay
    pe_time = pe_time - np.mean(pe_time)
    # print(pe_time)
    sample_index = pe_time / (4e-9)
    sample_index = sample_index.astype(int)
    # print(sample_index)
    waveform = np.zeros(3500)
    np.add.at(waveform, sample_index + 1750, pe_info["area"] / 6e6)
    std = np.std(sample_index) * 4e-9
    # waveform[sample_index + 1750] += (pe_info['area']/6e6)
    # print(pe_info['area']/6e6)
    # fig = plt.figure(figsize=(7, 4))
    # plt.plot(waveform[1000:-1000])
    # plt.show()
    return waveform, std


def top_waveform_gen(pe_info, sigma_se):

    pe_num = len(pe_info)

    Dl = 12
    vd = 0.174 * 1000000

    cevns_z = 0
    t = cevns_z / vd
    # sigma_d = math.sqrt((2 * Dl * t) / (vd ** 2) + sigma_0 ** 2)
    sigma_t = np.sqrt((2 * Dl * t) / (vd**2))
    sigma = sigma_t
    pe_e_time = np.zeros(len(pe_info))

    for e_id in np.unique(pe_info["e_id"]):
        e_time = np.random.normal(0, sigma)
        mask = pe_info["e_id"] == e_id
        pe_e_time[mask] = e_time
    # print("#########", pe_e_time)

    se_signal_delay = np.random.normal(0, sigma_se, pe_num)
    pe_time = pe_e_time + se_signal_delay
    pe_time = pe_time - np.mean(pe_time)
    # print(pe_time)
    sample_index = pe_time / (4e-9)
    sample_index = sample_index.astype(int)
    # print(sample_index)
    waveform = np.zeros(3500)
    np.add.at(waveform, sample_index + 1750, pe_info["area"] / 6e6)
    std = np.std(sample_index) * 4e-9
    # waveform[sample_index + 1750] += (pe_info['area']/6e6)
    # print(pe_info['area']/6e6)
    # fig = plt.figure(figsize=(7, 4))
    # plt.plot(waveform[1000:-1000])
    # plt.show()
    return waveform, std


e_num_list = np.arange(3, 9)
remain_cevns_id_list = []
accumulate_num = 0
for i in range(len(cevns_sim.cevns_points)):
    # for i in range(len(cevns_sim.cevns_e_spectrum)):

    cevns_points = cevns_sim.cevns_points[i]
    e_num = cevns_sim.cevns_e_num[i]
    # cevns_id = cevns_points['cevnsID'] - accumulate_num
    cevns_id = np.arange(0, len(cevns_points))
    # print(np.min(cevns_id))

    # st_mask
    st_mask = cevns_points["st_cor"] > 0

    cevns_info = np.zeros(
        len(cevns_points[st_mask]),
        dtype=np.dtype([("area", "<f8"), ("st_cor", "<f8"), ("pattern", "<f8")]),
    )
    cevns_info["area"] = np.sum(cevns_points[st_mask]["pe_by_area"], axis=1)
    cevns_info["st_cor"] = np.log(cevns_points[st_mask]["st_cor"])
    cevns_info["pattern"] = np.sum(
        calculate_poisson_probabilities_log(
            cevns_points[st_mask]["pe_by_area"][:, :64],
            cevns_points[st_mask]["recons_light_pattern"][:, :64],
        ),
        axis=1,
    )
    score_cevns = (
        cevns_info["pattern"]
        - (k_st * cevns_info["st_cor"])
        + k_area * cevns_info["area"]
        - b
    )

    cevns_pass_pattern = cevns_info[score_cevns > 0]

    cevns_id_remain = cevns_id[score_cevns > 0]

    cevns_pe = cevns_sim.cevns_pe_info[i]

    ########
    # print(cevns_pe['event_id'][-1])
    # print(np.max(cevns_id))
    print(len(cevns_id_remain) / len(cevns_id))

    accumulate_num += len(cevns_points)

    waveform_list = []
    std_list = []
    waveform_list_top = []
    std_list_top = []
    z_list = np.linspace(0, 24, len(cevns_id_remain))
    for j in tqdm(range(len(cevns_id_remain))):

        cevns_id_event = cevns_id_remain[j]
        # print('cevns_id_event: ', cevns_id_event)

        pe_mask = cevns_pe["event_id"] == cevns_id_event
        pe_info = cevns_pe[pe_mask]
        # print(len(pe_info))
        pe_num = len(pe_info)
        pe_area = np.sum(pe_info["area"])
        #         print("area1: ", pe_area / 6e6)

        #         print("area2: ", cevns_pass_pattern[j]['area'])

        #         print("Score: ", score_cevns[score_cevns > 0][j])

        waveform, std = cevns_waveform_gen(pe_info, sigma_se, z_list[j])
        # waveform_top,std_top = top_waveform_gen(pe_info,sigma_se)
        waveform[waveform < 0] = 0
        waveform_list.append(waveform)
        std_list.append(std)

    if e_num == e_num_sim:
        # np.save('waveform_result/z_pass_pattern_cevns_3_' + str(e_num) + '_' + str(index), z_list)
        # np.save('waveform_result/wf_pass_pattern_cevns_3_' + str(e_num) + '_' + str(index), waveform_list)
        # np.save('waveform_result/wf_std_pass_pattern_cevns_3_' + str(e_num) + '_' + str(index), std_list)
        # np.save('waveform_result/area_cevns_3_' + str(e_num) + '_' + str(index), cevns_info['area'])
        np.savez(
            "waveform_result_0824/wf_pass_pattern_cevns_"
            + str(e_num)
            + "_"
            + str(index)
            + ".npz",
            z_pass_pattern=z_list,
            wf_pass_pattern=waveform_list,
            wf_std_pass_pattern=std_list,
            area_cevns=cevns_info["area"],
        )
