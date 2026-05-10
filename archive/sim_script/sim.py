# %matplotlib widget
import sys

sys.path.append("..")

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
import position_rec as pos

import DE
from DE import DESimulation
import importlib

# from thundersvm import SVC


index = int(sys.argv[1])
filepath = sys.argv[2]

print("Received parameter:", index)

file_load_once = 5

test_DE_sim = DESimulation(filepath)

muon_file_list = []
for i in range(file_load_once):
    print(i)
    file_index = index * file_load_once + i
    filename = (
        test_DE_sim.load_folder_path
        + "muon_track/muon_track."
        + str(int(file_index % 10000))
        + (".npy")
    )
    print(filename)
    muon_file_list.append(filename)


# 根据生成的muon模拟file生成对应的延迟电子
print("############# \n Start generating DEs \n #############")
test_DE_sim.generate_muon_points(muon_file_list)
test_DE_sim.generate_DE_points()
print(
    "模拟对应仪器时间[H]: :",
    test_DE_sim.merged_muon_array["eventId"][-1] / 10.25 / 3600,
)

# 进行pattern模拟以及时空关联度计算
test_DE_sim.generate_Pile_UP_2e()
test_DE_sim.generate_Pile_UP_3e()
test_DE_sim.generate_Pile_UP_multi_e()
test_DE_sim.generate_DE_pattern_linear()
test_DE_sim.DE_position_recon()
test_DE_sim.generate_DE_recon_pattern()
test_DE_sim.DE_sp_cor()


def calculate_poisson_probabilities_log(data_array, mean_array):
    # 计算每个元素的对数概率
    log_probabilities = (
        -mean_array + data_array * np.log(mean_array) - gammaln(data_array + 1)
    )
    return log_probabilities


# test_DE_sim.pile_up_sp_cor
pile_up_sp_cor = np.log(np.concatenate(test_DE_sim.pile_up_sp_cor))
pile_up_e_num = []

for i in range(len(test_DE_sim.pile_up_e_num)):
    num_e = test_DE_sim.pile_up_e_num[i]
    pile_up_e_num.append(np.ones(len(test_DE_sim.pile_up_sp_cor[i])) * num_e)

pile_up_e_num = np.concatenate(pile_up_e_num)

pile_up_pattern = []
pile_up_area = []
pe_info_list = []
num_event = 0
for i in range(len(test_DE_sim.pile_up_e_num)):
    area_pattern = test_DE_sim.pile_up_pe_by_area[i]
    recon_pattern = test_DE_sim.pile_up_recons_light_pattern[i]
    pe_info = np.copy(test_DE_sim.pile_up_pe_info[i])
    print("pe_info['event_id']", pe_info["event_id"][-1])
    pe_info["event_id"] = pe_info["event_id"] + num_event
    print("pe_info['event_id']", pe_info["event_id"][-1])
    pe_info_list.append(pe_info)
    print("num of pattern: ", len(recon_pattern))
    print(num_event)
    num_event += len(recon_pattern)
    print(num_event)
    pattern = np.sum(
        calculate_poisson_probabilities_log(
            area_pattern[:, :64], recon_pattern[:, :64]
        ),
        axis=1,
    )
    pile_up_pattern.append(pattern)
    area = np.sum(area_pattern, axis=1)
    pile_up_area.append(area)

pile_up_pattern_coefficient = np.concatenate(pile_up_pattern)
pile_up_area = np.concatenate(pile_up_area)
area_pattern = np.concatenate(test_DE_sim.pile_up_pe_by_area)
pile_up_pe_info = np.concatenate(pe_info_list)

data_save = {"pile_up_pattern_coefficient": pile_up_pattern_coefficient}
data_save["pile_up_area"] = pile_up_area
data_save["area_pattern"] = area_pattern
data_save["pile_up_pe_info"] = pile_up_pe_info
data_save["pile_up_e_num"] = pile_up_e_num
data_save["pile_up_sp_cor"] = pile_up_sp_cor


np.savez("DE_result_0822/" + str(index * file_load_once) + ".npz", **data_save)
