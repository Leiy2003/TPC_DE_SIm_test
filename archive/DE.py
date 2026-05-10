import os
import sys
import numpy as np
import matplotlib
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

import yaml

# import cupy as cp
import time
import numba as nb

# 画图部分
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
from functools import partial

# ROC曲线部分
from sklearn.metrics import roc_curve, auc
from sklearn import metrics

# 插值
import torch
import torch.cuda as cuda
from scipy.interpolate import griddata

# 系数拟合部分
from scipy.optimize import curve_fit

# 概率比较部分
from scipy.special import gammaln
import torch.distributions as dist

# 位置重建函数
import position_rec as pos


pmt_dtype = np.dtype(
    [
        ("ChannelID", "<i4"),
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("rot_x", "<f8"),
        ("rot_y", "<f8"),
        ("rot_z", "<f8"),
    ]
)


def rndm(a, b, g, size=1):
    """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r) ** (1.0 / g)


class DESimulation:
    def __init__(self, filepath):

        with open(filepath, "r") as file:
            params = yaml.safe_load(file)

        # 模拟需要的基本文件
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_folder_path = os.path.join(current_dir, "file_to_load/")

        # 模拟输出的文件路径
        self.result_save_path = params["output_pattern"]

        # muon结束之后的死时间,单位为s
        self.dead_time = params["dead_time"]

        # pmt 坐标信息
        pmt_top = np.loadtxt(params["pmt_top"], dtype=pmt_dtype)
        pmt_bot = np.loadtxt(params["pmt_bot"], dtype=pmt_dtype)
        self.pmt_info = np.concatenate((pmt_top, pmt_bot))

        # 光模拟数据
        self.file_list = ["./file_to_load/LCE_xs.npy", "./file_to_load/LCE_ys.npy"]
        self.LCE_xs = np.load(params["LCE_xs"])
        self.LCE_ys = np.load(params["LCE_ys"])
        self.LCE_value = np.load(params["LCE_value"])

        # 假设我们知道的真实LCE分布
        self.LCE_value_true = np.load(params["LCE_value_true"])
        
        # muon track file path
        self.muon_track_file = params["muon_track_file"]

        # muon rate
        self.muon_rate = params["muon_rate"]

        # muon每keV电子产额
        self.quenching_factor = params["quenching_factor"]

        # 合并事件最大事件间隔[s]
        self.gap = float(params["gap"])

        # 平均产生单个光子子各通道接收到pe数
        self.average_efficiency = params["average_efficiency"]
        self.single_electron_pe = params["single_electron_gain"]
        self.single_electron_gain = (
            params["single_electron_gain"] / self.average_efficiency
        )

        # 平均single PE gain 以及均方根
        self.pe_gain = float(params["pe_gain"])
        # self.pe_gain_std = 0.6e6
        self.pe_gain_std = float(params["pe_gain_std"])

        # RELICS死时间设计范围内的延迟电子数计算
        delay_time = rndm(0.002, 2, g=-0.1, size=int(1e7))
        self.dead_time_ratio = delay_time[delay_time >= self.dead_time].shape[0] / (
            delay_time.shape[0]
        )

        # pileup结果的存储，按照电子数与事件信息
        self.pile_up_e_num = []
        self.pile_up_result = []
        self.pile_up_indices = []

        # CEVNS电子数能谱，仅考虑1电子及以上，10电子及以下部分

        bg = np.load(params["cevns_e_spectrum"])
        self.cevns_e_spectrum = bg["CEvNS"][1:11]
        self.cevns_e_num = bg["CEvNS_bins"][1:11]
        self.cevns_e_spectrum = (self.cevns_e_spectrum).astype("int")

        # bg = np.load(params['cevns_e_spectrum'])
        # self.cevns_e_spectrum = bg['CEvNS'][3:10] * 32
        # self.cevns_e_spectrum = self.cevns_e_spectrum / \
        #     np.sum(self.cevns_e_spectrum)
        # self.cevns_e_num = bg['CEvNS_bins'][3:10]
        # self.cevns_num = int(1e6)
        # self.cevns_e_spectrum = (
        #     self.cevns_num * self.cevns_e_spectrum).astype('int')

        self.time_range = 0

        # 探测器内部信号范围半径
        self.radius = 139
        self.radius_fiducial = params["CEVNS_radius_range"]
        self.height = 310

        # 延迟电子水平方向扩散
        self.diffuse_length = params["diffuse_length"]

        # 保留事件面积上下限
        self.sim_up_limit = params["sim_up_limit"]
        self.sim_low_limit = params["sim_low_limit"]
        
        # 位置重建使用model
        self.position_reconstruction_model = params['position_reconstruction_model']

        # pe 数据格式
        self.pe_dtype = np.dtype(
            [
                ("event_id", "uint64"),
                ("area", "<f8"),
                ("channel", "uint16"),
                ("e_time", "<f8"),
                ("e_id", "uint32"),
            ]
        )

    def generate_muon_points(self, muon_file_list):
        # 延迟电子生成
        # 输入: muon粒子路径，单次muon死时间，延迟电子漂移范围
        # 输出: N个电子的信息 (x, y, t, N_e)

        pre_event_num = 0
        muon_arrays = []
        for index in tqdm(range(len(muon_file_list))):
            # 对于给定的文件列表，读取其中的muon径迹，并生成总体的muon_id
            filename = muon_file_list[index]
            if index == 0:
                print('First muon track file: {}'.format(filename))
            if index == len(muon_file_list) - 1:
                print('Last muon track file: {}'.format(filename))
            array = np.load(filename)
            file_event_num = array["eventId"][-1] + 1
            array["eventId"] += pre_event_num
            pre_event_num += file_event_num
            muon_arrays.append(array)

        self.merged_muon_array = np.concatenate(muon_arrays, axis=0)
        print("Number of muon events:", self.merged_muon_array["eventId"][-1] + 1)
        print("merged_muon_array data type:", self.merged_muon_array.dtype)

        total_muon_num = self.merged_muon_array["eventId"][-1]
        self.time_range = total_muon_num / self.muon_rate
        event_time = [
            random.uniform(0, self.time_range) for _ in range(total_muon_num + 1)
        ]
        event_time.sort()
        self.event_time = np.array(event_time)

        # 给定muon的电子产额
        electron_num = self.quenching_factor * self.merged_muon_array["energy"].astype(
            "uint32"
        )

        # 定义新字段
        new_field = [("electronNum", "uint32"), ("muon_time", "<f8")]

        # 在现有的dtype基础上增加新字段
        new_dtype = np.dtype(self.merged_muon_array.dtype.descr + new_field)
        modified_merged_muon_array = np.zeros(
            len(self.merged_muon_array), dtype=new_dtype
        )

        # 将merged_muon_array中的数据赋值给modified_merged_muon_array中的对应类型
        for field in self.merged_muon_array.dtype.names:
            modified_merged_muon_array[field] = self.merged_muon_array[field]

        # 将电子数量以及muon时间对应到每个event上
        modified_merged_muon_array["electronNum"] = electron_num
        modified_merged_muon_array["muon_time"] = self.event_time[
            self.merged_muon_array["eventId"]
        ]
        self.merged_muon_array = modified_merged_muon_array

        # 对muon径迹进行插值，并且根据不同深度的能量沉积计算对应的延迟电子数量，最后结果为柏松随机
        interp_num = 100  # 插值个数
        self.dense_muon_points = dense_muon_gen(
            modified_merged_muon_array,
            interp_num,
            self.event_time,
            self.dead_time_ratio,
        )  # 无随机过程
        print("dense_muon_points data type: ", self.dense_muon_points.dtype)

    def generate_DE_points(self):
        self.delayed_electron = delayed_electrons_sorted(
            self.dense_muon_points, self.dead_time, self.diffuse_length
        )

    def generate_CEVNS_points(self):
        points = []
        point_num = 0
        # 定义cevns的数据类型
        cevns_types = np.dtype(
            [
                ("cevnsID", "<i4"),
                (
                    "pos_original",
                    np.dtype([("xd", "<f8"), ("yd", "<f8"), ("zd", "<f8")]),
                ),
                ("t", "<f8"),
                ("num_e", "<f8"),
                ("light_pattern", "<f8", (128,)),
                ("pe_pattern", "uint16", (128,)),
                ("area_pattern", "<f8", (128,)),
                ("pe_by_area", "<f8", (128,)),
                ("pos_recon", np.dtype([("xd", "<f8"), ("yd", "<f8")])),
                ("recons_light_pattern", "<f8", (128,)),
                ("st_cor", "<f8"),
                ("st_cor_new", "<f8"),
            ]
        )
        for i in range(len(self.cevns_e_num)):
            random_points = np.zeros(self.cevns_e_spectrum[i], dtype=cevns_types)
            # 生成ID
            random_points["cevnsID"] = np.arange(self.cevns_e_spectrum[i]) + point_num

            # 生成坐标
            angle = (
                2 * math.pi * np.random.random(size=self.cevns_e_spectrum[i])
            )  # 随机生成一个角度
            distance = self.radius_fiducial * np.sqrt(
                np.random.random(size=self.cevns_e_spectrum[i])
            )  # 随机生成一个距离
            random_points["pos_original"]["zd"] = self.height * np.random.random(
                size=self.cevns_e_spectrum[i]
            )
            random_points["t"] = (self.time_range - 2) * np.random.random(
                size=self.cevns_e_spectrum[i]
            ) + 2
            random_points["pos_original"]["xd"] = distance * np.cos(
                angle
            )  # 计算 x 坐标
            random_points["pos_original"]["yd"] = distance * np.sin(
                angle
            )  # 计算 y 坐标

            # 对电子数量赋值
            random_points["num_e"] = self.cevns_e_num[i]

            # 按照时间排序
            random_points = np.sort(random_points, order="t")

            # point = np.array([x, y, z, t], dtype=point_dtype)
            points.append(random_points)
            point_num += self.cevns_e_spectrum[i]
        self.cevns_points = points

    def clean_pile_up(self):
        self.pile_up_e_num = []
        self.pile_up_result = []
        self.pile_up_indices = []

    def generate_Pile_UP_2e(self):
        adjecent_mask = (
            self.delayed_electron["e_time"][1:] - self.delayed_electron["e_time"][:-1]
        ) < self.gap
        # 连续2电子事件，需要开头电子与前一个不相连
        start_mask = np.insert(np.logical_not(adjecent_mask), 0, True)
        end_mask = np.insert(np.logical_not(adjecent_mask), -1, True)
        adjacent_2e = adjecent_mask
        fake_2e_mask = np.logical_and(
            np.logical_and(adjacent_2e, start_mask[:-1]), end_mask[1:]
        )

        fake_2e_event_indices = np.arange(len(fake_2e_mask))[fake_2e_mask][
            :, np.newaxis
        ] + np.arange(2)
        fake_2e_event = np.take(
            self.delayed_electron[:-1], fake_2e_event_indices, axis=0, mode="clip"
        )

        self.pile_up_e_num.append(2)
        self.pile_up_result.append(fake_2e_event)
        self.pile_up_indices.append(fake_2e_event_indices)

    def generate_Pile_UP_3e(self):
        adjecent_mask = (
            self.delayed_electron["e_time"][1:] - self.delayed_electron["e_time"][:-1]
        ) < self.gap
        # 连续3电子事件，需要开头电子与前一个不相连
        start_mask = np.insert(np.logical_not(adjecent_mask), 0, True)
        end_mask = np.insert(np.logical_not(adjecent_mask), -1, True)
        adjacent_3e = np.logical_and(adjecent_mask[:-1], adjecent_mask[1:])
        fake_3e_mask = np.logical_and(
            np.logical_and(adjacent_3e, start_mask[:-2]), end_mask[2:]
        )

        fake_3e_event_indices = np.arange(len(fake_3e_mask))[fake_3e_mask][
            :, np.newaxis
        ] + np.arange(3)
        fake_3e_event = np.take(
            self.delayed_electron[:-2], fake_3e_event_indices, axis=0, mode="clip"
        )

        self.pile_up_e_num.append(3)
        self.pile_up_result.append(fake_3e_event)
        self.pile_up_indices.append(fake_3e_event_indices)

    def generate_Pile_UP_multi_e(self):
        adjecent_mask = (
            self.delayed_electron["e_time"][1:] - self.delayed_electron["e_time"][:-1]
        ) < self.gap
        start_mask = np.insert(np.logical_not(adjecent_mask), 0, True)
        end_mask = np.insert(np.logical_not(adjecent_mask), -1, True)

        # 连续4电子事件，需要开头电子与前一个不相连
        adjacent_4e = np.logical_and(
            np.logical_and(adjecent_mask[:-2], adjecent_mask[1:-1]), adjecent_mask[2:]
        )
        fake_4e_mask = np.logical_and(
            np.logical_and(adjacent_4e, start_mask[:-3]), end_mask[3:]
        )

        # 连续5电子事件，需要开头电子与前一个不相连
        adjacent_5e = np.logical_and(
            np.logical_and(
                np.logical_and(adjecent_mask[:-3], adjecent_mask[1:-2]),
                adjecent_mask[2:-1],
            ),
            adjecent_mask[3:],
        )
        fake_5e_mask = np.logical_and(
            np.logical_and(adjacent_5e, start_mask[:-4]), end_mask[4:]
        )

        # 连续6电子事件，需要开头电子与前一个不相连
        adjacent_6e = np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.logical_and(adjecent_mask[:-4], adjecent_mask[1:-3]),
                    adjecent_mask[2:-2],
                ),
                adjecent_mask[3:-1],
            ),
            adjecent_mask[4:],
        )
        fake_6e_mask = np.logical_and(
            np.logical_and(adjacent_6e, start_mask[:-5]), end_mask[5:]
        )
        
        # 连续7电子事件，需要开头电子与前一个不相连
        adjacent_7e = np.logical_and(
            np.logical_and(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(
                            adjecent_mask[:-5], adjecent_mask[1:-4]),
                        adjecent_mask[2:-3],),
                    adjecent_mask[3:-2],),
                adjecent_mask[4:-1],)
            ,adjecent_mask[5:],
        )
        fake_7e_mask = np.logical_and(
            np.logical_and(adjacent_7e, start_mask[:-6]), end_mask[6:]
        )
        

        fake_4e_event_indices = np.arange(len(fake_4e_mask))[fake_4e_mask][
            :, np.newaxis
        ] + np.arange(4)
        fake_4e_event = np.take(
            self.delayed_electron[:-3], fake_4e_event_indices, axis=0, mode="clip"
        )

        fake_5e_event_indices = np.arange(len(fake_5e_mask))[fake_5e_mask][
            :, np.newaxis
        ] + np.arange(5)
        fake_5e_event = np.take(
            self.delayed_electron[:-4], fake_5e_event_indices, axis=0, mode="clip"
        )

        fake_6e_event_indices = np.arange(len(fake_6e_mask))[fake_6e_mask][
            :, np.newaxis
        ] + np.arange(6)
        fake_6e_event = np.take(
            self.delayed_electron[:-5], fake_6e_event_indices, axis=0, mode="clip"
        )
        
        fake_7e_event_indices = np.arange(len(fake_7e_mask))[fake_7e_mask][
            :, np.newaxis
        ] + np.arange(7)
        fake_7e_event = np.take(
            self.delayed_electron[:-6], fake_7e_event_indices, axis=0, mode="clip"
        )

        self.pile_up_e_num.append(4)
        self.pile_up_result.append(fake_4e_event)
        self.pile_up_indices.append(fake_4e_event_indices)

        self.pile_up_e_num.append(5)
        self.pile_up_result.append(fake_5e_event)
        self.pile_up_indices.append(fake_5e_event_indices)

        self.pile_up_e_num.append(6)
        self.pile_up_result.append(fake_6e_event)
        self.pile_up_indices.append(fake_6e_event_indices)
        
        self.pile_up_e_num.append(7)
        self.pile_up_result.append(fake_7e_event)
        self.pile_up_indices.append(fake_7e_event_indices)

    def generate_DE_pattern(self):
        """模拟事件根据光模拟结果在顶部的预期通道响应"""
        xd_list = []
        yd_list = []
        e_t_list = []
        # 对每一个电子数量的时间分别进行，将每个事件的n个xd,yd拿出来reshape
        for i in range(len(self.pile_up_e_num)):
            fake_event = self.pile_up_result[i]
            xd = fake_event["xd"].reshape(-1)
            yd = fake_event["yd"].reshape(-1)
            e_t = fake_event["e_time"].reshape(-1)
            xd_list.append(xd)
            yd_list.append(yd)
            e_t_list.append(e_t)
        xd = np.concatenate(xd_list)
        yd = np.concatenate(yd_list)
        xi = np.stack((xd, yd), axis=1)

        # 定义长度与xi相同的数组数据类型，用于存储一个通道所有事件的响应
        array_dtype = np.dtype((np.float64, (len(xi),)))

        results = []
        # 创建进程池
        with multiprocessing.Pool(processes=10) as pool:
            # 提交任务并获取结果
            for pmt_id in range(128):  # 仅对顶部pmt结果进行模拟
                result = pool.apply_async(
                    calculate_pattern,
                    args=(
                        pmt_id,
                        self.LCE_xs,
                        self.LCE_ys,
                        self.LCE_value_true,
                        xi,
                    ),
                )
                results.append(result)
            pool.close()
            pool.join()
            print("Sub-process(es) done.")
            # 等待所有任务完成并获取结果
            results = [result.get() for result in results]
        results = np.array(results)
        results = results.T

        # 将所有结果按照事件大小进行切割
        event_num = np.array([len(i) for i in self.pile_up_result])  # 各电子数事件数量
        event_e_num = event_num * np.array(
            self.pile_up_e_num
        )  # 各电子数事件包含的电子数数量
        indices = np.cumsum(event_e_num)
        indices = np.insert(indices, 0, 0)

        DE_pile_up_pattern_original = [
            results[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        self.pile_up_light_pattern = []
        self.pile_up_pe_pattern = []
        self.pile_up_area_pattern = []
        self.pile_up_pe_by_area = []
        self.pile_up_pe_info = []

        for i in range(len(self.pile_up_e_num)):
            # 对于不同大小的电子数的事件，模拟各自对应的响应

            # 放大SE响应的随机效应
            pattern = DE_pile_up_pattern_original[i]
            gamma = np.random.normal(1, 0.265, len(pattern))
            alter_pattern = pattern * self.single_electron_gain * gamma[:, np.newaxis]
            alter_pattern[alter_pattern < 0] = 0  # 随机效应导致小于0期望值的修正
            alter_pe_pattern = np.random.poisson(alter_pattern)

            fake_light_pattern_sum = np.sum(
                alter_pattern.reshape(-1, self.pile_up_e_num[i], 128), axis=1
            )  # 每N个电子为一组
            fake_pe_pattern = np.sum(
                alter_pe_pattern.reshape(-1, self.pile_up_e_num[i], 128), axis=1
            )  # 每N个电子为一组

            # 以下pe响应模拟处理部分后期需要修改为以fake_pe_pattern为输入的函数
            channel_indices_list = []
            event_id_list = []
            pe_area_list = []
            pile_up_area_pattern_list = []
            pe_time_list = []
            area_mask = []

            for j in tqdm(range(len(fake_pe_pattern))):
                # 针对每个事件，输入pe的event_id,area,channel信息,输出pe响应随机后的patttern
                channel_indices = np.repeat(np.arange(128), fake_pe_pattern[j])
                pe_area = np.random.normal(
                    self.pe_gain, self.pe_gain_std, size=len(channel_indices)
                )
                pe_area[pe_area < 0] = 0
                area_pattern = np.bincount(
                    channel_indices, weights=pe_area, minlength=128
                )

                # 判断事件面积是否在设定区间内
                if (np.sum(area_pattern) > self.sim_low_limit) and (
                    np.sum(area_pattern) < self.sim_up_limit
                ):
                    pile_up_area_pattern_list.append(area_pattern)
                    channel_indices_list.append(channel_indices)
                    event_id_list.append(np.ones(len(channel_indices)) * j)
                    pe_area_list.append(pe_area)
                    area_mask.append(True)
                else:
                    area_mask.append(False)

            # 针对每个电子，输出pe的对应电子时间信息
            for electron_id in tqdm(range(len(alter_pe_pattern))):
                channel_indices = np.repeat(
                    np.arange(128), alter_pe_pattern[electron_id]
                )
                pe_time_list.append(
                    np.ones(len(channel_indices)) * e_t_list[i][electron_id]
                )

            # 所有pe的对应通道
            channel_indices_array = np.concatenate(channel_indices_list)
            # 所有pe的对应event_id（只针对当前电子数事件）
            event_id_array = np.concatenate(event_id_list)
            # 所有pe的实际面积
            pe_area_array = np.concatenate(pe_area_list)
            # 所有的pe的电子时间
            pe_e_time = np.concatenate(pe_time_list)

            # 所有事件的各通道面积分布
            pile_up_area_pattern = np.array(pile_up_area_pattern_list)

            # 对于pe响应之前的pattern，很具pe响应以及面积筛选之后的mask
            area_mask = np.array(area_mask)

            pe_data = np.zeros(len(channel_indices_array), dtype=self.pe_dtype)
            pe_data["event_id"] = event_id_array
            pe_data["area"] = pe_area_array
            pe_data["channel"] = channel_indices_array
            # pe_data['e_time'] = pe_e_time
            # fake_area_pattern = np.random.normal(
            #     self.pe_gain * fake_pe_pattern, self.pe_gain_std * (fake_pe_pattern**0.5))

            self.pile_up_light_pattern.append(fake_light_pattern_sum[area_mask])
            self.pile_up_pe_pattern.append(fake_pe_pattern[area_mask])
            self.pile_up_area_pattern.append(pile_up_area_pattern)
            self.pile_up_pe_by_area.append(pile_up_area_pattern / self.pe_gain)
            self.pile_up_pe_info.append(pe_data)

    def generate_DE_pattern_linear(self):
        """模拟事件根据光模拟结果在顶部的预期通道响应"""
        xd_list = []
        yd_list = []
        e_t_list = []
        # 对每一个电子数量的事件分别进行，将每个事件的n个xd,yd拿出来reshape
        for i in range(len(self.pile_up_e_num)):
            fake_event = self.pile_up_result[i]
            xd = fake_event["xd"].reshape(-1)
            yd = fake_event["yd"].reshape(-1)
            e_t = fake_event["e_time"].reshape(-1)
            xd_list.append(xd)
            yd_list.append(yd)
            e_t_list.append(e_t)
        xd = np.concatenate(xd_list)
        yd = np.concatenate(yd_list)
        xi = np.stack((xd, yd), axis=1)

        # 定义长度与xi相同的数组数据类型，用于存储一个通道所有事件的响应
        array_dtype = np.dtype((np.float64, (len(xi),)))

        results = []
        for pmt_id in range(128):  # 仅对顶部pmt结果进行模拟
            result = calculate_pattern(
                pmt_id,
                self.LCE_xs,
                self.LCE_ys,
                self.LCE_value_true,
                xi,
            )
            results.append(result)
            # 等待所有任务完成并获取结果
        results = np.array(results)
        results = results.T

        # 将所有结果按照事件大小进行切割
        event_num = np.array([len(i) for i in self.pile_up_result])  # 各电子数事件数量
        event_e_num = event_num * np.array(
            self.pile_up_e_num
        )  # 各电子数事件包含的电子数数量
        indices = np.cumsum(event_e_num)
        indices = np.insert(indices, 0, 0)

        DE_pile_up_pattern_original = [
            results[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        self.pile_up_light_pattern = []
        self.pile_up_pe_pattern = []
        self.pile_up_area_pattern = []
        self.pile_up_pe_by_area = []
        self.pile_up_pe_info = []
        self.area_select_mask = []

        for i in range(len(self.pile_up_e_num)):
            # 对于不同大小的电子数的事件，模拟各自对应的响应

            # 放大SE响应的随机效应
            pattern = DE_pile_up_pattern_original[i]
            gamma = np.random.normal(1, 0.265, len(pattern))
            alter_pattern = pattern * self.single_electron_gain * gamma[:, np.newaxis]
            alter_pattern[alter_pattern < 0] = 0  # 随机效应导致小于0期望值的修正
            alter_pe_pattern = np.random.poisson(alter_pattern)

            fake_light_pattern_sum = np.sum(
                alter_pattern.reshape(-1, self.pile_up_e_num[i], 128), axis=1
            )  # 每N个电子为一组
            fake_pe_pattern = np.sum(
                alter_pe_pattern.reshape(-1, self.pile_up_e_num[i], 128), axis=1
            )  # 每N个电子为一组

            # 以下pe响应模拟处理部分后期需要修改为以fake_pe_pattern为输入的函数
            channel_indices_list = []
            event_id_list = []
            pe_area_list = []
            pile_up_area_pattern_list = []
            # 存储pe对应电子时间
            pe_time_list = []
            area_mask = []
            event_id_region = 0

            for j in tqdm(range(len(fake_pe_pattern))):
                # 针对每个事件，输入pe的event_id,area,channel信息,输出pe响应随机后的patttern
                channel_indices = np.repeat(np.arange(128), fake_pe_pattern[j])
                pe_area = np.random.normal(
                    self.pe_gain, self.pe_gain_std, size=len(channel_indices)
                )
                pe_area[pe_area < 0] = 0
                area_pattern = np.bincount(
                    channel_indices, weights=pe_area, minlength=128
                )

                # print("Area: ", np.sum(area_pattern))

                # 判断事件面积是否在设定区间内
                if (np.sum(area_pattern) > self.sim_low_limit * self.pe_gain) and (
                    np.sum(area_pattern) < self.sim_up_limit * self.pe_gain
                ):
                    # 获取事件的N个电子时间
                    e_time_event = self.pile_up_result[i][j]["e_time"]
                    # 获取对应每个电子的电子数
                    seperate_pattern = alter_pe_pattern[
                        j * self.pile_up_e_num[i] : (j + 1) * self.pile_up_e_num[i]
                    ]
                    num_electron = np.sum(seperate_pattern, axis=1)
                    # 生成时间序列
                    e_time = np.repeat(e_time_event, num_electron)
                    pe_time_list.append(e_time)
                    # print("Area: ", np.sum(area_pattern))
                    pile_up_area_pattern_list.append(area_pattern)
                    channel_indices_list.append(channel_indices)
                    event_id_list.append(
                        np.ones(len(channel_indices)) * event_id_region
                    )
                    pe_area_list.append(pe_area)
                    area_mask.append(True)
                    event_id_region += 1
                else:
                    area_mask.append(False)

            # # 针对每个电子，输出pe的对应电子时间信息
            # for electron_id in tqdm(range(len(alter_pe_pattern))):
            #     channel_indices = np.repeat(
            #         np.arange(128), alter_pe_pattern[electron_id])
            #     pe_time_list.append(
            #         np.ones(len(channel_indices))*e_t_list[i][electron_id])

            # 所有pe的对应通道
            # print("!!!!!!", channel_indices_list)
            channel_indices_array = np.concatenate(channel_indices_list)
            # 所有pe的对应event_id（只针对当前电子数事件）
            event_id_array = np.concatenate(event_id_list)
            # 所有pe的实际面积
            pe_area_array = np.concatenate(pe_area_list)
            # 所有的pe的电子时间
            pe_e_time = np.concatenate(pe_time_list)

            # 所有事件的各通道面积分布
            pile_up_area_pattern = np.array(pile_up_area_pattern_list)

            # 对于pe响应之前的pattern，很具pe响应以及面积筛选之后的mask
            area_mask = np.array(area_mask)

            pe_data = np.zeros(len(channel_indices_array), dtype=self.pe_dtype)
            pe_data["event_id"] = event_id_array
            pe_data["area"] = pe_area_array
            pe_data["channel"] = channel_indices_array
            pe_data["e_time"] = pe_e_time
            # fake_area_pattern = np.random.normal(
            #     self.pe_gain * fake_pe_pattern, self.pe_gain_std * (fake_pe_pattern**0.5))
            self.pile_up_result[i] = self.pile_up_result[i][area_mask]
            self.pile_up_light_pattern.append(fake_light_pattern_sum[area_mask])
            self.pile_up_pe_pattern.append(fake_pe_pattern[area_mask])
            self.pile_up_area_pattern.append(pile_up_area_pattern)
            self.pile_up_pe_by_area.append(pile_up_area_pattern / self.pe_gain)
            self.pile_up_pe_info.append(pe_data)
            self.area_select_mask.append(area_mask)

    def generate_DE_recon_pattern(self):
        """模拟事件根据光模拟结果在顶部的预期通道响应,针对重建得到的位置"""
        xd_list = []
        yd_list = []
        for i in range(len(self.pile_up_e_num)):
            fake_pos = self.pile_up_pos_recon[i]
            xd = fake_pos["xd"].reshape(-1)
            yd = fake_pos["yd"].reshape(-1)
            xd_list.append(xd)
            yd_list.append(yd)
        xd = np.concatenate(xd_list)
        yd = np.concatenate(yd_list)
        xi = np.stack((xd, yd), axis=1)

        # 定义长度与xi相同的数组数据类型，用于存储一个通道所有事件的响应
        array_dtype = np.dtype((np.float64, (len(xi),)))

        # results = []
        # # 创建进程池
        # with multiprocessing.Pool(processes=10) as pool:
        #     # 提交任务并获取结果
        #     for pmt_id in range(128):  # 对pmt结果进行模拟
        #         result = pool.apply_async(calculate_pattern, args=(
        #             pmt_id, self.LCE_xs, self.LCE_ys, self.LCE_value, xi,))
        #         results.append(result)
        #     # 等待所有任务完成并获取结果
        #     results = [result.get() for result in results]
        # results = np.array(results)
        # results = results.T

        results = []
        # 创建进程池
        for pmt_id in range(128):  # 仅对顶部pmt结果进行模拟
            result = calculate_pattern(
                pmt_id,
                self.LCE_xs,
                self.LCE_ys,
                self.LCE_value_true,
                xi,
            )
            results.append(result)
            # 等待所有任务完成并获取结果
        results = np.array(results)
        results = results.T

        # 将所有结果按照事件大小进行切割
        event_num = np.array([len(i) for i in self.pile_up_result])
        indices = np.cumsum(event_num)
        indices = np.insert(indices, 0, 0)

        self.pile_up_recons_pattern = [
            results[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        self.pile_up_recons_light_pattern = []
        for i in range(len(self.pile_up_e_num)):
            de_recon_pattern = (
                np.sum(self.pile_up_pe_by_area[i], axis=1)[:, np.newaxis]
                / (self.single_electron_gain * self.average_efficiency)
                * self.pile_up_recons_pattern[i]
                * self.single_electron_gain
            )
            self.pile_up_recons_light_pattern.append(de_recon_pattern)

    def generate_CEVNS_pattern(self):
        """模拟事件根据光模拟结果在顶部的预期通道响应,针对CEVNS事件"""
        xd_list = []
        yd_list = []
        for i in range(len(self.cevns_e_num)):
            fake_pos = self.cevns_points[i]["pos_original"]
            xd = fake_pos["xd"].reshape(-1)
            yd = fake_pos["yd"].reshape(-1)
            xd_list.append(xd)
            yd_list.append(yd)
        xd = np.concatenate(xd_list)
        yd = np.concatenate(yd_list)
        xi = np.stack((xd, yd), axis=1)

        # 定义长度与xi相同的数组数据类型，用于存储一个通道所有事件的响应
        array_dtype = np.dtype((np.float64, (len(xi),)))

        # results = []
        # # 创建进程池
        # with multiprocessing.Pool(processes=10) as pool:
        #     # 提交任务并获取结果
        #     for pmt_id in range(128):
        #         result = pool.apply_async(calculate_pattern, args=(
        #             pmt_id, self.LCE_xs, self.LCE_ys, self.LCE_value_true, xi,))
        #         results.append(result)
        #     # 等待所有任务完成并获取结果
        #     results = [result.get() for result in results]
        # results = np.array(results)
        # results = results.T

        results = []
        # 创建进程池
        for pmt_id in tqdm(range(128)):  
            result = calculate_pattern(
                pmt_id,
                self.LCE_xs,
                self.LCE_ys,
                self.LCE_value_true,
                xi,
            )
            results.append(result)
            # 等待所有任务完成并获取结果
        results = np.array(results)
        results = results.T

        # 将所有结果按照事件大小进行切割
        event_num = np.array([len(i) for i in self.cevns_points])
        indices = np.cumsum(event_num)
        indices = np.insert(indices, 0, 0)

        self.CEVNS_pattern = [
            results[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        # self.CEVNS_light_pattern = []
        # self.CEVNS_pe_pattern = []
        # self.CEVNS_area_pattern = []
        # self.CEVNS_pe_by_area = []

        self.cevns_pe_info = []

        for i in range(len(self.cevns_e_num)):
            # pattern = self.CEVNS_pattern[i]
            # 将pattern分为不同的电子进行处理
            pattern = np.repeat(self.CEVNS_pattern[i], self.cevns_e_num[i], axis=0)
            # 人工增加pe分布系数范围
            gamma = np.random.normal(1, 0.265, len(pattern))
            CEVNS_light_pattern = (
                pattern * self.single_electron_gain * gamma[:, np.newaxis]
            )

            CEVNS_light_pattern[CEVNS_light_pattern < 0] = 0

            CEVNS_pe_pattern = np.random.poisson(CEVNS_light_pattern)

            CEVNS_light_pattern_reshaped = CEVNS_light_pattern.reshape(
                -1, self.cevns_e_num[i], CEVNS_light_pattern.shape[1]
            )
            CEVNS_pe_pattern_reshaped = CEVNS_pe_pattern.reshape(
                -1, self.cevns_e_num[i], CEVNS_pe_pattern.shape[1]
            )
            CEVNS_light_pattern_sum = np.sum(CEVNS_light_pattern_reshaped, axis=1)
            CEVNS_pe_pattern_sum = np.sum(CEVNS_pe_pattern_reshaped, axis=1)

            channel_indices_list = []
            event_id_list = []
            pe_area_list = []
            CEVNS_area_pattern_list = []
            pe_e_id = []

            for j in tqdm(range(len(CEVNS_pe_pattern_sum))):
                # 针对每个事件，输pe的event_id,area,channel信息,输出pe响应随机后的patttern
                channel_indices = np.repeat(
                    np.arange(128), CEVNS_pe_pattern_sum[j]
                )  # 每个pe对应的channel序号
                # 每个pe对应的pattern序号
                event_id_list.append(np.ones(len(channel_indices)) * j)
                pe_area = np.random.normal(
                    self.pe_gain, self.pe_gain_std, size=len(channel_indices)
                )
                pe_area_list.append(pe_area)
                area_pattern = np.bincount(
                    channel_indices, weights=pe_area, minlength=128
                )  # 计算出128个通道每个通道实际响应面积
                area_pattern[area_pattern < 0] = 0
                CEVNS_area_pattern_list.append(area_pattern)
                channel_indices_list.append(channel_indices)

                # 针对每个电子，输出pe的对应电子序号
            for electron_id in tqdm(range(len(CEVNS_pe_pattern))):
                channel_indices = np.repeat(
                    np.arange(128), CEVNS_pe_pattern[electron_id]
                )
                pe_e_id.append(np.ones(len(channel_indices)) * electron_id)

            # 所有pe的对应通道
            channel_indices_array = np.concatenate(channel_indices_list)
            # 所有pe的对应event_id（只针对当前电子数事件）
            event_id_array = np.concatenate(event_id_list)
            # 所有pe的实际面积
            pe_area_array = np.concatenate(pe_area_list)
            # 所有pe的电子id
            pe_e_id = np.concatenate(pe_e_id)
            # 所有事件的各通道面积分布
            CEVNS_area_pattern = np.array(CEVNS_area_pattern_list)

            pe_data = np.zeros(len(channel_indices_array), dtype=self.pe_dtype)
            pe_data["event_id"] = event_id_array
            pe_data["area"] = pe_area_array
            pe_data["channel"] = channel_indices_array
            pe_data["e_id"] = pe_e_id
            self.cevns_points[i]["light_pattern"] = CEVNS_light_pattern_sum
            self.cevns_points[i]["pe_pattern"] = CEVNS_pe_pattern_sum
            self.cevns_points[i]["area_pattern"] = CEVNS_area_pattern
            self.cevns_points[i]["pe_by_area"] = CEVNS_area_pattern / self.pe_gain

            self.cevns_pe_info.append(pe_data)

    def generate_CEVNS_recon_pattern(self):
        """模拟事件根据光模拟结果在顶部的预期通道响应,针对CEVNS事件"""
        xd_list = []
        yd_list = []
        for i in range(len(self.cevns_e_num)):
            fake_pos = self.cevns_points[i]["pos_recon"]
            xd = fake_pos["xd"].reshape(-1)
            yd = fake_pos["yd"].reshape(-1)
            xd_list.append(xd)
            yd_list.append(yd)
        xd = np.concatenate(xd_list)
        yd = np.concatenate(yd_list)
        xi = np.stack((xd, yd), axis=1)

        # 定义长度与xi相同的数组数据类型，用于存储一个通道所有事件的响应
        array_dtype = np.dtype((np.float64, (len(xi),)))

        results = []
        # 创建进程池
        for pmt_id in range(128):  # 仅对顶部pmt结果进行模拟
            result = calculate_pattern(
                pmt_id,
                self.LCE_xs,
                self.LCE_ys,
                self.LCE_value_true,
                xi,
            )
            results.append(result)
            # 等待所有任务完成并获取结果
        results = np.array(results)
        results = results.T

        # 将所有结果按照事件大小进行切割
        event_num = np.array([len(i) for i in self.cevns_points])
        indices = np.cumsum(event_num)
        indices = np.insert(indices, 0, 0)

        recons_pattern = [
            results[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
        ]

        for i in range(len(self.cevns_e_num)):
            CEVNS_recon_pattern = (
                np.sum(self.cevns_points[i]["pe_by_area"], axis=1)[:, np.newaxis]
                / (self.single_electron_gain * self.average_efficiency)
                * recons_pattern[i]
                * self.single_electron_gain
            )
            self.cevns_points[i]["recons_light_pattern"] = CEVNS_recon_pattern

    def DE_position_recon(self):
        # 定义结构化数组的dtype
        dt = np.dtype([("xd", "<f8"), ("yd", "<f8")])

        self.pile_up_pos_recon = []
        for i in range(len(self.pile_up_e_num)):
            print("Start pos recon for event size: ", self.pile_up_e_num[i])
            if len(self.pile_up_result[i]) >= 0:
                pe_by_area = self.pile_up_pe_by_area[i]
                pos_pre = pos.position_construction(
                    pe_by_area[:, :64], self.position_reconstruction_model
                )
            else:
                pos_pre = np.zeros((0, 2), dtype="float32")

            fake_position_structed = np.zeros(len(pos_pre), dtype=dt)
            fake_position_structed["xd"], fake_position_structed["yd"] = (
                pos_pre[:, 0],
                pos_pre[:, 1],
            )
            self.pile_up_pos_recon.append(fake_position_structed)

    def CEVNS_position_recon(self):
        # 定义结构化数组的dtype
        dt = np.dtype([("xd", "<f8"), ("yd", "<f8")])

        for i in range(len(self.cevns_e_num)):
            print("Start pos recon for event size: ", self.cevns_e_num[i])
            if len(self.cevns_points[i]) >= 0:
                pe_by_area = self.cevns_points[i]["pe_by_area"]
                pos_pre = pos.position_construction(
                    pe_by_area[:, :64], self.position_reconstruction_model
                )
            else:
                pos_pre = np.zeros((0, 2), dtype="float32")

            CEVNS_position_structed = np.zeros(len(pos_pre), dtype=dt)
            CEVNS_position_structed["xd"], CEVNS_position_structed["yd"] = (
                pos_pre[:, 0],
                pos_pre[:, 1],
            )
            self.cevns_points[i]["pos_recon"] = CEVNS_position_structed

    def DE_sp_cor(self):
        self.de_delta = []
        self.pile_up_sp_cor = []
        self.pile_up_sp_cor_new = []
        for i in range(len(self.pile_up_e_num)):
            # print(i)
            fake_event = self.pile_up_result[i]
            fake_time = fake_event[:, 0]["e_time"]
            fake_e = (
                np.sum(self.pile_up_pe_by_area[i], axis=1) / self.single_electron_pe
            )
            last_muon_id = np.searchsorted(self.event_time, fake_time, side="right") - 1
            fake_space_time_cor, delta_x_list, delta_y_list, delta_t_list = (
                space_time_cor_new(
                    fake_time,
                    fake_e,
                    self.dense_muon_points,
                    last_muon_id,
                    self.pile_up_pos_recon[i],
                    3,
                    20,
                )
            )
            # fake_space_time_cor_new, delta_x_list, delta_y_list, delta_t_list = space_time_cor_new(
            #     fake_time, fake_e, self.dense_muon_points, last_muon_id, self.pile_up_pos_recon[i], 1.1, 20)
            self.pile_up_sp_cor.append(fake_space_time_cor)
            # self.pile_up_sp_cor_new.append(fake_space_time_cor_new)
            self.de_delta.append([delta_x_list, delta_y_list, delta_t_list])

    def CEVNS_sp_cor(self):
        self.cevns_delta = []
        for i in range(len(self.cevns_e_num)):
            # print(i)
            cevns_time = self.cevns_points[i]["t"]
            cevns_e = (
                np.sum(self.cevns_points[i]["pe_by_area"], axis=1)
                / self.single_electron_pe
            )
            last_muon_id = (
                np.searchsorted(self.event_time, cevns_time, side="right") - 1
            )
            cevns_space_time_cor, delta_x_list, delta_y_list, delta_t_list = (
                space_time_cor_new(
                    cevns_time,
                    cevns_e,
                    self.dense_muon_points,
                    last_muon_id,
                    self.cevns_points[i]["pos_recon"],
                    3,
                    20,
                )
            )
            # cevns_space_time_cor_new, delta_x_list, delta_y_list, delta_t_list = space_time_cor_new(
            #     cevns_time, cevns_e, self.dense_muon_points, last_muon_id, self.cevns_points[i]['pos_recon'], 1.1, 20)
            self.cevns_points[i]["st_cor"] = cevns_space_time_cor
            # self.cevns_points[i]['st_cor_new'] = cevns_space_time_cor_new
            self.cevns_delta.append([delta_x_list, delta_y_list, delta_t_list])


def calculate_pattern(pmt_id, LCE_xs, LCE_ys, LCE_value, xi):
    # print("Interpn for channel: ", pmt_id)
    return interpn(
        (LCE_xs, LCE_ys), LCE_value[pmt_id], xi, bounds_error=False, fill_value=np.nan
    )


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


def delayed_electrons_sorted(dense_muon_points, dead_time_start=0.001, radius=20):
    """radius 单位为mm, 创建对应所有延迟电子的信息"""
    # 定义电子的数据类型
    electron_types = [
        ("eventId", "<i4"),
        ("xd", "<f8"),
        ("yd", "<f8"),
        ("muon_time", "<f8"),
        ("e_time", "<f8"),
    ]
    num_electrons = np.sum(dense_muon_points["num_e_delayed"])
    delayed_electrons = np.zeros(num_electrons, dtype=electron_types)
    # 将eventId,x,y,t赋值给delayed_electrons，注意这里x,y,t未添加偏差
    delayed_electrons["eventId"] = np.repeat(
        dense_muon_points["eventId"], dense_muon_points["num_e_delayed"]
    )
    delayed_electrons["xd"] = np.repeat(
        dense_muon_points["xd"], dense_muon_points["num_e_delayed"]
    )
    delayed_electrons["yd"] = np.repeat(
        dense_muon_points["yd"], dense_muon_points["num_e_delayed"]
    )
    delayed_electrons["muon_time"] = np.repeat(
        dense_muon_points["muon_time"], dense_muon_points["num_e_delayed"]
    )

    # 计算所有涉及到的电子的随机delay time
    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag) * r) ** (1.0 / g)

    delay_time = rndm(dead_time_start, 2, g=-0.1, size=num_electrons)

    # 计算所有涉及到的电子的随机坐标差异,radius单位为mm
    def generate_gaussian_coordinates(center, cov_matrix, num_points):
        # 生成二维高斯分布的随机样本
        coordinates = np.random.multivariate_normal(center, cov_matrix, num_points)
        return coordinates

    # 协方差矩阵
    cov_matrix = [[radius, 0], [0, radius]]  # 对角矩阵，表示x和y方向上的方差均为radius
    center = [0, 0]
    coordinate = generate_gaussian_coordinates(center, cov_matrix, num_electrons)
    x, y = coordinate[:, 0], coordinate[:, 1]

    # 将随机产生的xy偏置以及delay time 加入原有delayed_electrons
    delayed_electrons["xd"] = delayed_electrons["xd"] + x
    delayed_electrons["yd"] = delayed_electrons["yd"] + y
    delayed_electrons["e_time"] = delayed_electrons["muon_time"] + delay_time

    # 提取每行第二个元素
    e_time = np.copy(delayed_electrons["e_time"])

    # 对数组进行排序并获取排序后的索引
    print("Start sorting DE's time")
    sorted_indices = np.argsort(e_time)
    delayed_electron_sorted = delayed_electrons[sorted_indices]
    print("Finish sorting DE's time")

    # 去除139mm以外的电子，物理意义上来说，这些电子过于靠近容器壁，可能被吸附或者别的效应，且根据光模拟，容易有nan出现
    electron_radius_square = (
        delayed_electron_sorted["xd"] ** 2 + delayed_electron_sorted["yd"] ** 2
    )
    electron_inner_mask = electron_radius_square < 139**2
    delayed_electron_sorted_in = delayed_electron_sorted[electron_inner_mask]
    return delayed_electron_sorted_in


def space_time_cor(
    event_t,
    event_e,
    dense_muon_points,
    last_muon_id,
    fake_position_structed,
    gamma,
    look_ahead,
):
    # 避免单次过大的内存开销，分批次进行计算，每次计算test_num个muon
    test_num = 100000
    run_id_array = np.arange(len(event_t) // test_num + 1)
    space_time_cor_list = []
    print("last_muon_id.shape: ", last_muon_id.shape)
    print("last_muon_id[-10:]: ", last_muon_id[-10:])
    delta_x_list = []
    delta_y_list = []
    delta_t_list = []

    for run_id in run_id_array:
        print(run_id)
        event_id_start = run_id * test_num
        event_id_stop = min(len(event_t), event_id_start + test_num)
        print(event_id_start, event_id_stop)
        expanded_points = np.zeros(
            (event_id_stop - event_id_start, 100 * look_ahead),
            dtype=dense_muon_points.dtype,
        )  # n*100个muon点的关联信息存储
        for i in tqdm(range(event_id_stop - event_id_start)):
            # 进度条显示构造包含所有前look_ahead个事件的向量化数组的速度
            muon_id = last_muon_id[i + event_id_start]
            muon_start = max(0, muon_id - 19)
            muon_previous = muon_id
            point_start = (muon_start) * 100
            point_stop = (muon_previous + 1) * 100
            # print(point_start)
            # print(point_stop)
            try:
                expanded_points[i, : (point_stop - point_start)] = dense_muon_points[
                    point_start:point_stop
                ]
            except ValueError as e:
                print("event_id_start: ", event_id_start)
                print("i:", i)
                print("point_start: ", point_start)
                print("point_stop: ", point_stop)
                print("muon_id: ", muon_id)
                print("expanded_points.shape: ", expanded_points.shape)
                print("dense_muon_points.shape: ", dense_muon_points.shape)
                print(
                    "expanded_points[i,:(point_stop - point_start)].shape: ",
                    expanded_points[i, : (point_stop - point_start)].shape,
                )
                print(
                    "dense_muon_points[point_start:point_stop].shape: ",
                    dense_muon_points[point_start:point_stop].shape,
                )

        radius = 20
        # 伪事件的x
        test_fake_x = fake_position_structed["xd"][event_id_start:event_id_stop]
        # 伪事件的y
        test_fake_y = fake_position_structed["yd"][event_id_start:event_id_stop]
        test_fake_t = event_t[event_id_start:event_id_stop]  # 伪事件的t

        # 关联点的x
        point_x = expanded_points["xd"][: event_id_stop - event_id_start]
        # 关联点的y
        point_y = expanded_points["yd"][: event_id_stop - event_id_start]
        # 关联点的t
        point_t = expanded_points["muon_time"][: event_id_stop - event_id_start]
        # 关联点的延迟电子数量
        point_de_num = expanded_points["num_e_delayed"][
            : event_id_stop - event_id_start
        ]

        delta_x = test_fake_x[:, np.newaxis] - point_x
        delta_y = test_fake_y[:, np.newaxis] - point_y
        delta_t = test_fake_t[:, np.newaxis] - point_t
        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)
        delta_t_list.append(delta_t)

        # 计算伪事件的时空关联
        distance_sqrt = delta_x**2 + delta_y**2
        exponent = -0.5 * distance_sqrt / (radius**2)
        xy_prob_density = (
            np.log((1 / (2 * np.pi * (radius**2)))) + np.log(point_de_num) + exponent
        )  # 水平方向偏差的概率的log
        delay_prob_log = -gamma * np.log(delta_t)
        muon_point_missing = point_t == 0
        # fake_prob = np.exp((delay_prob_log + xy_prob_density)) * muon_point_mask[0].astype('uint32')
        fake_prob = xy_prob_density + delay_prob_log
        fake_prob_no_inf = fake_prob
        fake_prob_no_inf[muon_point_missing] = 0
        fake_prob_no_inf = np.exp(fake_prob_no_inf)
        fake_prob_no_inf[muon_point_missing] = 0
        space_time_cor = np.sum(fake_prob_no_inf, axis=1)
        space_time_cor_list.append(space_time_cor)

    # 将list整合为一个完整的array
    fake_space_time_cor = np.zeros(len(event_t))
    for run_id in run_id_array:
        event_id_start = run_id * test_num
        event_id_stop = min(len(event_t), event_id_start + test_num)
        fake_space_time_cor[event_id_start:event_id_stop] = space_time_cor_list[run_id]

    return fake_space_time_cor


def space_time_cor_new(
    event_t,
    event_e,
    dense_muon_points,
    last_muon_id,
    fake_position_structed,
    gamma,
    radius,
    look_ahead=20,
):
    # 避免单次过大的内存开销，分批次进行计算，每次计算test_num个muon
    test_num = 100000
    run_id_array = np.arange(len(event_t) // test_num + 1)
    space_time_cor_list = []
    print("last_muon_id.shape: ", last_muon_id.shape)
    print("last_muon_id[-10:]: ", last_muon_id[-10:])
    delta_x_list = []
    delta_y_list = []
    delta_t_list = []

    for run_id in run_id_array:
        print(run_id)
        event_id_start = run_id * test_num
        event_id_stop = min(len(event_t), event_id_start + test_num)
        print(event_id_start, event_id_stop)
        expanded_points = np.zeros(
            (event_id_stop - event_id_start, 100 * look_ahead),
            dtype=dense_muon_points.dtype,
        )  # n*100个muon点的关联信息存储
        for i in tqdm(range(event_id_stop - event_id_start)):
            # 进度条显示构造包含所有前look_ahead个事件的向量化数组的速度
            muon_id = last_muon_id[i + event_id_start]
            muon_start = max(0, muon_id - 19)
            muon_previous = muon_id
            point_start = (muon_start) * 100
            point_stop = (muon_previous + 1) * 100
            # print(point_start)
            # print(point_stop)
            try:
                expanded_points[i, : (point_stop - point_start)] = dense_muon_points[
                    point_start:point_stop
                ]
            except ValueError as e:
                print("event_id_start: ", event_id_start)
                print("i:", i)
                print("point_start: ", point_start)
                print("point_stop: ", point_stop)
                print("muon_id: ", muon_id)
                print("expanded_points.shape: ", expanded_points.shape)
                print("dense_muon_points.shape: ", dense_muon_points.shape)
                print(
                    "expanded_points[i,:(point_stop - point_start)].shape: ",
                    expanded_points[i, : (point_stop - point_start)].shape,
                )
                print(
                    "dense_muon_points[point_start:point_stop].shape: ",
                    dense_muon_points[point_start:point_stop].shape,
                )

        # 伪事件的x
        test_fake_x = fake_position_structed["xd"][event_id_start:event_id_stop]
        # 伪事件的y
        test_fake_y = fake_position_structed["yd"][event_id_start:event_id_stop]
        test_fake_t = event_t[event_id_start:event_id_stop]  # 伪事件的t
        test_fake_e = event_e[event_id_start:event_id_stop]  # 伪事件的电子数

        # 关联点的x
        point_x = expanded_points["xd"][: event_id_stop - event_id_start]
        # 关联点的y
        point_y = expanded_points["yd"][: event_id_stop - event_id_start]
        # 关联点的t
        point_t = expanded_points["muon_time"][: event_id_stop - event_id_start]
        # 关联点的延迟电子数量
        point_de_num = expanded_points["num_e_delayed"][
            : event_id_stop - event_id_start
        ]

        delta_x = test_fake_x[:, np.newaxis] - point_x
        delta_y = test_fake_y[:, np.newaxis] - point_y
        delta_t = test_fake_t[:, np.newaxis] - point_t
        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)
        delta_t_list.append(delta_t)
        # print("平均时间延迟： ", np.mean(delta_t))

        # start_time = time.time()  # 开始计时
        # 计算伪事件的时空关联
        distance_sqrt = delta_x**2 + delta_y**2

        exponent = -0.5 * distance_sqrt / (radius**2)
        xy_prob = (1 / (2 * np.pi * (radius**2))) * (
            np.exp((exponent))
        )  # 水平方向偏差的概率的log
        # print("平均xy概率： ", np.mean(xy_prob))
        time_prob = delta_t**-gamma
        # print("平均时间概率： ", np.mean(time_prob))

        # end_time = time.time()  # 结束计时
        # elapsed_time = end_time - start_time  # 计算经过的时间
        # print(f'The code ran in {elapsed_time} seconds.')  # 输出运行时间

        muon_point_missing = point_t == 0
        # fake_prob = np.exp((delay_prob_log + xy_prob_density)) * muon_point_mask[0].astype('uint32')
        fake_prob = xy_prob * time_prob * point_de_num

        # end_time = time.time()  # 结束计时
        # elapsed_time = end_time - start_time  # 计算经过的时间
        # print(f'The code ran in {elapsed_time} seconds.')  # 输出运行时间

        fake_prob[muon_point_missing] = 0
        # space_time_cor = np.sum(fake_prob, axis=1) ** test_fake_e
        space_time_cor = np.sum(fake_prob, axis=1)
        print("平均电子数： ", np.mean(test_fake_e))

        # end_time = time.time()  # 结束计时
        # elapsed_time = end_time - start_time  # 计算经过的时间
        # print(f'The code ran in {elapsed_time} seconds.')  # 输出运行时间

        space_time_cor_list.append(space_time_cor)

    # 将list整合为一个完整的array
    fake_space_time_cor = np.zeros(len(event_t))
    for run_id in run_id_array:
        event_id_start = run_id * test_num
        event_id_stop = min(len(event_t), event_id_start + test_num)
        fake_space_time_cor[event_id_start:event_id_stop] = space_time_cor_list[run_id]

    return fake_space_time_cor, delta_x_list, delta_y_list, delta_t_list
