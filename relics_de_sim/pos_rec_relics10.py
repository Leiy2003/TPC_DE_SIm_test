import torch
from joblib import load,dump
import numpy as np
from relics_de_sim.utils import PEperChannel2input
import json
import os
import copy
import argparse
import pickle
from torch.utils.data import DataLoader, TensorDataset
from relics_de_sim.model import Generator2D
from relics_de_sim.model import DeepResNet
from relics_de_sim.utils import load_generator
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

# 全局变量来存储配置，避免重复加载
_config_cache = None
_config_dir_cache = None

fname = '../LCE_info/topPMTs.txt'
toltal_N = 5


pos_type = np.dtype([('index', np.uint32),
                     ('x', np.float64), ('y', np.float64), ('z', np.float64),
                     ('rotX', np.float64),('rotY', np.float64), ('rotZ', np.float64)])
PMTs = np.loadtxt(fname, dtype=pos_type, delimiter=' ')


PMTs['x'] = PMTs['x']*10
PMTs['y'] = PMTs['y']*10

pmts = 27.8 #PMT 边长1
pmtw = 20.5 #PMT 边长2

x = PMTs['x'] - pmts / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)
y = PMTs['y'] - pmts / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)

xreal = x + (pmts - pmtw) / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)
yreal = y + (pmts - pmtw) / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)

# PMT 顶点1
xreal = x + (pmts - pmtw) / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)
yreal = y + (pmts - pmtw) / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi)

x = PMTs['x'] - pmts / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi)
y = PMTs['y'] - pmts / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi)

xreal2 = x + (pmts - pmtw) / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi)
yreal2 = y + (pmts - pmtw) / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi)

x = PMTs['x'] - pmts / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi/2)
y = PMTs['y'] - pmts / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi/2)

xreal3 = x + (pmts - pmtw) / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi/2)
yreal3 = y + (pmts - pmtw) / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi/2)

x = PMTs['x'] - pmts / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi*3/2)
y = PMTs['y'] - pmts / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi*3/2)

xreal4 = x + (pmts - pmtw) / 2 * np.sqrt(2) * np.cos(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi*3/2)
yreal4 = y + (pmts - pmtw) / 2 * np.sqrt(2) * np.sin(np.pi / 4 + PMTs['rotZ'] / 180 * np.pi+np.pi*3/2)

def load_config(config_path="/home/leiyang/Relics_DE_Sim/config.json"):
    """加载配置文件的辅助函数"""
    global _config_cache, _config_dir_cache
    
    if _config_cache is None:
        with open(config_path, "r") as f:
            _config_cache = json.load(f)
        
        # 获取 config.json 的路径
        config_abs_path = os.path.abspath(config_path)
        # 获取 config.json 所在目录
        _config_dir_cache = os.path.dirname(config_abs_path)
    
    return _config_cache, _config_dir_cache


def get_pixel_points(i_x,i_y):
    
    x_pixel = np.linspace(-85,85,toltal_N+1)
    y_pixel = np.linspace(-85,85,toltal_N+1)
    # print(len(x_pixel))
    a =np.array( (x_pixel[i_x],y_pixel[i_y]))
    b =np.array( (x_pixel[i_x+1],y_pixel[i_y]))
    c =np.array( (x_pixel[i_x+1],y_pixel[i_y+1]))
    d =np.array( (x_pixel[i_x],y_pixel[i_y+1]))
    return np.array([a, b , c, d])

def get_PMT_points(PMT_num):
    a = np.array((xreal[PMT_num],yreal[PMT_num]))
    b = np.array((xreal3[PMT_num],yreal3[PMT_num]))
    c = np.array((xreal2[PMT_num],yreal2[PMT_num]))
    d = np.array((xreal4[PMT_num],yreal4[PMT_num]))
    return np.array([a, b , c, d])

def data_trans(pe_per_channel):
    """
    transforming PE per channel signals to inputs for NN
    
    Parameters:
    -----------
    pe_per_channel : array-like
        输入数据
    config_path : str
        配置文件路径，默认为 "./config.json"
    """
    # 加载配置（使用缓存机制）
    # config, config_dir = load_config(config_path)
    
    # 切换到配置文件所在目录
    # original_dir = os.getcwd()
    # os.chdir(config_dir)
    
    # try:
    fraction = []
    PMT_id = []
    for i_y in tqdm(range(toltal_N)):    
        for i_x in range(toltal_N): 
            fraction_tmp = []
            PMT_id_tmp = []        
            for pmt_i in range(28):
                PMT = Polygon(get_PMT_points(pmt_i))
                pixel = Polygon(get_pixel_points(i_x,i_y))
                intersection = PMT.intersection(pixel).area
                if intersection != 0:
                        fraction_tmp.append(intersection/645.16 ) #PMT area 
                        PMT_id_tmp.append(pmt_i)
            fraction.append(fraction_tmp)
            PMT_id.append(PMT_id_tmp)
                
    i=0
    enabledPixelId=[]
    for sublit in fraction:
        if sublit != []:

            enabledPixelId.append(i)
        i+=1
    pixelxyById = np.empty((toltal_N*toltal_N,2),int)
    i = 0
    for iy in range(toltal_N):
        for ix in range(toltal_N):
            pixelxyById[i] = np.array((ix,iy))
            i+=1
        
    weight = np.zeros((28,len(enabledPixelId)))
    for i in range(len(enabledPixelId)):
        id = enabledPixelId[i]
        for j in range(len(fraction[id])):
            weight[PMT_id[id][j]][i] = fraction[id][j]  

    X = pixelxyById[enabledPixelId][0:len(enabledPixelId),0]
    Y = pixelxyById[enabledPixelId][0:len(enabledPixelId),1]        

    #gennerating pseudo inverse of weight
    weight_pseudo_inv = np.linalg.pinv(weight)
    
    # Normalization
    norm_pe_per_channel = copy.deepcopy(pe_per_channel)
    for i in range(len(pe_per_channel)):
        norm_pe_per_channel[i][0:28] = pe_per_channel[i][0:28] / (1e-20 + np.sum(pe_per_channel[i][0:28]))
    
    eventNum = len(pe_per_channel)
    map_all = np.zeros((eventNum , 1, toltal_N, toltal_N), dtype=np.float32)
    out_top_batch = np.array(norm_pe_per_channel)
    aaa_top_batch = np.dot(out_top_batch, weight)
    for i in tqdm(range(eventNum )):
        map_all[i, 0, Y, X] = aaa_top_batch[i]
    
    map_all[map_all<0] =0

    return map_all

def position_construction(pe_per_channel):
    """
    using a Regression model for inference
    
    Parameters:
    -----------
    pe_per_channel : array-like
        输入数据
    config_path : str
        配置文件路径，默认为 "./config.json"
    """
    # 加载配置（使用缓存机制）
    # config, config_dir = load_config(config_path)
    
    # 切换到配置文件所在目录
    # original_dir = os.getcwd()
    # os.chdir(config_dir)
    
    # try:
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载训练好的生成器 G
    # checkpoint_path = config["inference"]["Regression"]["checkpoint_path"]  # 选择最新的 checkpoint
    checkpoint_path = "../models/DeepResNet25.ckpt"
    model = DeepResNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path,map_location=device, weights_only=True))
    model.eval()
    model = model.float()  # 确保模型使用 float32
    print(f"model loaded from epoch {checkpoint_path} ,using device: {device} ")
    
    # 读取模拟数据
    simulated_data = data_trans(pe_per_channel)
    
    # 预处理数据
    # num_samples = len(pe_per_channel)
    # simulated_data = simulated_data[0:num_samples, 0].reshape(num_samples, 1, 5, 5)
    
    # 转换为tensor
    simulated_data = torch.tensor(simulated_data)
    dataset = TensorDataset(simulated_data)
    # batch_size = config["inference"]["Regression"]["batch_size"]
    batch_size = 512
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("total samples: ", len(dataset), ", using a " + str(batch_size) + " batch for data loader")
    
    # 进行推理并存储结果
    generated_data = []
    with torch.no_grad():  # 关闭梯度计算，减少显存占用
        for batch in tqdm(data_loader):
            batch = batch[0].to(device)
            generated_batch = model(batch).cpu().numpy()
            generated_data.append(generated_batch)

    generated_data = np.concatenate(generated_data, axis=0)

    return generated_data
        
    # finally:
    #     # 恢复原始工作目录
    #     os.chdir(original_dir)

# 如果需要在模块级别预先加载配置，可以这样调用：
# config, config_dir = load_config("./config.json")