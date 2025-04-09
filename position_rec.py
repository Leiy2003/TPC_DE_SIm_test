import pandas as pd
import torch.nn as nn
from numba import jit
import sys
import torch
import numpy as np
import pickle as pkl
import time

print("Trans into torch tensor input")


def CNN_ouput(out):
    train_Num = 0
    event_Num = len(out)
    test_Num = event_Num
    pos = np.zeros((test_Num, 2))
    print(event_Num)
    layer0 = [18, 17, 16, 15, 14]
    layer1 = [19, 40, 39, 38, 37, 13]
    layer2 = [20, 41, 56, 55, 54, 36, 12]
    layer3 = [21, 42, 57, 4, 5, 6, 53, 35, 11]
    layer4 = [22, 43, 58, 0, 1, 2, 3, 52, 34, 10]
    layer5 = [23, 44, 59, 7, 8, 9, 63, 51, 33]
    layer6 = [24, 45, 60, 61, 62, 50, 32]
    layer7 = [25, 46, 47, 48, 49, 31]
    layer8 = [26, 27, 28, 29, 30]
    layerlist = [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

    xy_index = np.empty((64, 2), int)

    for i in range(9):
        layeri = layerlist[i]
        id_x = int((10 - len(layeri)) / 2)
        for num in layeri:
            xy_index[num][0] = id_x
            xy_index[num][1] = i
            id_x += 1

    output_test = []
    outpos_test = []

    # t=time.time()
    for i in range(test_Num):
        tmp = np.zeros((10, 10), float)
        outpos_test.append(pos[i + train_Num])
        for j in range(64):
            xj = xy_index[j][0]
            yj = xy_index[j][1]
            tmp[yj][xj] = out[i + train_Num][j] / (
                np.sum(out[i + train_Num][0:64]) + np.e
            )
        output_test.append(tmp)
        # if (i+1)%10000 ==0:
        # print (i+1,'/',test_Num,',time:',time.time()-t)
        # t= time.time()
    print("done of generating test files")

    outpos_test = np.array(outpos_test)

    output_test = np.array(output_test)

    outpos_test = torch.tensor(outpos_test)

    outpos_test = outpos_test.type(torch.FloatTensor)

    output_test = torch.tensor(output_test)

    output_test = output_test.type(torch.FloatTensor)

    return output_test, outpos_test

def CNN_output_vec(PMT_response):
    event_Num = len(PMT_response)
    # pos = np.zeros((test_Num, 2))
    # train_Num = 0
    print(event_Num)
    layer0 = [18, 17, 16, 15, 14]
    layer1 = [19, 40, 39, 38, 37, 13]
    layer2 = [20, 41, 56, 55, 54, 36, 12]
    layer3 = [21, 42, 57, 4, 5, 6, 53, 35, 11]
    layer4 = [22, 43, 58, 0, 1, 2, 3, 52, 34, 10]
    layer5 = [23, 44, 59, 7, 8, 9, 63, 51, 33]
    layer6 = [24, 45, 60, 61, 62, 50, 32]
    layer7 = [25, 46, 47, 48, 49, 31]
    layer8 = [26, 27, 28, 29, 30]
    layerlist = [layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8]

    # xy_index = np.empty((64, 2), int)
    index = np.empty(64, int)

    # 将对应的通道对应到10*10矩阵的位置
    for i in range(9):
        layeri = layerlist[i]
        id_x = int((10 - len(layeri)) / 2)
        for num in layeri:
            index[num] = id_x + i * 10
            id_x += 1

    channel_eventID = np.arange(event_Num)[:, np.newaxis]

    output_CNN = np.zeros((event_Num, 100))
    output_CNN[channel_eventID, index] = PMT_response
    output_CNN = output_CNN.reshape(event_Num,10,10)
    output_CNN = output_CNN / np.sum(output_CNN,axis = (1,2)).reshape((event_Num, 1, 1)) 
    output_CNN = torch.FloatTensor(output_CNN)
    # output_CNN为N*9*9矩阵，各个通道值按照对应关系填入对应坐标中,outpos_tes为空白的N*2位置矩阵
    return output_CNN
# CNN


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # (10-5+2*2)/1+1 = 10
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # (10-2)/2+1 = 5
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            # (5-2+2*1)/1+1 = 6
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # (6-2)/2+1 = 3
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            # (3-3+2*1)/1+1 = 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # (3-2)/1+1 = 2
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.fc1 = nn.Linear(2 * 2 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # input features  # output features
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.dropout(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # print (out[0],out.shape )
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def CNN_array(data, modelname, devicename):

    data_test = data
    data_test = torch.reshape(data_test, (-1, 1, 10, 10))

    import torch.utils.data as Data

    model = ConvNet()
    model.load_state_dict(torch.load(modelname, map_location=devicename))
    model.eval()
    pre_pos = model(data_test)
    pre_pos = torch.detach(pre_pos).numpy()

    return pre_pos


def position_construction(pe_per_channel,model):
    print(pe_per_channel.shape)

    # 检查cuda占用状态
    while open("cuda_status.txt", "r").read() == "1":
        time.sleep(np.random.random() * 5 + 1)
        print("GPU is busy, waiting.")

    pe_per_channel_convert = CNN_output_vec(pe_per_channel)
    # x2,y2,pos_pre2,loss2=CNN_array('../CNN_Relics_g2/data/',output_test,outpos_test,'../CNN_Relics/data/model0.99.16.ckpt','cuda:0')
    
    # 打开文件以进行写入gpu状态为占用
    with open("cuda_status.txt", "w") as file:
        # 写入修改后的内容
        file.write(str(1))
        file.flush()
    
    pos_pre2 = CNN_array(
        pe_per_channel_convert,
        model,
        "cuda:0",
    )

    # 打开文件以进行写入gpu状态为闲置
    with open("cuda_status.txt", "w") as file:
        # 写入修改后的内容
        file.write(str(0))
        file.flush()

    return pos_pre2
