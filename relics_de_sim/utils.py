import torch
from joblib import load
import numpy as np
def load_generator(model, checkpoint_path, device):
    """
    加载生成器模型和优化器的状态到指定设备。
    
    参数:
    model -- 需要加载的生成器模型
    optimizer -- 生成器的优化器
    checkpoint_path -- 模型检查点的文件路径
    device -- 设备 ('cpu' 或 'cuda')
    
    返回:
    model -- 加载状态后的生成器模型
    optimizer -- 加载状态后的优化器
    epoch -- 当前训练的 epoch
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # 恢复模型状态
    model.load_state_dict(checkpoint['G_state_dict'])
    
    # 恢复优化器状态
#     optimizer.load_state_dict(checkpoint['optimizer_G'])
    
    # 获取 epoch 信息
    epoch = checkpoint['epoch']
    
    print(f"model loaded from epoch {epoch} ,using device: {device} ")
    
    return model.to(device)



from tqdm import tqdm
def PEperChannel2input(PE_per_channel,toltal_N,X,Y,weight): # PEperChannel:(N,64),toltal_N means the PMT arrary is interpreted with a 10by10 matrix
      eventNum = len(PE_per_channel)
      map_all = np.zeros((eventNum, 1, toltal_N, toltal_N), dtype=np.float32)
      PE_per_channel = np.array(PE_per_channel)  
      aaa_top_batch = np.dot(PE_per_channel, weight)  
      for i in tqdm(range(eventNum)):
         map_all[i, 0, Y, X] = aaa_top_batch[i]
      map_all[map_all<0] =0
      return map_all


def inverse_CNN_output(map_all,weight_pseudo_inv,X,Y):
    out_top_reconstructed = []
    for i in tqdm(range(len(map_all))):

        aaa_top_i = map_all[i][0][Y, X]
        out_top_reconstructed.append(aaa_top_i @ weight_pseudo_inv)


    out_top_reconstructed = np.array(out_top_reconstructed)
    out_top_reconstructed[out_top_reconstructed<0] =0


    return   out_top_reconstructed













