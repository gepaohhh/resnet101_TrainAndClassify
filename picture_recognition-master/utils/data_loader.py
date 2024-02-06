from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import os
import random
import math
import shutil

def get_dataset_dataloader(root,batch_size=128,shuffle=True,num_workers=0,resize=[224, 224]):
    '''加载数据集'''
    data_transform = transforms.Compose([
        transforms.Resize(resize),    # 缩放图像大小为 224*224，第一个网络需要的输入尺寸是32*32
        transforms.ToTensor()     # 仅对数据做转换为 tensor 格式操作
    ])
    out_dataset = datasets.ImageFolder(root=root,transform=data_transform)
    out_dataloader = DataLoader(dataset=out_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers)
    return out_dataset,out_dataloader

def judge_file(path):
    '''创建文件，但如果名称存在则添加编号'''
    filename, extension=os.path.splitext(path)
    counter=1
    while os.path.exists(path=path):
        path=filename+"({0})".format(str(counter))+extension
        counter+=1
    return path

def data_split(old_path):
    '''分割数据集'''
    print("=============================分割数据集=============================")
    # 获取父级路径
    old_path_file_name = os.path.basename(old_path)
    parent=os.path.join(old_path,os.pardir)
    parent_path = os.path.abspath(parent)
    # 创建新的子级路径
    new_path = parent_path+'\split_data_{0}'.format(old_path_file_name)
    new_path=judge_file(new_path)
    os.makedirs(new_path)
    # 分割数据集
    for root_dir, sub_dirs, file in os.walk(old_path):# 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))# 遍历每个次级目录
            file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))# 去掉列表中的非jpg格式的文件
            random.shuffle(file_names)# 随机排序
            for i in range(len(file_names)):
                if i < math.floor(0.9*len(file_names)):# 前90%的作为训练集,剩下10%作为测试集
                    sub_path = os.path.join(new_path, 'train_set', sub_dir)
                elif i < len(file_names):
                    sub_path = os.path.join(new_path, 'test_set', sub_dir)
                if os.path.exists(sub_path) == 0:# 创建文件夹
                    os.makedirs(sub_path)
                try:
                    shutil.copy(os.path.join(root_dir, sub_dir, file_names[i]), 
                                os.path.join(sub_path, file_names[i]))# 复制图片，从源到目的地
                except:
                    continue
    print("============================完成数据集分割============================")
    return new_path