import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
from utils import data_loader, Run, draw_conf_matrix

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) # 打开梯度跟踪
# print(torch.__version__)
# print(torchvision.__version__)

def train(params,root_dir,cmt_save_root):
    m=Run.RunManger()
    for run in Run.RunBuilder.get_runs(params=params):
        print(f'=================={run}==================')
        '''network,train_set,train_loader,optimizer'''
        device = torch.device(run.device)
        network = models.resnet101().to(device=device) #设置骨干模型
        train_set,train_loader = data_loader.get_dataset_dataloader(root=root_dir[0],
                                                                    batch_size=run.batch_size,
                                                                    shuffle=run.shuffle,
                                                                    num_workers=run.num_workers) #设置数据集
        test_set,test_loader = data_loader.get_dataset_dataloader(root=root_dir[1],
                                                                  batch_size=run.batch_size,
                                                                  shuffle=run.shuffle,
                                                                  num_workers=run.num_workers)
        optimizer = optim.Adam(network.parameters(), lr=run.lr) #设置优化器
        '''预测-计算损失-梯度清零-反向传播-优化器前进'''
        m.begin_run(run=run,network=network,loader=train_loader,val_loader=test_loader)
        for epoch in range(run.epoches):
            print(f"=======================epoch-{epoch+1}-trian=======================")
            # 开始训练
            network.train() #BN层 Dropout层--训练
            m.begin_epoch()
            for batch in tqdm(train_loader,total=len(train_loader)):
                images = batch[0].to(device)
                labels = batch[1].to(device)

                preds = network(images)
                loss = F.cross_entropy(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                m.track_loss(loss=loss)
                m.track_num_correct(preds=preds, labels=labels)
            m.end_epoch()
            # 开始验证
            print(f"=======================epoch-{epoch+1}-test/val=======================")
            network.eval()
            m.begin_val()
            for batch in tqdm(test_loader,total=len(test_loader)):
                val_images= batch[0].to(device)
                val_labels= batch[1].to(device)

                val_preds = network(val_images)
                val_loss = F.cross_entropy(val_preds, val_labels)

                m.track_val_loss(loss=val_loss)
                m.track_val_num_correct(preds=val_preds,labels=val_labels)
            m.end_val()
            # 保存模型
            m.save_model(epoch=epoch,present_network=network,model_path=cmt_save_root)
        m.end_run()
        draw_conf_matrix.draw_confusion_matrix(run=run,device=device,network=network,all_set=train_set,all_loader=train_loader,save_root=cmt_save_root)
    m.save('results',file_path=cmt_save_root)
    print("=======================================最中结果=======================================")
    print(pd.DataFrame.from_dict(m.run_data, orient='columns').sort_values('epoch duration'))

def experimenting_with_hyperparameter_values(batch_size,lr,epoches,
                                             shuffle,num_workers,device,
                                             root_dir="",
                                             cmt_save_root=""):
    '''批量测试超参---5个参数以列表的方式输入'''
    parameters = dict(
        batch_size=batch_size,
        lr=lr,
        epoches=epoches,
        shuffle=shuffle,
        num_workers=num_workers,
        device=device
    )
    train(params=parameters,root_dir=root_dir,cmt_save_root=cmt_save_root)


if __name__=="__main__":
    # 训练集和测试集路径
    root_dir = [r"D:\StreetPicture\picture_recognition\classify\split_data_floatingobjects\train_set",
            r"D:\StreetPicture\picture_recognition\classify\split_data_floatingobjects\test_set"]
    # 混淆矩阵保存路径
    cmt_save_root = r"D:\StreetPicture\picture_recognition\classify\log"
    '''设置超参'''
    batch_size_list = [16]
    lr_list = [.01]
    epoches_list = [1]
    shuffle_list = [True]
    num_workers_list = [2]
    device=['cuda' if torch.cuda.is_available() else 'cpu']
    '''运行'''
    experimenting_with_hyperparameter_values(batch_size=batch_size_list,
                                             lr=lr_list,
                                             epoches=epoches_list,
                                             shuffle=shuffle_list,
                                             num_workers=num_workers_list,
                                             root_dir=root_dir,
                                             cmt_save_root=cmt_save_root,
                                             device=device)   