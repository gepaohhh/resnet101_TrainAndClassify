import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
from collections import OrderedDict, namedtuple
from itertools import product
import pandas as pd
import json

class RunBuilder(object):
    '''参数装入列表--输出为字典'''
    @staticmethod
    def get_runs(params):
        Run=namedtuple('Run', params.keys())
        runs=[]
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class RunManger():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        # 获取训练集精度最小值
        self.epoch_loss_list = []
        self.epoch_loss_min = 0

        self.val_count = 0
        self.val_loss = 0
        self.val_num_correct = 0
        self.val_start_time = None
        # 获取验证集精度最大值
        self.val_acc_list = []
        self.val_acc_max = 0

        self.run_params = None
        self.run_count = 0
        self.run_data = [] #存入json/csv
        self.val_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.val_loader = None
        self.tb = None
        self.judge_min_loss_epoch = False
        self.judge_max_acc_val = False
    
    def begin_run(self, run, network, loader, val_loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader
        self.val_loader = val_loader
        self.tb = SummaryWriter(comment=f'-{run}')
        # 照片格网
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        # 绘制批处理
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, 
                          images.to(getattr(run,'device','cpu')))

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
    
    '''训练集用'''
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        self.epoch_loss_list.append(loss)
        # 判断是否是最小的
        if loss == min(self.epoch_loss_list):
            self.judge_min_loss_epoch = True
        else:
            self.judge_min_loss_epoch = False

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)

        df = pd.DataFrame.from_dict(results, orient='index').T
        print(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    '''验证集用'''
    def begin_val(self):
        self.val_start_time = time.time()
        self.val_count += 1
        self.val_loss = 0
        self.val_num_correct = 0

    def end_val(self):
        val_duration = time.time() - self.val_start_time

        val_loss = self.val_loss / len(self.val_loader.dataset)
        val_acc = self.val_num_correct / len(self.val_loader.dataset)
        self.val_acc_list.append(val_acc)
        # 判读验证集精度是否最大
        if val_acc == max(self.val_acc_list):
            self.judge_max_acc_val = True
        else:
            self.judge_max_acc_val = False

        self.tb.add_scalar('val Loss',val_loss,self.val_count)
        self.tb.add_scalar('val Accuracy',val_acc,self.val_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.val_count
        results["val_loss"] = val_loss
        results["val_acc"] = val_acc
        results["val duration"] = val_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.val_data.append(results)

        df = pd.DataFrame.from_dict(results, orient='index').T
        print(df)

    def track_val_loss(self, loss):
        self.val_loss += loss.item() * self.loader.batch_size

    def track_val_num_correct(self, preds, labels):
        self.val_num_correct += self._get_num_correct(preds, labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save_model(self, epoch, present_network, model_path=""):
        '''---训练集中最小损失值---'''
        if self.judge_min_loss_epoch == True:
            print(f"保存---训练集中最小损失值---{epoch+1}轮")
            torch.save(present_network.state_dict(),model_path+"\\epoch_loss_min.pt")
        else:
            print("不是---训练集---中的最小损失值")
        '''---测试集中最大精度---'''
        if self.judge_max_acc_val == True:
            print(f"保存---测试集中最大精度---{epoch+1}轮")
            torch.save(present_network.state_dict(),model_path+"\\val_acc_max.pt")
        else:
            print("不是---测试集---中的最大精度")
        '''---保存最后一轮---'''
        print(f"保存最后一轮模型---{epoch+1}轮")
        torch.save(present_network.state_dict(), model_path+"\\last.pt")

    def save(self, fileName, file_path=""):
        # csv
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(file_path+"\\"+f'{fileName}_train.csv')
        pd.DataFrame.from_dict(
            self.val_data, orient='columns'
        ).to_csv(file_path+"\\"+f'{fileName}_val.csv')
        # json
        with open(file_path+"\\"+f'{fileName}_train.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)
        with open(file_path+"\\"+f'{fileName}_val.json', 'w', encoding='utf-8') as f:
            json.dump(self.val_data, f, ensure_ascii=False, indent=4)