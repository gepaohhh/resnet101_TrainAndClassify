import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def get_num_correct(preds, labels):
    '''input:预测+标注   output:正确个数'''
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def _get_all_preds(device,model, loader):
    '''input:模型+数据加载  output:预测结果'''
    all_preds = torch.tensor([]).to(device=device)
    for batch in loader:
        images, labels = batch
        preds = model(images.to(device))
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
    return all_preds

def _plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''绘制混淆矩阵'''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def draw_confusion_matrix(device,network,all_loader,all_set,save_root,run):
    # 绘制混淆矩阵--train_preds/pre_set
    train_preds = _get_all_preds(device=device,model=network, loader=all_loader)
    preds_correct = get_num_correct(train_preds, torch.IntTensor(all_set.targets).to(device=device))
    print("total correct:{preds_correct} | accuracy:{accuracy}".format(preds_correct=preds_correct,
                                                                    accuracy=preds_correct/len(all_set)))
    stacked = torch.stack(
        (
            torch.IntTensor(all_set.targets).to(device=device),
            train_preds.argmax(dim=1)
        ),
        dim=1
    )
    cmt = torch.zeros(10,10,dtype=torch.int64) #建立空矩阵
    for twist in stacked:
        true_list, pre_list = twist.tolist()
        cmt[true_list,pre_list] = cmt[true_list,pre_list] + 1
    cm = confusion_matrix(torch.IntTensor(all_set.targets), train_preds.argmax(dim=1).cpu().numpy())
    plt.figure(figsize=(10,10))
    _plot_confusion_matrix(cm, all_set.classes)
    plt.savefig(save_root+"\\"+f"cmt_{run}.png")
    plt.show()