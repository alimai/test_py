import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from un import Unet
from DataHelper import *
from tqdm import tqdm
import numpy as np
#import skimage.io as io
from skimage import io

PATH = './model/unet_model.pt'

# 是否使用cuda
device = "cpu"#"cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 4

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
label_transforms = transforms.ToTensor()


def train_model(model, criterion, optimizer, dataload, num_epochs=10):
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload):#for x, y in dataload#data/label
            step += 1
            inputs = x.to(device)
            labels = y.to(device)


            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #backward
            # zero the parameter gradients
            # 清空过往梯度,一般放在backward()前
            optimizer.zero_grad()
            loss.backward()#反向传播，计算当前梯度；
            optimizer.step()#根据梯度更新网络参数

            '''
            #backward梯度累加(可替换上边的3行backward)
            # 1> loss regularization
            loss = loss/accumulation_steps
            loss.backward()
            # 2> update parameters of net
            if((i+1)%accumulation_steps)==0:
                # optimizer the net
                optimizer.step()        # update parameters of net
                optimizer.zero_grad()   # reset gradient
            '''

            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) 
            # // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
    torch.save(best_model.state_dict(), PATH)
    return best_model
'''
<梯度累加>
总结来说：梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，
不断累加，累加一定次数后，根据累加的梯度更新网络参数，
然后清空梯度，进行下一次循环。

一定条件下，batchsize越大训练效果越好，梯度累加则实现了batchsize的变相扩大，
如果accumulation_steps为8，则batchsize '变相' 扩大了8倍，
是我们这种乞丐实验室解决显存受限的一个不错的trick，
使用时需要注意，学习率也要适当放大。
'''
# 训练模型
def train():
    model = Unet(1, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()#损失函数，包含边界
    optimizer = optim.Adam(model.parameters())#参数优化，包含学习速率等
    train_dataset = TrainDataset("./dataset/train/raw", "./dataset/train/tooth",\
         transform=x_transforms,target_transform=label_transforms)#"./dataset/train/image", "./dataset/train/label"
    
    dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 保存模型的输出结果
def test():
    model = Unet(1, 1)
    model.load_state_dict(torch.load(PATH))
    test_dataset = TestDataset("dataset/test", transform=x_transforms,target_transform=label_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            img = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
            io.imsave("./dataset/test/" + str(index) + "_predict.png", img)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    print("开始训练")
    train()
    print("训练完成，保存模型")
    print("-"*20)
    print("开始预测")
    test()
