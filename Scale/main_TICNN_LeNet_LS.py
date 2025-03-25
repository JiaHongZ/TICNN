import torch
from torch.autograd import Variable
import os
import torchvision.transforms as trans
from torchvision.transforms import ToPILImage
from torchvision.datasets.mnist import MNIST
from data.MNIST_LS import MNIST_LS
show = ToPILImage()
import torch.nn as nn
import utils.logger as log
import numpy as np
import tqdm
from models.LeNet5 import *
from torch.utils.data import DataLoader
import utils.update_function as uf
import csv
from models.loss import elbo, Elbo

# 固定随机数种子
def setup_seed(seed):
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True   # 选择确定性算法

torch.autograd.set_detect_anomaly(True)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def test(model, test_loader):
    # 习惯性的用法，测试的开始
    model.eval()
    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(test_loader)):
        img, lab = data
        b, c, h, w = img.shape
        img = img.cuda()
        lab = lab.cuda()
        # no_grad表示不计算梯度
        with torch.no_grad():
            pre,*_ = model(img)
        _, lab_pre = torch.max(pre.data, 1)
        currect = torch.sum(lab_pre == lab.data)
        count += b
        correct += currect
        
    return torch.true_divide(correct, count)

def train(mynet, train_loader, test_loader, netname):
    mynet.train()
    print('-----start train model------')
    # 定义优化器， 初始学习率0.0001， 权重衰减
    optimiter = torch.optim.Adam(mynet.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    # 余弦退火调整学习率， 逐渐减小学习率
    warmup = True
    if warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7)

    # 损失函数 交叉熵损失函数
    loss_f1 = torch.nn.CrossEntropyLoss()  # 与pattern的损失函数
    elbo_loss = Elbo()
    loss_f2 = torch.nn.MSELoss()  # 与pattern的损失函数

    best = 0
    best_epo = 0
    train_accs = []
    train_losss = []
    test_accs = []
    for epo in range(all_epoch):
        mynet.eval()
        currect = 0
        count = 0
        code_all = []
        losss = 0
        # 获取一个batch的数据
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            img, lab = data # img 它的大小是 (64,3,224,224)
            b, c, h, w = img.shape
            img = img.cuda()
            lab = lab.cuda()
            if '_learnw' in netname:
                loss_other = 0
                if mnist_sl_train == 0:
                    scale_c = 0.25
                elif mnist_sl_train == 1:
                    scale_c = 0.5
                elif mnist_sl_train == 2:
                    scale_c = 1
                else:
                    scale_c = 1
                pre, codes = mynet(img,False,scale_c)
                weightsr = codes[1]
                weightss = codes[0]
                scales3 = codes[2]
                if weightss!=None and scales3!=None:
                    scales3[:, 0] = 1 / (scales3[:, 0])
                    scales3[scales3>1] = 1
                    loss_other = loss_f2(torch.cat([weightss,weightsr],1), scales3)
                loss = loss_f1(pre, lab.long()) + loss_other
            elif '_ablation' in netname:
                pre, codes = mynet(img,False)
                loss = loss_f1(pre, lab.long())
            elif '_sc' in netname:
                pre, codes = mynet(img,False)
                loss = loss_f1(pre, lab.long())
            elif 'PSTN' in netname:
                pre, thetas, beta = mynet(img,False)
                loss_other, individual_terms = elbo_loss(pre, beta, lab.long())
                loss = loss_f1(pre, lab.long()) + loss_other
            else:
                pre, codes = mynet(img)
                loss = loss_f1(pre, lab.long())
            # torch.autograd.set_detect_anomaly(True)
            # 误差反向传播，优化
            optimiter.zero_grad() # 清空梯度
            loss.backward() # 反向传播
            optimiter.step() # 更新

            _, lab_pre = torch.max(pre.data, 1)
            currect += torch.sum(lab_pre == lab.data).cpu()
            count += b
            
        train_accs.append(int(currect)/int(count))

        # 每5轮更新学习率
        if epo % 10 == 0 and epo > 0:
            scheduler.step()
            print('epoch', epo, ' current learning rate', optimiter.param_groups[0]['lr'])

        test_val_ = test(mynet, test_loader)
        test_accs.append(test_val_.cpu())

        # 保存参数
        if test_val_ > best:
            # 保存参数文件
            torch.save(mynet.state_dict(), path +'/best.pth')
            best = test_val_
            best_epo = epo
            print('best',best)
    logger.info('----dataname:[{}]----netname:[{}]----not balance:[{}]----seed:[{}]---'.format(netname,dataname,notb,seed))
    logger.info('best:[{}] epoch:[{}]'.format(best, best_epo))
    logger.info('---------------------')
    np.savetxt(path +'/train_acc.csv', train_accs, delimiter=',')
    # np.savetxt(path +'/train_loss.csv', train_losss, delimiter=',')
    np.savetxt(path +'/test_acc.csv', test_accs, delimiter=',')
    return mynet

def test_all(test_loader, model):
    model.eval()

    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(test_loader)):
        img, lab = data
        b, c, h, w = img.shape
        img = img.cuda()
        lab = lab.cuda()
        # no_grad表示不计算梯度
        with torch.no_grad():
            pre, *_ = model(img)
        _, lab_pre = torch.max(pre.data, 1)
        currect = torch.sum(lab_pre == lab.data)
        count += b
        correct += currect
    return torch.true_divide(correct, count)

if __name__ == '__main__':
    '''
        Init parameters
    '''

    batch_size = 128
    notbs = [False]
    netnames = ['LeNet5_Retinal_learnw_free_max1']
    datanames = ['mnist_ls']
    # 一共跑3次，防止随机误差,每次使用不同的随机种子 0 50 100
    seeds = [0]
    all_epoch = 60
    if not os.path.exists(os.path.join(os.getcwd(), 'model_zoo')):
        os.makedirs(os.path.join(os.getcwd(), 'model_zoo'))
    att_alphas = [5] # 对放缩权重的修正系数,目前没用了
    retinal_patchs = [(112,112),(96,96),(84,84)] # retinal感受野
    mnist_sl_trains = [0,1,2]
    mnist_sl_tests = [4,8,12,16]
    file_train = [
        'mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr1p000_scte1p000.h5',
        'mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr2p000_scte2p000.h5',
        'mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr4p000_scte4p000.h5',
        'mnist_large_scale_tr50000_vl10000_te10000_outsize112-112_sctr1p000-4p000_scte1p000-4p000.h5',
    ]
    file_test = [
        'mnist_large_scale_te10000_outsize112-112_scte0p500.h5',
        'mnist_large_scale_te10000_outsize112-112_scte0p595.h5',
        'mnist_large_scale_te10000_outsize112-112_scte0p707.h5',
        'mnist_large_scale_te10000_outsize112-112_scte0p841.h5',
        'mnist_large_scale_te10000_outsize112-112_scte1p000.h5',
        'mnist_large_scale_te10000_outsize112-112_scte1p189.h5',
        'mnist_large_scale_te10000_outsize112-112_scte1p414.h5',
        'mnist_large_scale_te10000_outsize112-112_scte1p682.h5',
        'mnist_large_scale_te10000_outsize112-112_scte2p000.h5',
        'mnist_large_scale_te10000_outsize112-112_scte2p378.h5',
        'mnist_large_scale_te10000_outsize112-112_scte2p828.h5',
        'mnist_large_scale_te10000_outsize112-112_scte3p364.h5',
        'mnist_large_scale_te10000_outsize112-112_scte4p000.h5',
        'mnist_large_scale_te10000_outsize112-112_scte4p757.h5',
        'mnist_large_scale_te10000_outsize112-112_scte5p657.h5',
        'mnist_large_scale_te10000_outsize112-112_scte6p727.h5',
        'mnist_large_scale_te10000_outsize112-112_scte8p000.h5',
    ]
    for netname in netnames:
        hidden = 84
        classes = 10
        rw = 0.05
        updateW_neurons = hidden // 12
        sigma = 12

        # 创建路径
        path_ = os.getcwd()
        path = os.path.join(path_, 'model_zoo', str(netname))
        # model path
        if not os.path.exists(path):
            os.makedirs(path)
            print('path create')
        log_path = os.path.join(path, 'train.log')
        logger = log.get_logger(log_path)

        for seed in seeds:
            setup_seed(seed)
            for dataname in datanames:
                for notb in notbs:
                    # 加载数据
                    # 数据预处理 （防止过拟合）
                    transform = trans.Compose([
                        # trans.RandomResizedCrop((112,112),(0.5,4)),
                        trans.ToTensor(),
                    ])
                    if dataname == 'mnist':
                        test_data = MNIST(root='E:\dataset', transform=transform, train=False)
                    # input
                    if 'mnist' in dataname:
                        input = 1
                    else:
                        input = 3
                    if netname == 'LeNet':
                        hidden = 128
                    if dataname == 'cifar100':
                        classes = 100
                    # balance
                    if notb:
                        netvision = 'notb'
                    else:
                        netvision = 'all'
                    # 创建路径
                    path_ = os.getcwd()
                    path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname))
                    # model path
                    if not os.path.exists(path):
                        os.makedirs(path)
                        print('path create')

                    # net
                    if dataname == 'mnist_ls':
                        create_obj = compile(netname, 'models.' + netname.split('_')[0] + '.py', 'eval')
                        if 'Retinal' in netname:
                            path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision))
                            test_acc = []
                            for retinal_patch in retinal_patchs:
                                for mnist_sl_train in mnist_sl_trains:
                                    # path update
                                    path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname),
                                                        'retinal_patch_' + str(retinal_patch),
                                                        'image_r_patch_' + str(retinal_patch),
                                                        'trainscale_' + str(mnist_sl_train))
                                    if not os.path.exists(path):
                                        os.makedirs(path)
                                        print('path create')

                                    # dataset and train
                                    train_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                            filename=file_train[mnist_sl_train],
                                                            transform=transform,
                                                            data_model='train')
                                    test_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                            filename=file_train[mnist_sl_train],
                                                            transform=transform,
                                                            data_model='test')
                                    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=16, shuffle=True)
                                    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=16, shuffle=True)

                                    logger.info('train data:[{}]'.format(file_train[mnist_sl_train]))
                                    logger.info('retinal size和recover image size一样')
                                    logger.info('netname:[{}], retinal_patch:[{}], image_r_patch:[{}]'.format(netname,
                                                                                                                retinal_patch,
                                                                                                                retinal_patch))
                                    mynet = eval(create_obj)(in_channels=input, num_classes=classes,
                                                                retinal_H=retinal_patch[0],
                                                                retinal_W=retinal_patch[1],
                                                                image_H=retinal_patch[0],image_W=retinal_patch[1],).cuda()
                                    mynet = train(mynet, train_loader, test_loader, netname)
                                    logger.info('train finished!')

                                    # dataset and test
                                    mynet.load_state_dict(torch.load(path + '/best.pth'))
                                    for mnist_sl_test in mnist_sl_tests:
                                        test_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                                filename=file_test[mnist_sl_test],
                                                                transform=transform,
                                                                data_model='test')
                                        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=16, shuffle=True)
                                        test_val_ = test_all(test_loader, mynet)
                                        test_acc.append(test_val_.cpu())
                                        logger.info('---------------------')
                                        logger.info('train data:[{}], retinal_patch:[{}], test data:[{}], overall test:[{}]'.format(
                                            file_train[mnist_sl_train], retinal_patch[0], file_test[mnist_sl_test], test_val_))
                                        logger.info('---------------------')
                                np.savetxt(path_svc + '/'+str(retinal_patch[0])+'_'+'test_acc.csv', test_acc, delimiter=',')
                                test_acc = []
                        else:
                            path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision))
                            test_acc = []
                            for mnist_sl_train in mnist_sl_trains:
                                # path update
                                path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname),
                                                    'trainscale_' + str(mnist_sl_train))
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    print('path create')

                                # dataset and train
                                train_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                      filename=file_train[mnist_sl_train],
                                                      transform=transform,
                                                      data_model='train')
                                test_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                     filename=file_train[mnist_sl_train],
                                                     transform=transform,
                                                     data_model='test')
                                train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=16, shuffle=True)
                                test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=16, shuffle=True)

                                logger.info('train data:[{}]'.format(file_train[mnist_sl_train]))
                                mynet = eval(create_obj)(in_channels=input, num_classes=classes,
                                                        ).cuda()
                                mynet = train(mynet, train_loader, test_loader, netname)
                                logger.info('train finished!')

                                # dataset and test
                                mynet.load_state_dict(torch.load(path + '/best.pth'))
                                for mnist_sl_test in mnist_sl_tests:
                                    test_data = MNIST_LS(path='/zjh/data/MNIST_Large_Scale/',
                                                         filename=file_test[mnist_sl_test],
                                                         transform=transform,
                                                         data_model='test')
                                    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=16, shuffle=True)
                                    test_val_ = test_all(test_loader, mynet)
                                    test_acc.append(test_val_.cpu())
                                    logger.info('---------------------')
                                    logger.info('train data:[{}], test data:[{}], overall test:[{}]'.format(
                                        file_train[mnist_sl_train], file_test[mnist_sl_test], test_val_))
                                    logger.info('---------------------')
                            np.savetxt(path_svc + '/test_acc.csv', test_acc, delimiter=',')
