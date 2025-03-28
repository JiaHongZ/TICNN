import torch
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from data.Caltech101_LS import DatasetCaltech101
from data.MNIST_LS import MNIST_LS
show = ToPILImage()
import torch.nn as nn
import utils.logger as log
import numpy as np
import tqdm
from models.ResNet18 import *
from torch.utils.data import DataLoader
import utils.update_function as uf
import csv

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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def val(model, val_loader):
    # 习惯性的用法，测试的开始
    model.eval()

    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(val_loader)):
        img, lab, scale = data
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
    model.train()
    return torch.true_divide(correct, count)
def train(mynet, train_loader, val_loader, netname):
    print('-----start train model------')
    # 定义优化器， 初始学习率0.0001， 权重衰减
    optimiter = torch.optim.Adam(mynet.parameters(),lr=0.0001,betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0001)
    ######### Scheduler ###########
    # 余弦退火调整学习率， 逐渐减小学习率
    warmup = True
    if warmup:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiter, 10, eta_min=1e-7)
    # 损失函数 交叉熵损失函数
    loss_f1 = torch.nn.CrossEntropyLoss()  # 与pattern的损失函数
    loss_f2 = torch.nn.MSELoss()  # 与pattern的损失函数

    best = 0
    best_epo = 0
    train_accs = []
    train_losss = []
    test_accs = []
    for epo in range(all_epoch):
        currect = 0
        count = 0
        code_all = []
        losss = 0
        # 获取一个batch的数据
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            img, lab, scales = data # img 它的大小是 (64,3,224,224)
            b, c, h, w = img.shape
            # 把tensor变成变量
            img = Variable(img).cuda() # .cuda()用GPU
            lab = Variable(lab).cuda()
            loss_other = 0

            if '_learnw' in netname:
                if len(sl_train) == 3:
                    pre, codes = mynet(img, False, scales.to(dtype=torch.float).unsqueeze(1).cuda())
                elif len(sl_train) == 2:
                    pre, codes = mynet(img, False, scales.to(dtype=torch.float).unsqueeze(1).cuda())
                elif sl_train[0] == 0.5:
                    pre, codes = mynet(img,False,0.5)
                elif sl_train[0] == 1.0:
                    pre, codes = mynet(img, False, 1)
                # elif sl_train[0] == [0.5,1.0,1.5]:
                    # pre, codes = mynet(img, False, 1)
                else:
                    pre, codes = mynet(img, False, 1)
                # x_scaled_lp = codes[2]
                # print(weightsr[0], weightss[0], scales3[0])
                weightsr = codes[1]
                weightss = codes[0]
                scales3 = codes[2]
                if weightsr != None:
                    scales3[:, 0] = 1 / (scales3[:, 0])
                    scales3[scales3>1] = 1
                    loss_other = loss_f2(torch.cat([weightss,weightsr],1), scales3)
                    loss = loss_f1(pre, lab.long()) + loss_other
                else:
                    loss = loss_f1(pre, lab.long())
            elif '_sc' in netname:
                import datetime
                # print('0', datetime.datetime.now())
                pre, _ = mynet(img, False)
                loss = loss_f1(pre, lab.long()) + loss_other
                # print('1', datetime.datetime.now())
            elif '_ms' in netname:
                pre, codes = mynet(img,False)
                weights = codes[0]
                scales3 = codes[1]
                # x_scaled_lp = codes[2]
                # print(scales3)
                for scale_i in range(len(weights)-1):
                    loss_other += loss_f2(weights[scale_i], weights[-1]*(scales3[scale_i]))
                    # loss_other += loss_f2(weights[scale_i], torch.ones_like(weights[scale_i])*torch.Tensor(np.array(scales3[scale_i])).cuda())
                    print(weights[scale_i][0], (scales3[scale_i]))
                # loss_other += loss_f2(weights[-1], torch.ones_like(weights[-1]).cuda())
                print(weights[-1][0], 1)
                # loss_other += loss_f2(x_scaled_lp[scale_i], x_scaled_lp[-1])
                loss = loss_f1(pre, lab.long()) + loss_other
            else:
                pre, codes = mynet(img)
                loss = loss_f1(pre, lab.long()) + loss_other

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

        test_val_ = val(mynet, val_loader)
        test_val_ = test_val_.cpu()
        test_accs.append(test_val_)
        logger.info('Training test val:[{}] weightss:[{}] weightr:[{}] scale:[{}] epoch:[{}]'.format(test_val_, weightss[0], weightsr[0], scales3[0], epo))
        # 保存参数
        if test_val_ > best:
            # 保存参数文件
            torch.save(mynet.state_dict(), path +'/best.pth')
            best = test_val_
            best_epo = epo
            print('best',int(currect)/int(count),best)
    logger.info('----dataname:[{}]----netname:[{}]----not balance:[{}]----seed:[{}]---'.format(netname,dataname,notb,seed))
    logger.info('best:[{}] epoch:[{}]'.format(best, best_epo))
    logger.info('---------------------')
    np.savetxt(path +'/train_acc.csv', train_accs, delimiter=',')
    # np.savetxt(path +'/train_loss.csv', train_losss, delimiter=',')
    np.savetxt(path +'/val_acc.csv', test_accs, delimiter=',')
    return mynet
def test_all(test_loader, model):
    model.eval()
    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(test_loader)):
        img, lab, scale = data
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
    model.train()
    return torch.true_divide(correct, count)

if __name__ == '__main__':
    '''
        Init parameters
    '''

    batch_size = 32
    notbs = [False]
    netnames = ['ResNet18_Retinal_learnw_free1_max1']
    datanames = ['caltech101']
    seeds = [0]
    all_epoch = 100
    if not os.path.exists(os.path.join(os.getcwd(), 'model_zoo')):
        os.makedirs(os.path.join(os.getcwd(), 'model_zoo'))
    retinal_patchs = [(224,224)]
    train_scales = [[1.0],[0.5,1.0],[0.5]]
    test_scales = [[0.25],[0.5],[0.75],[1.0],[1.25],[1.5],[1.75],[2.0]]

    for netname in netnames:
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
                    if dataname == 'caltech101':
                        classes = 101
                        inputchannel = 3

                        data_transforms = {
                            'train': transforms.Compose([
                                transforms.Resize((230, 230)),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                            ]),
                            'val': transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                            ]),
                            'test': transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                            ]),
                        }

                        for sl_train in train_scales:
                            root = r'/zjh/data/caltech101_scale'

                            trainset = DatasetCaltech101(root, sl_train, 'train', data_transforms['train'])
                            valset = DatasetCaltech101(root, sl_train, 'val', data_transforms['val'])
                            train_loader = DataLoader(trainset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      drop_last=False,
                                                      pin_memory=True)
                            val_loader = DataLoader(valset,
                                                    batch_size=batch_size,
                                                    num_workers=8,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    pin_memory=True)

                            if 'Retinal' in netname:
                                create_obj = compile(netname, 'models.' + netname.split('_')[0] + '.py', 'eval')
                                path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision))
                                test_acc = []
                                for retinal_patch in retinal_patchs:
                                    # for image_r_patch in image_r_patchs:
                                        # path update
                                    path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname),
                                                        'retinal_patch_' + str(retinal_patch),
                                                        'image_r_patch_' + str(retinal_patch),
                                                        'trainscale_' + str(sl_train))
                                    if not os.path.exists(path):
                                        os.makedirs(path)
                                        print('path create')

                                    logger.info('netname:[{}], retinal_patch:[{}], image_r_patch:[{}]'.format(netname,
                                                                                                            retinal_patch,
                                                                                                            retinal_patch))
                                    mynet = eval(create_obj)(in_channels=inputchannel, num_classes=classes,
                                                            retinal_H=retinal_patch[0],
                                                            retinal_W=retinal_patch[1],
                                                            image_H=retinal_patch[0],image_W=retinal_patch[1],).cuda()
                                    mynet = train(mynet, train_loader, val_loader, netname)
                                    logger.info('train finished!')

                                    # dataset and test
                                    mynet.load_state_dict(torch.load(path + '/best.pth'))
                                    for sl_test in test_scales:
                                        testset = DatasetCaltech101(root, sl_test, 'test', data_transforms['test'])
                                        test_loader = DataLoader(testset,
                                                                batch_size=batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=8,
                                                                pin_memory=True)           
                                        test_val_ = test_all(test_loader, mynet)
                                        test_acc.append(test_val_.cpu())
                                        logger.info('---------------------')
                                        logger.info(
                                            'train data:[{}], test data:[{}], overall test:[{}]'.format(sl_train,
                                                                                                        sl_test,
                                                                                                        test_val_))
                                        logger.info('---------------------')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc.csv', test_acc, delimiter=',')

                            else:
                                path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname),
                                                    'trainscale_' + str(sl_train))
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    print('path create')
                                path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision))
                                create_obj = compile(netname, 'models.' + netname.split('_')[0] + '.py', 'eval')
                                mynet = eval(create_obj)(in_channels=inputchannel,num_classes=classes).cuda()

                                logger.info('netname:[{}] train dataset:[{}] train data:[{}]'.format(netname, dataname, sl_train))
                                mynet = train(mynet, train_loader, val_loader, netname)
                                logger.info('train finished!')
                                test_acc = []
                                # dataset and test
                                mynet.load_state_dict(torch.load(path + '/best.pth'))
                                for sl_test in test_scales:
                                    testset = DatasetCaltech101(root, sl_test, 'test', data_transforms['test'])
                                    test_loader = DataLoader(testset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             num_workers=8,
                                                             pin_memory=True)

                                    test_val_ = test_all(test_loader,mynet)
                                    test_acc.append(test_val_.cpu())
                                    logger.info('---------------------')
                                    logger.info('train data:[{}], test data:[{}], overall test:[{}]'.format(sl_train,sl_test,test_val_))
                                    logger.info('---------------------')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc.csv', test_acc, delimiter=',')


