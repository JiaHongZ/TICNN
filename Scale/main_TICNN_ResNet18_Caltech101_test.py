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


