import torch
from torch.autograd import Variable
import os
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from data.ImageNet_LS import DatasetImageNet1
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
from helper import AverageMeter, accuracy, adjust_learning_rate
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
def eyemovement_init(mynet,batch_size,test):
    if test:
        h_t = torch.zeros(
            batch_size,
            mynet.hidden_size,
            dtype=torch.float,
            requires_grad=True,
        ).cuda()
        l_t = torch.FloatTensor(batch_size, 2).uniform_(0, 0).cuda()
        l_t.requires_grad = False
    else:
        h_t = torch.zeros(
            batch_size,
            mynet.hidden_size,
            dtype=torch.float,
            requires_grad=True,
        ).cuda()
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).cuda()
        l_t.requires_grad = False
    return h_t, l_t
def scale_init(mynet,batch_size,test):
    if test:
        h_t = torch.zeros(
            batch_size,
            mynet.hidden_size,
            dtype=torch.float,
            requires_grad=True,
        ).cuda()
        l_t = (1/mynet.w_scale) * torch.ones(batch_size, 2).cuda()
        l_t[:,1] = l_t[:,1]*0
        l_t.requires_grad = False
    else:
        h_t = torch.zeros(
            batch_size,
            mynet.hidden_size,
            dtype=torch.float,
            requires_grad=True,
        ).cuda()
        l_t = torch.FloatTensor(batch_size, 2).uniform_(0, 1).cuda()
        l_t.requires_grad = False
    return h_t, l_t

def val(model, val_loader):
    # 习惯性的用法，测试的开始
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(val_loader)):
        img, lab = data
        b, c, h, w = img.shape
        img = img.cuda()
        lab = lab.cuda()
        # no_grad表示不计算梯度
        with torch.no_grad():
            if 'RFscaleyemoment' in netname:
                h_t, s_t = scale_init(model, b, True)
                _, l_t = eyemovement_init(model, b, True)
                for t in range(model.movements):
                    pre, h_t, s_t, l_t, b_t, s_log_pi, l_log_pi = model(img, l_t, s_t, h_t, True)
            elif 'eyemovement' in netname:
                if 'RFscale' in netname:
                    h_t, l_t = scale_init(model, b, True)
                else:
                    h_t, l_t = eyemovement_init(model, b, True)
                for t in range(model.movements):
                    pre, h_t, l_t, b_t, p, codes = model(img, l_t, h_t, True)
            else:
                pre,*_ = model(img)
        prec1, prec5 = accuracy(pre.data, lab.data, topk=(1, 5))
        top1.update(prec1[0], b)
        top5.update(prec5[0], b)
    return top1, top5
def train(mynet, train_loader, val_loader, netname):
    mynet.train()
    print('-----start train model------')
    optimizer = torch.optim.SGD(mynet.parameters(), lr=learning_rate,
                        momentum=0.9,
                        weight_decay=1e-4)
    # 损失函数 交叉熵损失函数
    loss_f1 = torch.nn.CrossEntropyLoss()  # 与pattern的损失函数
    loss_f2 = torch.nn.MSELoss()  # 与pattern的损失函数

    best = 0
    best_epo = 0
    train_accs1 = []
    train_accs5 = []
    train_losss = []
    test_accs1 = []
    test_accs5 = []
    for epo in range(all_epoch):
        mynet.train()
        adjust_learning_rate(optimizer, epo, 0.01)
        currect = 0
        count = 0
        code_all = []
        losss = 0
        
        # 获取一个batch的数据
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            img, lab = data # img 它的大小是 (64,3,224,224)
            scales = train_scales[0]
            b, c, h, w = img.shape
            # 把tensor变成变量
            img = Variable(img).cuda() # .cuda()用GPU
            lab = Variable(lab).cuda()
            loss_other = 0
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            pre, codes = mynet(img)
            loss = loss_f1(pre, lab.long()) + loss_other
            # print(weightss[0])
            
            prec1, prec5 = accuracy(pre.data, lab, topk=(1, 5))
            losses.update(loss.item(), b)
            top1.update(prec1[0], b)
            top5.update(prec5[0], b)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_accs1.append(top1.avg.cpu())
        train_accs5.append(top5.avg.cpu())
        
        test_top1, test_top5 = val(mynet, val_loader)
        test_accs1.append(test_top1.avg.cpu())
        test_accs5.append(test_top5.avg.cpu())
        logger.info('train top1 :[{}] top5 :[{}] loss:[{}] epoch:[{}]'.format(top1.avg, top5.avg, losses.avg, epo))
        logger.info('val top1 :[{}] top5 :[{}] weightss :[{}] epoch:[{}]'.format(test_top1.avg, test_top5.avg, weightss[0].cpu(), epo))
        # 保存参数
        if test_top1.avg > best:
            # 保存参数文件
            torch.save(mynet.state_dict(), path +'/best.pth')
            best = test_top1.avg
            best_epo = epo
            logger.info('!!!! val top1 best:[{}] top5 best:[{}] epoch:[{}]'.format(best, test_top5.avg, best_epo))

    logger.info('----dataname:[{}]----netname:[{}]----not balance:[{}]----seed:[{}]---'.format(netname,dataname,notb,seed))
    logger.info('best:[{}] epoch:[{}]'.format(best, best_epo))
    logger.info('---------------------')
    np.savetxt(path +'/train_acc1.csv', train_accs1, delimiter=',')
    np.savetxt(path +'/train_acc5.csv', train_accs5, delimiter=',')
    # np.savetxt(path +'/train_loss.csv', train_losss, delimiter=',')
    np.savetxt(path +'/val_acc1.csv', test_accs1, delimiter=',')
    np.savetxt(path +'/val_acc5.csv', test_accs5, delimiter=',')
    return mynet
def test_all(val_loader,model):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    count = 0
    correct = 0
    # 加载测试数据
    for index, data in enumerate(tqdm.tqdm(val_loader)):
        img, lab = data
        b, c, h, w = img.shape
        img = img.cuda()
        lab = lab.cuda()
        # no_grad表示不计算梯度
        with torch.no_grad():
            if 'RFscaleyemoment' in netname:
                h_t, s_t = scale_init(model, b, True)
                _, l_t = eyemovement_init(model, b, True)
                for t in range(model.movements):
                    pre, h_t, s_t, l_t, b_t, s_log_pi, l_log_pi = model(img, l_t, s_t, h_t, True)
            elif 'eyemovement' in netname:
                if 'RFscale' in netname:
                    h_t, l_t = scale_init(model, b, True)
                else:
                    h_t, l_t = eyemovement_init(model, b, True)
                for t in range(model.movements):
                    pre, h_t, l_t, b_t, p, codes = model(img, l_t, h_t, True)
            else:
                pre,*_ = model(img)
        prec1, prec5 = accuracy(pre.data, lab.data, topk=(1, 5))
        top1.update(prec1[0], b)
        top5.update(prec5[0], b)
    return top1, top5
if __name__ == '__main__':
    '''
        Init parameters
    '''

    batch_size = 256
    notbs = [False]
    netnames = ['ResNet18_Retinal_frozenRetinal_ImageNet_Pretrain']
    datanames = ['imagenet']
    seeds = [0]
    all_epoch = 100
    load = False
    load_epoch = 0
    use_pretrained = True
    if use_pretrained:
        learning_rate = 0.01
    else:
        learning_rate = 0.01
    if not os.path.exists(os.path.join(os.getcwd(), 'model_zoo')):
        os.makedirs(os.path.join(os.getcwd(), 'model_zoo'))
    retinal_patchs = [(224,224)] 
    train_scales = [[1]]
    test_scales = [[0.25],[0.5],[0.75],[1],[1.25],[1.5],[1.75],[2]]

    for netname in netnames:
        # 创建路径
        path_ = os.getcwd()
        path = os.path.join(path_, 'model_zoo', str(netname), 'all', str(datanames[0]))
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
                    if dataname == 'imagenet':
                        classes = 1000
                        inputchannel = 3
                        # 为了研究不变性，所以不做过多的数据增强，比如放缩旋转等等
                        data_transforms = {
                            'train': transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]),
                            'val': transforms.Compose([
                                # transforms.Resize((224, 224)),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]),
                            'test': transforms.Compose([
                                # transforms.Resize((224, 224)),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]),
                        }

                        for sl_train in train_scales:
                            root = r'/zjh/data/imagenet_scale'

                            trainset = DatasetImageNet1(root, sl_train, 'train', data_transforms['train']).imagenet_val_dataset
                            valset = DatasetImageNet1(root, sl_train, 'val', data_transforms['val']).imagenet_val_dataset
                            train_loader = DataLoader(trainset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=32,
                                                      drop_last=False,
                                                      pin_memory=True)
                            val_loader = DataLoader(valset,
                                                    batch_size=batch_size,
                                                    num_workers=32,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    pin_memory=True)
                            if 'Retinal' in netname:
                                create_obj = compile(netname, 'models.' + netname.split('_')[0] + '.py', 'eval')
                                path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname))
                                test_acc1 = []
                                test_acc5 = []
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

                                    logger.info('train data:[{}], 注意retinal size和inverse的image size一样了'.format(sl_train))
                                    logger.info('netname:[{}], retinal_patch:[{}], image_r_patch:[{}]'.format(netname,
                                                                                                            retinal_patch,
                                                                                                            retinal_patch))
                                    mynet = eval(create_obj)(in_channels=inputchannel, num_classes=classes,
                                                            retinal_H=retinal_patch[0],
                                                            retinal_W=retinal_patch[1],
                                                            image_H=retinal_patch[0],image_W=retinal_patch[1],).cuda()
                                    
                                    if load == True:
                                        mynet.load_state_dict(torch.load(path + '/best.pth'))
                                        all_epoch = all_epoch - load_epoch
                                        print(all_epoch)
                                    
                                    mynet = train(mynet, train_loader, val_loader, netname)
                                    logger.info('train finished!')

                                    # dataset and test
                                    mynet.load_state_dict(torch.load(path + '/best.pth'))
                                    for sl_test in test_scales:
                                        testset = DatasetImageNet1(root, sl_test, 'val', data_transforms['val']).imagenet_val_dataset
                                        test_loader = DataLoader(testset,
                                                                batch_size=batch_size,
                                                                shuffle=False,
                                                                drop_last=False,
                                                                num_workers=32,
                                                                pin_memory=True)           
                                        test_top1, test_top5 = test_all(test_loader, mynet)
                                        test_acc1.append(test_top1.avg.cpu())
                                        test_acc5.append(test_top5.avg.cpu())
                                        logger.info('---------------------')
                                        logger.info(
                                            'train data:[{}], test data:[{}], overall test top1:[{}], overall test top5:[{}]'.format(sl_train,
                                                                                                        sl_test,
                                                                                                        test_top1.avg, test_top5.avg))
                                        logger.info('---------------------')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc1.csv', test_acc1, delimiter=',')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc5.csv', test_acc5, delimiter=',')
                            else:
                                path = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname),
                                                    'trainscale_' + str(sl_train))
                                path_svc = os.path.join(path_, 'model_zoo', str(netname), str(netvision), str(dataname))
                                test_acc1 = []
                                test_acc5 = []
                                if not os.path.exists(path):
                                    os.makedirs(path)
                                    print('path create')

                                create_obj = compile(netname, 'models.' + netname.split('_')[0] + '.py', 'eval')
                                mynet = eval(create_obj)(in_channels=inputchannel,num_classes=classes).cuda()

                                logger.info('netname:[{}] train dataset:[{}] train data:[{}]'.format(netname, dataname, sl_train))
                                if load == True:
                                    mynet.load_state_dict(torch.load(path + '/best.pth'))
                                    all_epoch = all_epoch - load_epoch
                                    print(all_epoch)
                                mynet = train(mynet, train_loader, val_loader, netname)
                                logger.info('train finished!')
                                # dataset and test
                                mynet.load_state_dict(torch.load(path + '/best.pth'))
                                for sl_test in test_scales:
                                    testset = DatasetImageNet1(root, sl_test, 'val', data_transforms['val']).imagenet_val_dataset
                                    test_loader = DataLoader(testset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             drop_last=False,
                                                             num_workers=32,
                                                             pin_memory=True)

                                    test_top1, test_top5 = test_all(test_loader, mynet)
                                    test_acc1.append(test_top1.avg.cpu())
                                    test_acc5.append(test_top5.avg.cpu())
                                    logger.info('---------------------')
                                    logger.info(
                                        'train data:[{}], test data:[{}], overall test top1:[{}], overall test top5:[{}]'.format(sl_train,
                                                                                                    sl_test,
                                                                                                    test_top1.avg, test_top5.avg))
                                    logger.info('---------------------')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc1.csv', test_acc1, delimiter=',')
                                np.savetxt(path_svc + '/'+str(sl_train)+'_test_acc5.csv', test_acc5, delimiter=',')




