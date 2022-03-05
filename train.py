#-*- coding:utf-8 _*-
from __future__ import print_function
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import numpy as np
from args import args
from build_net import make_model
from transform import get_transforms
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint
state = {k: v for k, v in args._get_kwargs()}     #得到paser中的所有的参数及其默认值
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id#指定显卡跑
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)#为CPU设置种子用于生成随机数
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)#为当前GPU设置随机种子；
best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # 从epoch 0或最后一个检查点epoch开始

    if not os.path.isdir(args.checkpoint):#是否是目录
        mkdir_p(args.checkpoint)#创建目录，/data0/search/qlmx/clover/garbage/res_16_288_last1'

    # Data
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)


    print('==> Preparing dataset %s' % args.trainroot)#'--trainroot', default='data/new_shu_label.txt'
    trainset = dataset.Dataset(root=args.trainroot, transform=transform['val_train'])
    train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    #已有的数据读取接口的输入按照batch size封装成Tensor

    valset = dataset.TestDataset(root=args.valroot, transform=transform['val_test'])#valroot，data/val1.txt
    val_loader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = make_model(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        #会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总。
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()#交叉熵损失函数
    optimizer = get_optimizer(model, args)#优化函数SGD
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    #epoch训练次数进行学习率调整


    # Resume
    title = 'ImageNet-' + args.arch   #resnext101_32x8d_wsl
    if args.resume:
        # 存 checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])#参数拷贝
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc, train_5 = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_5 = test(val_loader, model, criterion, epoch, use_cuda)
        scheduler.step(test_loss)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])
        print('train_loss:%f, val_loss:%f, train_acc:%f, train_5:%f, val_acc:%f, val_5:%f' % (train_loss, test_loss, train_acc, train_5, test_acc, test_5))

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if len(args.gpu_id) > 1:
            save_checkpoint({
                'fold': 0,
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'train_acc': train_acc,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, single=True, checkpoint=args.checkpoint)

        else:
            save_checkpoint({
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc':train_acc,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, single=True, checkpoint=args.checkpoint)
                                         #将信息保存起来
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()#启用batch normalization和drop out。

    batch_time = AverageMeter()#平均值
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #计算数据加载的时间
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        #.autograd实现求导，.Variable把tensor转换并将所有的计算节点都连接起来，最后进行误差反向传递


        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()#沿用batch normalization的值，并不使用drop out

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


if __name__ == '__main__':
    main()

