'''
Implementation of 'Temporal-attentive Covariance Pooling Networks for Action Recognition'
Authors: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu and Peihua Li.
'''

import pdb
import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from functions import *

from ops.dataset import TSNDataSet
from ops.models import TSN
from opts import parser
from ops.transforms import *
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.keys_mapping import *
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

import datetime

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name += '_'.join(
        ['TSN', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.TCP :
        args.store_name += '_TCP'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    # assert args.TCP_level in ['video', 'frame']

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                TCP=args.TCP, TCP_level=args.TCP_level,
                TCP_dim=args.TCP_dim,
                pretrained_dim=args.pretrained_dim,
                TCP_sp=args.TCP_sp, TCP_ch=args.TCP_ch, TCP_1d=args.TCP_1d)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies(args)
    train_augmentation = model.get_augmentation( flip=False if 'something' in args.dataset or
                                                              'jester' in args.dataset or
                                                              'somethingv2' in args.dataset or
                                                               'diving48' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.TCP and args.load_GCP_from:
        print(("=> loading pretrained GCP model from '{}'".format(args.load_GCP_from)))
        sd = torch.load(args.load_GCP_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        sd = adaptive_mapping_pretrained_keys(sd, model_dict, args)

        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load:')
        for di in set_diff:
            print(di)
        assert [di for di in set_diff if 'TCP' in di or 'fc' in di], "invalid parameter name"

        print('=> New dataset, do not load fc weights')
        sd = {k: v for k, v in sd.items() if 'fc' not in k}

        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}

        # adaptively load pretrained dimension reduction layers
        if args.TCP_dim != args.pretrained_dim :
            # REMOVE LAST DIMENSION REDUCTION LAYERS IN GCP
            sd = {k: v for k, v in sd.items() if 'reduce2' not in k and 'reduce_bn2' not in k}
            print('-'*20)
            print('=> remove LAST dimension reduction layers in GCP')
            print('-' * 20)
        else :
            print('====load ALL pretrained dimension reduction layer for TCP====')

        model_dict.update(sd)
        model.load_state_dict(model_dict)


    if  (not args.keys_old) and [ii for ii in model.state_dict().keys() if 'iSQRT' in ii ]:
        args.keys_old = True

    if args.resume or args.load_TCP_from: #load trained TCP model
        load_path = args.resume if args.resume else args.load_TCP_from
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        assert os.path.isfile(load_path), "=> no checkpoint found at %s"%load_path
        print(("=> loading checkpoint '{}'".format(load_path)))

        checkpoint = torch.load(load_path)
        if args.resume :
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
        elif  [ii for ii in checkpoint['state_dict'].keys() if 'iSQRT' in ii ]:
             # load pretrained old keys TCP model
             args.keys_old = True

        if not args.keys_old :
            if args.resume: #not load optimizer for old keys model
                optimizer.load_state_dict(checkpoint['optimizer'])
        else :#mapping old style keys to the new
            checkpoint['state_dict'] = adaptive_mapping_old_style_keys(
                checkpoint['state_dict'], model.state_dict().keys(), args)

        if args.load_TCP_from:
            print("==> new dataset, remove pretrained fc weights")
            sd = checkpoint['state_dict']
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
            if args.modality == 'Flow' and 'Flow' not in args.tune_from:
                sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
            model_dict = model.state_dict()
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint evaluate:'{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_process = [
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['resnet152','BNInception', 'InceptionV3'])),
                   ]
    if 'resnet152' not in args.arch:
        train_process.append(normalize)

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose(train_process), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU


    val_process = [
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['resnet152','BNInception', 'InceptionV3'])),
                   ]
    if  'resnet152' not in args.arch :
        val_process.append(normalize)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose(val_process), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'a+')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    for group in policies:
        policy_i = ('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult']))
        print(policy_i)

        if log_training is not None :
            log_training.write(policy_i + '\n')
            log_training.flush()

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return


    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    # make directory for storing checkpoint files
    args.store_name = os.path.join('results', args.store_name)
    if not os.path.exists('results') :
        os.mkdir('results')
    if not os.path.exists(args.store_name) :
        os.mkdir(args.store_name)
    assert (not args.resume) or os.path.exists(os.path.join(args.store_name,'stats.mat')), \
        'not found stats.mat when resuming'


    stats_ = stats(args.store_name, args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        trainObj, top1, top5 = train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        # evaluate on validation set
        valObj, prec1, prec5 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

        # update stats
        stats_._update(trainObj, top1, top5, valObj, prec1, prec5)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = []

        tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
        print(output_best)
        log_training.write(output_best + '\n')
        log_training.flush()

        filename.append(os.path.join(args.store_name, 'net-epoch-%s.pth.tar' % (epoch + 1)))
        filename.append(os.path.join(args.store_name, 'model_best.pth.tar'))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename)
        plot_curve(stats_, os.path.join(args.store_name), True)
        data = stats_
        sio.savemat(os.path.join(args.store_name, 'stats.mat'), {'data': data})



def print_requires_grad(model):
    """
    print all the module requires_grad=False
    """
    for name, m in model.named_parameters():
            print(name + ' requires_grad = ' + format(m.requires_grad))


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            time_stamp = datetime.datetime.now()

            time_stamp = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')

            output = (time_stamp + ' Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[0]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(filename[0]))
    if is_best:
        shutil.copyfile(os.path.join(filename[0]), os.path.join(filename[1]))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log,
                    os.path.join(args.root_log, args.store_name),
                    ]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()