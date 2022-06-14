import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
from dataloaders import dataset
import models
from metrics import AverageMeter, Result
from evaluate import Evaluater
import criteria
import utils
max_depths = {
    'kitti': 80.0,
    'nyu_reduced' : 10.0,
}

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        evaluation_module = Evaluater(args)
        evaluation_module.evaluate()

    # create new model
    if args.train:
        train_loader = dataset.get_dataloader(args.data,
                                                 path=args.data_path,
                                                 split='train',
                                                 augmentation=args.eval_mode,
                                                 batch_size=args.batch_size,
                                                 resolution=args.resolution,
                                                 workers=args.workers)
        val_loader = dataset.get_dataloader(args.data,
                                                path=args.data_path,
                                                split='val',
                                                augmentation=args.eval_mode,
                                                batch_size=args.batch_size,
                                                resolution=args.resolution,
                                                workers=args.workers)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        # model = models.MobileNetSkipAdd(output_size=(224,224))
        # model = models.MobileNetV2SkipAdd(output_size=(224,224))
        model = models.MobileNetV2SkipConcat(output_size=(224,224))
        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        model = model.cuda()

    # define loss function (criterion) and optimizer
        if args.criterion == 'l2':
            criterion = criteria.MaskedMSELoss().cuda()
        elif args.criterion == 'l1':
            criterion = criteria.MaskedL1Loss().cuda()

        # create results folder, if not already exists
        output_directory = utils.get_output_directory(args)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        train_csv = os.path.join(output_directory, 'train.csv')
        test_csv = os.path.join(output_directory, 'test.csv')
        best_txt = os.path.join(output_directory, 'best.txt')

        # create new csv files with only header
        if not args.resume:
            with open(train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(test_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        for epoch in range(start_epoch, args.epochs):
            utils.adjust_learning_rate(optimizer, epoch, args.lr)
            train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
            result = validate(val_loader, model, epoch)

            # remember best rmse and save checkpoint
            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model': model,
                'best_result': best_result,
                'optimizer' : optimizer,
            }, is_best, epoch, output_directory)

def unpack_and_move(data):
        if isinstance(data, (tuple, list)):
            image = data[0]
            gt = data[1]
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image']
            gt = data['depth']
            return image, gt
        print('Type not supported')

def train(train_loader, model, criterion, optimizer, epoch):
    # print('train')
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()
    for i, data in enumerate(train_loader):
        input, target = unpack_and_move(data)
        input, target = input.cuda(), target.cuda()
        # print(target.shape)
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})

def inverse_depth_norm(depth):
        zero_mask = depth == 0.0
        maxDepth = max_depths[args.data]
        depth = maxDepth / depth
        depth = torch.clamp(depth, maxDepth / 100, maxDepth)
        depth[zero_mask] = 0.0
        return depth

def depth_norm(depth):
        zero_mask = depth == 0.0
        maxDepth = max_depths[args.data]
        depth = torch.clamp(depth, maxDepth / 100, maxDepth)
        depth = maxDepth / depth
        depth[zero_mask] = 0.0
        return depth

def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input,target  = unpack_and_move(data)
            input, target = input.cuda(), target.cuda()
            data_time = time.time() - end

            # compute output
            end = time.time()
            inv_pred = model(input)
            pred = inverse_depth_norm(inv_pred)
            gpu_time = time.time() - end
            # measure accuracy and record loss
            result = Result()
            result.evaluate(pred.data, target.data)
            average_meter.update(result, gpu_time, data_time, input.size(0))
            end = time.time()

            if (i+1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg

if __name__ == '__main__':
    main()