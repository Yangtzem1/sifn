import os
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import config
from models import sifn
from datasets import get_trainval_datasets
from utils import FLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment

assert torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = False
cross_entropy_loss = nn.CrossEntropyLoss()
a_loss = nn.MSELoss(reduce=False, size_average=False)
f_loss = FLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
c_metric = TopKAccuracyMetric(topk=(1, 5))
d_metric = TopKAccuracyMetric(topk=(1, 5))

def main():
    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0
    net = sifn(num_classes=num_classes, M=config.num_parts, net=config.net, pretrained=True)
    feature_center = torch.zeros(num_classes, 1080).to(device)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    ##################################
    # TRAINING
    ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              pbar=pbar)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net, feature_center=feature_center)
        pbar.close()


def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    c_metric.reset()
    d_metric.reset()

    # begin training
    start_time = time.time()
    net.train()
    for i, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        X = X.to(device)
        y = y.to(device)

        ##################################
        # Raw Image
        ##################################
        # raw images forward
        y_pred_raw, feature_matrix, attention_map, agg_sampler = net(X)

        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)
        with torch.no_grad():
            c_images = batch_augment(X, attention_map[:, :1, :, :], mode='c', theta=(0.4, 0.6), padding_ratio=0.1)

        y_pred_c, _, _, agg_sampler = net(c_images)

        with torch.no_grad():
            d_images = batch_augment(X, attention_map[:, 1:, :, :], mode='d', theta=(0.2, 0.5))

        y_pred_d, _, _ = net(d_images)

        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_c, y) / 3. + \
                     cross_entropy_loss(y_pred_d, y) / 3. + \
                     f_loss(feature_matrix, feature_center_batch) + \
                     a_loss(attention_map, agg_sampler)

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            #epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_c_acc = c_metric(y_pred_c, y)
            epoch_d_acc = d_metric(y_pred_d, y)

        # end of this batch
        batch_info = 'Loss {:.4f}, C Acc ({:.2f}, {:.2f}), D Acc ({:.2f}, {:.2f})'.format(
            epoch_loss, 
            epoch_c_acc[0], epoch_c_acc[1], epoch_d_acc[0], epoch_d_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_c_{}'.format(c_metric.name)] = epoch_c_acc
    logs['train_d_{}'.format(d_metric.name)] = epoch_d_acc
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))

def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(data_loader):
            # obtain data
            X = X.to(device)
            y = y.to(device)

            y_pred_raw, _, attention_map,  agg_sampler= net(X)

            c_images = batch_augment(X, attention_map, mode='c', theta=0.1, padding_ratio=0.05)
            y_pred_c, _, _ = net(c_images)

            y_pred = (y_pred_raw + y_pred_c) / 2.

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)

    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


if __name__ == '__main__':
    main()
