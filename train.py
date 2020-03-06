import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import *
from torch.utils.data import DataLoader, random_split

from PIL import Image

train_dir_img = 'data/imgs/'
train_dir_mask = 'data/masks/'
n_train = 2000

val_dir_img = 'data/val_imgs/'
val_dir_mask = 'data/val_masks/'
n_val = 150

test_dir_img = 'data/test_imgs/'
test_dir_mask = 'data/test_masks/'
n_val = 600


dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=120,
              batch_size=16,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

#    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    train_dataset = Dataset_Aug(train_dir_img, train_dir_mask)
    val_dataset = Dataset_No_Aug(val_dir_img, val_dir_mask)
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = random_split(dataset, [n_train, n_val])
    #train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    #val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_Epoch_{epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
        #Training size:   {n_train}
        #Validation size: {n_val}

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                reshaped_std = np.expand_dims(np.array([0.229, 0.224, 0.225]),1)
                reshaped_std = np.expand_dims(reshaped_std,1)
                reshaped_mean = np.expand_dims(np.array([0.485, 0.456, 0.406]),1)
                reshaped_mean = np.expand_dims(reshaped_mean,1)
                image_log = (imgs.numpy()*reshaped_std+reshaped_mean)*255
                rescaled = (imgs.numpy()[0]*reshaped_std+reshaped_mean)*255
                rescaled = np.swapaxes(rescaled,0,1)
                rescaled = np.swapaxes(rescaled,1,2)
                im = Image.fromarray(np.uint8(rescaled))
                im.save('input_image.png')
                #rescaled = true_masks.numpy()[0,0]*255
                #im = Image.fromarray(rescaled.astype(np.uint8))
                #im.save('input_label.png')
                #print(imgs.shape)
                #print(torch.max(imgs))
                #print(torch.min(imgs))
                #print(true_masks.shape)
                #print(true_masks)
                #print(true_masks[0,0,100])
                #print(true_masks[0,0,:,100])
                #print(torch.max(true_masks))
                #print(torch.min(true_masks))
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(train_dataset) // (1 * batch_size)) == 0:
                    val_score, jac = eval_net(net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)
                        writer.add_scalar('Jac/val', jac, global_step)

                    writer.add_images('images', imgs, global_step)
                    writer.add_images('yukun_images', image_log, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
