##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.utils import make_grid
import torchvision

from torch.optim import Adam
import pytorch_lightning as pl
import numpy as np

from PIL import Image, ImageDraw
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm



##################################################
# Hands Segmentor
##################################################

class HandSegModel(pl.LightningModule):
    """
    Based on PyTorch DeepLab model for semantic segmentation.
    """
    def __init__(self, pretrained=False, lr=1e-4):
        super().__init__()
        self.deeplab = self._get_deeplab(pretrained=pretrained, num_classes=2)
        self.denorm_image_for_tb_log = None # For tensorboard logging
        self.lr = lr
        if pretrained:
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]) 
            self.denorm_image_for_tb_log = Denorm(mean, std)

    def _get_deeplab(self, pretrained=False, num_classes=2):
        """
        Get the PyTorch DeepLab model architecture.
        """
        deeplab = models.segmentation.deeplabv3_resnet50(
            pretrained=False,
            num_classes=num_classes
        )
        if pretrained:
            deeplab_21 = models.segmentation.deeplabv3_resnet50(
                pretrained=True,
                progress=True,
                num_classes=21
            )
            for c1, c2 in zip(deeplab.children(), deeplab_21.children()):
                for p1, p2 in zip(c1.parameters(), c2.parameters()):
                    if p1.shape == p2.shape:
                        p1.data = p2.data
        return deeplab

    def forward(self, x):
        return self.deeplab(x)['out']

    def training_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        self.log('train_bce', loss, prog_bar=True)
        self.log('train_mIoU', miou, prog_bar=True)
        return loss

    def validation_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        # Cache
        self.log('validation_bce', loss, prog_bar=True)
        self.log('validation_mIoU', miou, prog_bar=True)
        if idx_batch == 0:
            tb_log = self.trainer.logger.experiment
            if self.denorm_image_for_tb_log:
                x = self.denorm_image_for_tb_log(x)
            x_grid = make_grid(x[:16], nrow=4)
            y_hat_grid = make_grid(y_hat[:16].argmax(1).unsqueeze(1), nrow=4)[0:1]
            tb_log.add_image('validation_images', x_grid.cpu().numpy())
            tb_log.add_image('validation_preds', y_hat_grid.cpu().numpy())
        return loss

    def test_step(self, batch, idx_batch):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        y_hat = F.softmax(logits, 1).detach()
        miou = meanIoU(y_hat, y.argmax(1))

        # Cache
        self.log('test_bce', loss, prog_bar=True)
        self.log('test_mIoU', miou, prog_bar=True)
        if idx_batch == 0:
            tb_log = self.trainer.logger.experiment
            if self.denorm_image_for_tb_log:
                x = self.denorm_image_for_tb_log(x)
            x_grid = make_grid(x[:16], nrow=4)
            y_hat_grid = make_grid(y_hat[:16].argmax(1).unsqueeze(1), nrow=4)[0:1]
            tb_log.add_image('test_images', x_grid.cpu().numpy())
            tb_log.add_image('test_preds', y_hat_grid.cpu().numpy())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def set_denorm_fn(self, denorm_fn):
        self.denorm_image_for_tb_log = denorm_fn

##################################################
# metric
##################################################  

def meanIoU(logits, labels):
    """
    Computes the mean intersection over union (mIoU).
    
    Args:
        logits: tensor of shape [bs, c, h, w].
        labels: tensor of shape [bs, h, w].
    
    Output:
        scalar.
    """
    num_classes = logits.shape[1]
    preds = F.softmax(logits, 1)
    preds_oh = F.one_hot(preds.argmax(1), num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w] 
    labels_oh = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).to(torch.float32) # [bs, c, h, w]
    tps = (preds_oh * labels_oh).sum(-1).sum(-1) # true positives [bs, c]
    fps = (preds_oh * (1 - labels_oh)).sum(-1).sum(-1) # false positives [bs, c]
    fns = ((1 - preds_oh) * labels_oh).sum(-1).sum(-1) # false negatives [bs, c]
    iou = tps / (tps + fps + fns + 1e-8) # [bs, c]
    return iou.mean(-1).mean(0)

def get_image_transform(args):
    """
    build the image transforms.
    """
    image_transform = None
    image_transform = transforms.Compose([
            transforms.Resize((args["height"], args["width"])),
            transforms.ToTensor(),
            lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return image_transform


# Utils
def show_sample(image, mask=None, alpha=0.7):
    print('Image shape:', image.shape)
    plt.imshow(image.permute(1, 2, 0))
    if mask is not None:
        print('Mask shape:', mask.shape)

        plt.imshow(mask[0], alpha=alpha)
    plt.show()

def show_samples(images, masks=None, alpha=0.7, nrow=4):
    print('Images shape:', images.shape)
    if masks is not None:
        print('Masks shape:', masks.shape)
        B, C, H, W = images.shape
        col = [0.2, 0.3, 0.8]
        col = torch.tensor(col).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(B, 1, H, W)
        images = torch.where(masks.repeat(1, 3, 1, 1) > 0, 
                             alpha * col + (1 - alpha) * images, images)
    image_grid = make_grid(images, nrow=nrow, padding=0)
    plt.figure(figsize=(15, 15))
    plt.imshow(image_grid.permute(1, 2, 0), aspect='auto')
    plt.axis(False)
    plt.show()

class Denorm(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array([0.0, 0.0, 0.0]) if mean is None else mean
        self.std = np.array([1.0, 1.0, 1.0]) if std is None else std

    def __call__(self, x):
        """
        Denormalize the image.

        Args:
            x: tensor of shape [bs, c, h, w].

        Output:
            x_denorm: tensor of shape [bs, c, h, w].
        """
        denorm_fn = transforms.Normalize(mean=- self.mean / (self.std + 1e-8), std=1.0 / (self.std + 1e-8))
        x_denorm = []
        for x_i in x:
            x_denorm += [denorm_fn(x_i)]
        x_denorm = torch.stack(x_denorm, 0)
        return x_denorm


class EgoHandsDataset(Dataset):
    """
        EgoHands dataset from http://vision.soic.indiana.edu/projects/egohands.
        Images and masks of dims [720, 1280].
    """
    def __init__(self, data_base_path, partition, image_transform=None, 
                 mask_transform=None, seed=1234, frame_tmpl='frame_{:04d}.jpg', 
                 mask_shape=None):
        super(EgoHandsDataset, self).__init__()
        self.data_base_path = data_base_path
        self.partition = partition
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.seed = seed
        self.frame_tmpl = frame_tmpl
        self.metadata = scipy.io.loadmat(os.path.join(self.data_base_path, 'metadata.mat'))
        self.mask_shape = mask_shape
        self.image_paths, self.mask_poly = self._get_paths()

    def _compute_mask(self, polygons, height, width):
        mask = Image.new('L', (width, height), 0)
        for poly in polygons:
            ImageDraw.Draw(mask).polygon(poly, outline=255, fill=255)
        return mask

    def _get_paths(self):

        annotations = self.metadata['video'][0] # 48 annotations (of the 48 videos)
        image_paths = []
        masks_poly = []
        for x in annotations:
            x = list(x)
            video_id, _, _, _, _, _, labeled_frames = x 
            video_id = video_id[0]
            labeled_frames = labeled_frames[0]

            # Get frame annotation
            for frame_ann in labeled_frames:
                frame_id = frame_ann[0].reshape(-1)[0]
                polygons = []
                for idx, ll in enumerate(frame_ann):
                    if (idx > 0) and len(ll) > 0:
                        p = [tuple(pp) for pp in ll]
                        polygons += [p]
                masks_poly += [polygons]

                image_path = os.path.join(self.data_base_path, '_LABELLED_SAMPLES', video_id, self.frame_tmpl.format(frame_id))
                image_paths += [image_path]

        # Split data
        num_samples = len(image_paths)
        num_train = int(np.round(0.6 * num_samples))
        num_validation = int(np.round(0.2 * num_samples))
        num_test = int(np.round(0.2 * num_samples))
        idxs = np.arange(num_samples)
        np.random.seed(self.seed)
        np.random.shuffle(idxs)
        idxs_train = idxs[:num_train]
        idxs_validation = idxs[num_train:num_train + num_validation]
        idxs_test = idxs[-num_test:]
        if self.partition in ['train', 'training']:
            image_paths = [image_paths[i] for i in idxs_train]
            masks_poly = [masks_poly[i] for i in idxs_train]
        elif self.partition in ['val', 'validation', 'validating']:
            image_paths = [image_paths[i] for i in idxs_validation]
            masks_poly = [masks_poly[i] for i in idxs_validation]
        elif self.partition in ['test', 'testing']:
            image_paths = [image_paths[i] for i in idxs_test]
            masks_poly = [masks_poly[i] for i in idxs_test]
        else:
            raise Exception(f'Error. Partition "{self.partition}" is not supported.')
        return image_paths, masks_poly

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image and mask
        image = Image.open(self.image_paths[idx])
        if self.mask_shape is None:
            w, h = image.size
        else:
            h, w = self.mask_shape
        mask = self._compute_mask(self.mask_poly[idx], h, w)

        # Transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return image, mask


def get_dataloader(data_base_path, partition, datasets, image_transform=None,
                   mask_transform=None, batch_size=32, num_workers=0, pin_memory=True, shuffle=False):
    """
    Get the dataloader.

    Args:
        data_base_path: string where the data are stored.
        partition: string in ['train', 'validation', 'test'].
        datasets: list of strings for selecting the sounrce of the data.
        image_transforms: transform applied to the image.
        mask_transform: transform applied to the mask.
        batch_size: integer that specifies the batch size.
        num_workers: the number of workers.
        pin_memory: boolean.
    
    Output:
        dl: the dataloader (PyTorch DataLoader).
    """
    ds_list = []

    if 'eh' in datasets:
        tranform = transforms.ToTensor()
        ds_eh = EgoHandsDataset(
            data_base_path=os.path.join(data_base_path, 'egohands_data'),
            partition=partition,
            image_transform=tranform if image_transform is None else image_transform,
            mask_transform=tranform if mask_transform is None else mask_transform,
        )
        ds_list += [ds_eh]

    # Concatenate datasets
    ds_cat = ConcatDataset(ds_list)
    dl = DataLoader(ds_cat, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=shuffle)
    return dl

def get_args():
    """
    read the input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', 
                        help='Mode of the program. Can be "train", "test" or "predict".')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs used for the training.')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size.')
    parser.add_argument('--datasets', type=str, default='eh', help='List of datasets to use. For now only egohand')
    parser.add_argument('--height', type=int, default=256, help='The height of the input image.')
    parser.add_argument('--width', type=int, default=256, help='THe width of the input image.')
    parser.add_argument('--data_base_path', type=str, required=True, help='The path of the input dataset.')
    parser.add_argument('--model_pretrained', default=False, action='store_true', 
                        help='Load the PyTorch pretrained model.')
    parser.add_argument('--model_checkpoint', type=str, default='', help='The model checkpoint to load.')
    parser.add_argument('--lr', type=float, default=3e-4, help='The learning rate.')
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    return args

def get_model(args):
    """
    build the model.
    """
    model_args = {
        'pretrained': args["model_pretrained"],
        'lr': args["lr"],
    }
    model = HandSegModel(**model_args)
    if len(args["model_checkpoint"]) > 0:
        model = model.load_from_checkpoint(args["model_checkpoint"], **model_args)
        print(f'Loaded checkpoint from {args["model_checkpoint"]}.')
    return model

def get_image_transform(args):
    """
    build the image transforms.
    """
    image_transform = transforms.Compose([
        transforms.Resize((args["height"], args["width"])),
        transforms.ToTensor(),
        lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return image_transform

def get_dataloaders(args):
    """
    build the dataloaders.
    """
    image_transform = get_image_transform(args)
    mask_transform = transforms.Compose([
        transforms.Resize((args["height"], args["width"])),
        transforms.ToTensor(),
        lambda m: torch.where(m > 0, torch.ones_like(m), torch.zeros_like(m)),
        lambda m: F.one_hot(m[0].to(torch.int64), 2).permute(2, 0, 1).to(torch.float32),
    ])
    dl_args = {
        'data_base_path': args["data_base_path"],
        'datasets': args["datasets"].split(' '),
        'image_transform': image_transform,
        'mask_transform': mask_transform,
        'batch_size': args["batch_size"],
    }
    dl_train = get_dataloader(**dl_args, partition='train', shuffle=True)
    dl_validation = get_dataloader(**dl_args, partition='validation', shuffle=False)
    dl_test = get_dataloader(**dl_args, partition='test', shuffle=False)
    dls = {
        'train': dl_train,
        'validation': dl_validation,
        'test': dl_test,
    }
    return dls

def get_predict_dataset(args):
    """
    """
    image_paths = sorted(os.listdir(args["data_base_path"]))
    image_paths = [os.path.join(args["data_base_path"], f) for f in image_paths]
    print(f'Found {len(image_paths)} in {args["data_base_path"]}.')
    transform = get_image_transform(args)
    class ImageDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            super(ImageDataset, self).__init__()
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            if self.transform is not None:
                image = self.transform(image)
            return image, image_path
    return ImageDataset(image_paths, transform=transform)

def training(args):
    """
    main function.
    """

    # Model
    model = get_model(args)
    print("model built")
    print(args)
    # Mode
    if args["mode"] == 'train':
        dls = get_dataloaders(args) # Dataloader
        print(dls)
        print("got dls ok")
        trainer = pl.Trainer(max_epochs=args["epochs"])
        trainer.fit(model, dls['train'], dls['validation'])
    else:
        raise Exception(f'Error. Mode "{args["mode"]}" is not supported.')
    

def handseg_predict(picture):
   """
   Load model and predict segmentation picture, return the picture and bounding box of the hand.
   """
   args = {
    "mode": "predict",
    "epochs": 50,
    "batch_size": 16,
    "gpus": 1,
    "datasets": "eh",
    "height": 256,
    "width": 256,
    "data_base_path": "/image_path",
    "model_pretrained": True,
    "model_checkpoint": "checkpoint/checkpoint.ckpt",
    "lr": 0.0003,
    "in_channels": 3
   }
   # Model
   model = get_model(args)
   print("model built")
   # Save prediction
   _ = model.eval()
   device = next(model.parameters()).device
   H, W = picture.shape[-2:]
   x = transforms.Resize((256, 256))(picture)
   x = x.unsqueeze(0).to(device)
   logits = model(x).detach().cpu()
   preds = F.softmax(logits, 1).argmax(1)[0] * 255 # [h, w]
   preds_array = preds
   preds = Image.fromarray(preds.numpy().astype(np.uint8), 'L')
   preds = preds.resize((W, H))
   # creates contours based on predicted mask
   # get boundary locations of non zero pixels in the image
   nz = np.nonzero(preds)
   # get the min and max of the non zero pixels
   top = np.min(nz[0])
   bottom = np.max(nz[0])
   left = np.min(nz[1])
   right = np.max(nz[1])
   return picture, [top, bottom, left, right]

