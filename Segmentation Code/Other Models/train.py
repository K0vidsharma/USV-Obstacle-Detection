import os
import json
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import segmentation_models_pytorch as smp
from utils import get_splits, get_loaders, evaluation, train_fn, get_dict, save_checkpoint
from dataset import WaterEdgeDataset
from unet import UNET

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
PIN_MEMORY = True
LOAD_MODEL = False
IMAGE_DIR = "/home/rp273/Kovid/Code/data/images/"
MASK_DIR = "/home/rp273/Kovid/Code/data/masks/"
train_split_pth = "/home/rp273/Kovid/Code/data/image_splits/train_images.txt"
val_split_pth = "/home/rp273/Kovid/Code/data/image_splits/val_images.txt"
test_split_pth = "/home/rp273/Kovid/Code/data/image_splits/test_images.txt"

train_split, val_split, test_split = get_splits(train_split_pth, val_split_pth, test_split_pth)
train_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ],
)

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


def train(model_name: str):
    model = None
    if model_name == "res-unet++":
        model = smp.UnetPlusPlus(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "res-unet":
        model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "unet":
        model = UNET()
    elif model_name == "dense-unet":
        model = smp.Unet(
            encoder_name='densenet169',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "eff-unet":
        model = smp.Unet(
            encoder_name="efficientnet-b2",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "mobnet-unet":
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "mobileone-unet":
        model = smp.Unet(
            encoder_name="mobileone_s4",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "dense-unet++":
        model = smp.UnetPlusPlus(
            encoder_name='densenet169',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "eff-unet++":
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b2",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "mobnet-unet++":
        model = smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "mobileone-unet++":
        model = smp.UnetPlusPlus(
            encoder_name="mobileone_s4",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "linknet-resnet18":
        model = smp.Linknet(
            encoder_name="resnet18",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == "pspnet-resnet18":
        model = smp.PSPNet(
            encoder_name="resnet18",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == "manet-resnet18":
        model = smp.MAnet(
            encoder_name="resnet18",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    elif model_name == "pspnet-effnet-b4":
        model = smp.PSPNet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == "linknet-effnet-b4":
        model = smp.Linknet(
            encoder_name="efficientnet-b4",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )
    elif model_name == "manet-effnet-b4":
        model = smp.MAnet(
            encoder_name="efficientnet-b4",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=[512, 256, 128, 64, 32]
        )
    else:
        print("No models found")

    if model is not None:
        model.to(DEVICE)
        loss_fn = smp.losses.DiceLoss(mode='binary')
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1)
        train_loader, val_loader = get_loaders(
            WaterEdgeDataset,
            IMAGE_DIR,
            MASK_DIR,
            train_split,
            val_split,
            BATCH_SIZE,
            train_transforms,
            val_transforms,
            NUM_WORKERS,
            PIN_MEMORY
        )
        ret_dict = get_dict(model_name)
        for epoch in range(NUM_EPOCHS):
            print("Epoch: ", epoch+1)
            # evaluation(val_loader, model, DEVICE, results)
            train_fn(train_loader, model, optimizer, loss_fn, DEVICE, ret_dict["results"])
            scheduler.step()
            evaluation(val_loader, model, DEVICE, ret_dict["results"])
            if (epoch+1) % 5 == 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_checkpoint(checkpoint, filename=f"./checkpoints/{model_name}_checkpoint.pth.tar")

    return ret_dict


if __name__ == "__main__":
    lis_models = [
        "linknet-effnet-b4"
    ]

    for model in lis_models:
        ret_dict = train(model)
        with open(f"./Results/{model}_results.json", "w") as outfile:
            json.dump(ret_dict, outfile)
