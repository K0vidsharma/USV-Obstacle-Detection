from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader


def train_fn(loader, model, optimizer, loss_fn, device, results):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        results["loss"].append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Updating tqdm loop
        loop.set_postfix(loss=loss.item())


def evaluation(loader, model, device, metric_dict):
    dice_score = 0
    acc_score = 0
    iou_score = 0
    rec_score = 0
    prec_score = 0
    with torch.inference_mode():
        for x, y in loader:
            model.eval()
            x, y = x.to(device), y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds.type(torch.LongTensor).to(device),
                y.type(torch.LongTensor).to(device),
                mode="binary",
                threshold=0.5
            )
            tp, fp, fn, tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()
            dice_score += (2. * tp) / (2. * tp + fp + fn + 1e-8)
            iou_score += tp / (tp + fp + fn + 1e-8)
            acc_score += (tp + tn) / (tp + tn + fp + fn + 1e-8)
            rec_score += tp / (tp + fn + 1e-8)
            prec_score += tp / (tp + fp + 1e-8)

    dice_score = (dice_score / len(loader)).item()
    iou_score = (iou_score / len(loader)).item()
    acc_score = (acc_score / len(loader)).item()
    prec_score = (prec_score / len(loader)).item()
    rec_score = (rec_score / len(loader)).item()

    print(f"Dice Score: {dice_score}")
    print(f"IOU Score: {iou_score}")

    metric_dict["dice_list"].append(dice_score)
    metric_dict["iou_list"].append(iou_score)
    metric_dict["acc_list"].append(acc_score)
    metric_dict["rec_list"].append(rec_score)
    metric_dict["prec_list"].append(prec_score)
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model: torch.nn.Module):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
        dataset,
        img_dir,
        mask_dir,
        train_splits,
        val_splits,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory=True
):
    train_ds = dataset(image_dir=img_dir,
                       mask_dir=mask_dir,
                       images=train_splits,
                       transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = dataset(image_dir=img_dir,
                     mask_dir=mask_dir,
                     images=val_splits,
                     transform=val_transform)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def get_splits(train_split_pth, val_split_pth, test_split_pth):
    with open(train_split_pth, 'r') as file:
        train_split = [fil.replace("\n", "") for fil in file.readlines()]
        new_split = []
        for pth in train_split:
            pth_split = pth.replace(".jpg", "").split("_")
            img_id = str(int(pth_split[1]) - 1)
            new_split.append(f"{pth_split[0]}_{img_id}.jpg")
        train_split = new_split

            

    file.close()

    with open(val_split_pth, 'r') as file:
        val_split = [fil.replace("\n", "") for fil in file.readlines()]
        new_split = []
        for pth in val_split:
            pth_split = pth.replace(".jpg", "").split("_")
            img_id = str(int(pth_split[1]) - 1)
            new_split.append(f"{pth_split[0]}_{img_id}.jpg")
        val_split = new_split

    file.close()

    with open(test_split_pth, 'r') as file:
        test_split = [fil.replace("\n", "") for fil in file.readlines()]

    file.close()
    return train_split, val_split, test_split


def get_dict(model_name):
    var = {
        "model_name": model_name,
        "results": {
            "dice_list": [],
            "iou_list": [],
            "acc_list": [],
            "prec_list": [],
            "rec_list": [],
            "loss": []
        }
    }

    return var
