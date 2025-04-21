import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet, UNetDPC
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dpc_dataloader import DPCDataset
import numpy as np

import wandb

import argparse

from torch.utils.data._utils.collate import default_collate

from warmup_scheduler import GradualWarmupScheduler  # or use native version




def my_collate(batch):
    print(f"Collating a batch of size: {len(batch)}")
    return default_collate(batch)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def get_data(batch_size, image_size, data_dir, num_workers):
    ds_obj = DPCDataset(data_dir, img_sz=image_size)
    trainset, valset = ds_obj.get_dataset()
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Number of samples in TRAIN dataset: {len(trainset)}")
    print(f"Number of samples in VALIDATION dataset: {len(valset)}")
    return train_dataloader, val_dataloader

    
def train(args, wandb_run=None):
    setup_logging(args.run_name)
    device = args.device
    train_dataloader, val_dataloader = get_data(args.batch_size, args.image_size, args.data_dir, args.num_workers)
    # print("Data_loaded:\n",torch.cuda.memory_summary(), "\n--------------\n")
    model = UNetDPC(c_in=5, c_out=1, device=args.device).to(device)
    model.load_model("checkpoints/unconditional_ckpt.pt")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # print("Model_loaded:\n",torch.cuda.memory_summary(), "\n--------------\n")
    # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"init_ckpt.pt"))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    warmup_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,  # target LR multiplier
        total_epoch= args.warmup_epochs,  # number of warm-up epochs
        after_scheduler=base_scheduler
    )

    train_losses = []

    best_val_loss = float("inf")

    model.train()
    for epoch in range(args.num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))
        scaler = torch.amp.GradScaler(device=device)

        for i, (input_tensor, target) in enumerate(pbar):
            input_tensor = input_tensor.to(device)
            target_noise = target['noise'].to(device)

            with torch.amp.autocast(device):
                predicted_noise = model(input_tensor)
                # print("Forwardpass:\n",torch.cuda.memory_summary(), "\n--------------\n")
                loss_noise = mse(predicted_noise, target_noise)
                loss_depth = mse(target['noisy_depth'].to(device)-predicted_noise, target['depth'].to(device))
                loss = (loss_noise + loss_depth) / 2

            optimizer.zero_grad()
            if wandb_run:
                wandb_run.log({"train_loss_step": loss.item()})
            train_losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            # print("backward pass:\n",torch.cuda.memory_summary(), "\n--------------\n")
            torch.cuda.empty_cache()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len(train_dataloader) + i)
        
        warmup_scheduler.step()
        if wandb_run:
            wandb_run.log({"train_loss_epoch": np.mean(train_losses)})

        # Validation
        if (epoch+1) % args.val_interval == 0:
            val_loss = test(args, model, val_dataloader)
            logger.add_scalar("Validation MSE", val_loss, global_step=epoch)
            if wandb_run:
                wandb_run.log({"val_loss_epoch": val_loss})
            logging.info(f"Epoch {epoch} Validation MSE: {val_loss}")
            print(f"Epoch {epoch} Validation MSE: {val_loss}")
            # Save the model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"Best validation loss so far: {best_val_loss}")
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"best_ckpt.pt"))
    
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"final_ckpt.pt"))

def test(args, model, test_loader):
    device = args.device
    model = model.to(device)

    mse = nn.MSELoss()

    losses = []
    
    model.eval()
    with torch.no_grad():
        for i, (input_tensor, target) in enumerate(test_loader):
            input_tensor = input_tensor.to(device)
            target_noise = target['noise'].to(device)

            predicted_noise = model(input_tensor)
            loss_noise = mse(predicted_noise, target_noise)
            loss_depth = mse(target['noisy_depth'].to(device)-predicted_noise, target['depth'].to(device))
            loss = (loss_noise + loss_depth) / 2
            losses.append(loss.item())
    
    loss = np.mean(losses)
    model.train()

    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/rgbd-scenes-v2")
    parser.add_argument("--run_name", type=str, default="overfitting_test")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_epochs", "--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--val_interval", "--test_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Start a new wandb run to track this script.
    wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        # entity="Failure-Prediction-group-SLURM",
        # Set the wandb project where this run will be logged.
        project="Denoising Point Cloud",
        name=args.run_name,
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "run_name": args.run_name,
            "epochs": args.num_epochs,
            "val_interval": args.val_interval,},
    )

    train(args, wandb_run)

    wandb_run.finish()

    # ds_obj = DPCDataset(args.data_dir, img_sz=args.image_size, transform=True)
    # dataset = ds_obj.get_dataset()
    # sample = dataset[0]
    # print(f"Sample shape: {sample[0].shape}")

if __name__ == '__main__':
    main()
