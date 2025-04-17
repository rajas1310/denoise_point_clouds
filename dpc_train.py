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

import argparse

from torch.utils.data._utils.collate import default_collate

def my_collate(batch):
    print(f"Collating a batch of size: {len(batch)}")
    return default_collate(batch)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def get_data(batch_size, image_size, data_dir):
    ds_obj = DPCDataset(data_dir, img_sz=image_size, transform=False)
    dataset = ds_obj.get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=my_collate)
    print(f"Number of samples in dataset: {len(dataset)}")
    print(f"Number of batches in dataloader: {len(dataloader)}")
    return dataloader

    
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args.batch_size, args.image_size, args.data_dir)
    model = UNetDPC(c_in=5, c_out=1, device=args.device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)

    model.train()
    for epoch in range(args.num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (input_tensor, target) in enumerate(pbar):
            input_tensor = input_tensor.to(device)
            target_noise = target['noise'].to(device)

            predicted_noise = model(input_tensor)
            loss_noise = mse(predicted_noise, target_noise)
            loss_depth = mse(target['noisy_depth'].to(device)-predicted_noise, target['depth'].to(device))

            loss = (loss_noise + loss_depth) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
    
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))

def inference(args):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/rgbd-scenes-v2")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)

    # ds_obj = DPCDataset(args.data_dir, img_sz=args.image_size, transform=True)
    # dataset = ds_obj.get_dataset()
    # sample = dataset[0]
    # print(sample)

if __name__ == '__main__':
    main()
