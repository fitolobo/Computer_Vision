import collections
import pathlib
import random
import pickle
from typing import Dict, Tuple, Sequence

import cv2
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import config
import engine
from model import FeatureEncoder, RecoloringDecoder
from engine import get_illuminance, viz_color_palette

def run():

    class ColorTransferDataset(Dataset):
        def __init__(self, data_folder, transform):
            super().__init__()
            self.data_folder = data_folder
            self.transform = transform

        def __len__(self):
            output_folder = self.data_folder/"output"
            return len(list(output_folder.glob("*")))

        def __getitem__(self, idx):
            input_img_folder = self.data_folder/"input"
            old_palette = self.data_folder/"old_palette"
            new_palette = self.data_folder/"new_palette"
            output_img_folder = self.data_folder/"output"
            files = list(output_img_folder.glob("*"))

            f = files[idx]
            ori_image = transform(cv2.imread(str(input_img_folder/f.name)))
            new_image = transform(cv2.imread(str(output_img_folder/f.name)))
            illu = get_illuminance(ori_image)

            new_palette = pickle.load(open(str(new_palette/f.stem) +'.pkl', 'rb'))
            new_palette = new_palette[:, :6, :].ravel() / 255.0

            old_palette = pickle.load(open(str(old_palette/f.stem) +'.pkl', 'rb'))
            old_palette = old_palette[:, :6, :].ravel() / 255.0

            ori_image = ori_image.double()
            new_image = new_image.double()
            illu = illu.double()
            new_palette = torch.from_numpy(new_palette).double()
            old_palette = torch.from_numpy(old_palette).double()

            return ori_image, new_image, illu, new_palette, old_palette

################################################################################
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-processsing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((432, 288)),
        transforms.ToTensor(),
    ])

    # Obs: Pensando en un diseño de sketch
    #    transforms.Resize((375, 288))


    # Cargando la data
    train_data = ColorTransferDataset(pathlib.Path(config.DATA_PATH), transform)
    train_loader = DataLoader(train_data, batch_size=config.bz)

    # create model, criterion and optimzer
    FE = FeatureEncoder().float().to(device)
    RD = RecoloringDecoder().float().to(device)
    criterion =  nn.MSELoss()
    optimizer = torch.optim.AdamW(list(FE.parameters()) + list(RD.parameters()), lr=config.lr, weight_decay=4e-3)


    # train FE and RD
    min_loss = float('inf')
    for e in range(config.epochs):
        print(e)
        total_loss = 0.
        for i_batch, sampled_batched in enumerate(tqdm(train_loader)):

            # Imagenes y Paletas
            ori_image, new_image, illu, new_palette, ori_palette = sampled_batched
            palette = new_palette.flatten()

            # Encoder-Decoder
            c1, c2, c3, c4 = FE.forward(ori_image.float().to(device))
            out = RD.forward(c1, c2, c3, c4, palette.float().to(device), illu.float().to(device),device)

            optimizer.zero_grad()
            loss = criterion(out, new_image.float().to(device))
            print(loss.item())
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        print(e, total_loss)

        if total_loss < min_loss:
            min_loss = total_loss
            state = {
                'epoch': e,
                'FE': FE.state_dict(),
                'RD': RD.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "model/modelo.pth")


if __name__ == "__main__":
    run()
