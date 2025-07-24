from pathlib import Path
from loguru import logger
from tempfile import TemporaryDirectory
from typing import Tuple
from collections import OrderedDict

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_edge_length: int):
        super(Conv3DBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_edge_length = kernel_edge_length
        self.kernel_size = (self.kernel_edge_length,) * 3
        self.padding_edge_length = (self.kernel_edge_length - 1) // 2
        self.padding = (self.padding_edge_length,) * 3

        self.conv1 = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.batch_norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.batch_norm2 = nn.BatchNorm3d(self.out_channels)
        self.batch_norm2 = nn.BatchNorm3d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv_out = self.batch_norm2(
            self.conv2(self.relu(self.batch_norm1(self.conv1(x))))
        )
        out = self.relu(x + conv_out)
        return out


class ElectrostaticVolumeCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        kernel_edge_length: int,
        grid_size: Tuple[int, int, int],
        pooling_edge_length: int,
        dropout_rate: float = 0,
        device: str = "cpu",
    ):
        super(ElectrostaticVolumeCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.kernel_edge_length = kernel_edge_length
        self.grid_size = grid_size
        self.pooling_edge_length = pooling_edge_length
        self.dropout_rate = dropout_rate
        self.device = device
        self.kernel_size = (self.kernel_edge_length,) * 3
        self.pooling_size = (self.pooling_edge_length,) * 3
        self.block_output_size = tuple(
            grid_dim // self.pooling_edge_length for grid_dim in self.grid_size
        )

        self.conv = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding="same",
        )
        self.batch_norm = self.batch_norm1 = nn.BatchNorm3d(self.out_channels)
        self.pooling = nn.MaxPool3d(kernel_size=self.pooling_size)
        self.conv_blocks = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"Conv block {i}",
                        Conv3DBlock(
                            in_channels=self.out_channels,
                            out_channels=self.out_channels,
                            kernel_edge_length=self.kernel_edge_length,
                        ),
                    )
                    for i in range(self.n_blocks)
                ]
            )
        )
        self.reducer = nn.Conv3d(
            in_channels=self.out_channels,
            out_channels=1,
            kernel_size=1,
        )
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(
            in_features=int(np.prod(self.block_output_size)), out_features=1
        )

    def forward(self, x):
        x = self.pooling(self.relu(self.batch_norm(self.conv(x))))
        x = self.conv_blocks(x)
        x = self.reducer(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

    def n_parameters(self):
        return sum(
            [
                np.prod(params.size())
                for params in self.parameters()
                if params.requires_grad
            ]
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss: str = "mse",
        learning_rate: float = 1e-3,
        epochs: int = 10,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        patience: int = 10,
    ):
        criterion = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "huber": nn.HuberLoss,
        }[loss]()
        optimizer = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
        }[optimizer](self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        patience_counter = 0
        best_val_loss = np.inf
        with TemporaryDirectory() as temp_dir:
            for epoch in range(epochs):
                running_train_loss = 0
                running_val_loss = 0
                self.train(True)
                for step, (inputs, outputs) in enumerate(train_loader):
                    optimizer.zero_grad()
                    predictions = self(inputs.to(self.device))
                    loss = criterion(predictions, outputs.to(self.device))
                    running_train_loss += loss
                    loss.backward()
                    optimizer.step()
                train_loss = running_train_loss / (step + 1)

                self.eval()
                with torch.no_grad():
                    for step, (inputs, outputs) in enumerate(val_loader):
                        predictions = self(inputs.to(self.device))
                        loss = criterion(predictions, outputs.to(self.device))
                        running_val_loss += loss
                val_loss = running_val_loss / (step + 1)
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss.item():.3e}, "
                    f"Val loss: {val_loss.item():.3e}."
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), f"{temp_dir}/best_model.pt")
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter == patience:
                    logger.info("Early stopping, restoring best weights.")
                    break
            if Path(f"{temp_dir}/best_model.pt").is_file():
                saved_weights = torch.load(f"{temp_dir}/best_model.pt")
                self.load_state_dict(saved_weights)
