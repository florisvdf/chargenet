import h5py
from loguru import logger
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, Subset


class ElectrostaticDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        molecule_name: str,
        grid_size: tuple,
        n_channels: int,
        batch_size: int,
        n_workers: int,
        pin_memory: bool = False,
        target: str = None,
        mode: str = "volume",
    ):
        self.data = data
        self.molecule_name = molecule_name
        self.grid_size = grid_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.target = target
        self.mode = mode
        self.input_shape = self.set_input_shape()
        self.inputs = torch.empty(self.input_shape, dtype=torch.float32)
        self.outputs = (
            torch.tensor(
                self.data[self.target].values.astype(np.float32).reshape(-1, 1)
            )
            if self.target is not None
            else None
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    @property
    def variant_name_to_array_index(self):
        return {
            f"{self.molecule_name}_{data_idx}": array_index
            for data_idx, array_index in zip(self.data.index, range(len(self.data)))
        }

    def update_inputs(self, electrostatics: dict):
        {
            "volume": self.update_volumes,
            "mesh": self.update_meshes,
        }[self.mode](electrostatics)

    def update_volumes(self, electrostatics: dict):
        for variant_name, values in electrostatics.items():
            self.inputs[self.variant_name_to_array_index[variant_name]] = torch.tensor(
                values
            )

    def update_meshes(self):
        raise NotImplementedError

    def set_input_shape(self):
        if self.mode == "volume":
            return (len(self.data), self.n_channels, *self.grid_size)
        elif self.mode == "mesh":
            raise NotImplementedError

    def prepare_data_loader(
        self,
        data: pd.DataFrame,
        split: Optional[str] = None,
        inference: bool = False,
    ):
        sample_idx = (
            np.where(data["split"] == split)[0]
            if split is not None
            else np.arange(len(data))
        )
        return DataLoader(
            Subset(self, sample_idx),
            batch_size=self.batch_size,
            shuffle=not inference,
            pin_memory=self.pin_memory,
            num_workers=self.n_workers,
        )

    def write_electrostatics(self, write_electrostatics_path: Path):
        output_path = write_electrostatics_path / "electrostatics.h5"
        logger.info(f"Writing electrostatics to {str(output_path)}")
        with h5py.File(output_path, "a") as h5file:
            for i, idx in enumerate(self.data.index):
                key = f"{self.molecule_name}_{idx}"
                value = self.inputs[i].numpy()
                h5file.create_dataset(key, data=value)
