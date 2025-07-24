import pytest

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from chargenet.models import ElectrostaticVolumeCNN


class TestElectrostaticVolumeCNN:
    @pytest.fixture(scope="session")
    def n_samples(self):
        return 10

    @pytest.fixture(scope="session")
    def in_channels(self):
        return 3

    @pytest.fixture(scope="session")
    def volume_size(self):
        return (12, 12, 12)

    @pytest.fixture(scope="session")
    def input_shape(self, n_samples, in_channels, volume_size):
        return (n_samples, in_channels, *volume_size)

    @pytest.fixture(scope="session")
    def train_loader(self, input_shape):
        inputs = torch.randn(input_shape)
        return DataLoader(
            TensorDataset(
                inputs, torch.unsqueeze(torch.mean(inputs, dim=(1, 2, 3, 4)), 1)
            ),
            batch_size=6,
            shuffle=True,
        )

    @pytest.fixture(scope="session")
    def val_loader(self, input_shape):
        inputs = torch.randn(input_shape)
        return DataLoader(
            TensorDataset(
                inputs, torch.unsqueeze(torch.mean(inputs, dim=(1, 2, 3, 4)), 1)
            ),
            batch_size=6,
            shuffle=False,
        )

    @pytest.fixture(scope="session")
    def model(self, in_channels, volume_size):
        return ElectrostaticVolumeCNN(
            in_channels=in_channels,
            out_channels=2,
            n_blocks=1,
            kernel_edge_length=3,
            grid_size=volume_size,
            pooling_edge_length=3,
        )

    def test_model_trains_with_mse_loss(self, model, train_loader, val_loader):
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss="mse",
            epochs=2,
        )
        assert not np.isnan(
            model(train_loader.__iter__().__next__()[0]).detach().cpu().numpy()
        ).any()

    def test_model_trains_with_mae_loss(self, model, train_loader, val_loader):
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss="mae",
            epochs=2,
        )
        assert not np.isnan(
            model(train_loader.__iter__().__next__()[0]).detach().cpu().numpy()
        ).any()

    def test_model_trains_with_huber_loss(self, model, train_loader, val_loader):
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            loss="mae",
            epochs=2,
        )
        assert not np.isnan(
            model(train_loader.__iter__().__next__()[0]).detach().cpu().numpy()
        ).any()
