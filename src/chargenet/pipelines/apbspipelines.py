import json
import joblib
import shutil
import h5py
import multiprocessing as mp
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from chargenet.processors import StructureMutator, PQRSolver, ElectrostaticDataset
from chargenet.models import ElectrostaticVolumeCNN


class ChargeNet:
    def __init__(
        self,
        pdb_file_path: str,
        reference_sequence: str,
        target: str,
        mutagenesis_tool: str = "foldx",
        use_ph: int = 0,
        use_temp: int = 0,
        channel_configuration: str = "all",
        resolution: float = 1.5,
        out_channels: int = 13,
        n_blocks: int = 1,
        kernel_edge_length: int = 3,
        pooling_edge_length: int = 3,
        dropout_rate: float = 0.3,
        batch_size: int = 20,
        epochs: int = 300,
        learning_rate: float = 1e-4,
        loss: str = "mse",
        optimizer: str = "adam",
        weight_decay: float = 5e-6,
        patience: int = 20,
        device: str = "cpu",
        n_cores: int = 1,
        intermediate_data_path: str = None,
        electrostatics_path: str = None,
        write_electrostatics_path: str = None,
    ):
        self.pipeline_config = {
            key: value for key, value in locals().items() if key != "self"
        }
        self.pdb_file_path = pdb_file_path
        self.reference_sequence = reference_sequence
        self.mutagenesis_tool = mutagenesis_tool
        self.use_ph = use_ph
        self.use_temp = use_temp
        self.target = target
        self.channel_configuration = channel_configuration
        self.resolution = resolution
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.kernel_edge_length = kernel_edge_length
        self.pooling_edge_length = pooling_edge_length
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.n_cores = n_cores
        self.intermediate_data_path = intermediate_data_path
        self.electrostatics_path = electrostatics_path
        self.write_electrostatics_path = write_electrostatics_path
        self.scaler = StandardScaler()
        self.grid_size = None
        self.modalities = None
        self.model = None
        self.dataset = None

    def initialize_model(self):
        self.model = ElectrostaticVolumeCNN(
            in_channels=len(self.modalities),
            out_channels=self.out_channels,
            n_blocks=self.n_blocks,
            kernel_edge_length=self.kernel_edge_length,
            grid_size=self.grid_size,
            pooling_edge_length=self.pooling_edge_length,
            dropout_rate=self.dropout_rate,
            device=self.device,
        ).to(self.device)
        cuda_device_count = torch.cuda.device_count()
        if self.device == "cuda" and cuda_device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            logger.info(f"Fitting model on {cuda_device_count} GPUs.")
            if cuda_device_count > self.batch_size:
                logger.info(
                    f"Detected a batch size smaller than the number of GPUs, "
                    f"adjusting batch size to match the number of GPUs "
                    f"({cuda_device_count})."
                )
                self.batch_size = cuda_device_count

    def run(self, data: pd.DataFrame):
        data = data.copy()
        self.scaler.fit(
            data[data["split"] == "train"][self.target].values.reshape(-1, 1)
        )
        data[self.target] = self.scaler.transform(
            data[self.target].values.reshape(-1, 1)
        ).flatten()
        self.mapper = APBSElectrostaticMapper(
            pdb_file_path=self.pdb_file_path,
            reference_sequence=self.reference_sequence,
            mutagenesis_tool=self.mutagenesis_tool,
            resolution=self.resolution,
            use_ph=self.use_ph,
            use_temp=self.use_temp,
            batch_size=self.batch_size,
            pin_memory=self.device == "cuda",
            channel_configuration=self.channel_configuration,
            n_cores=self.n_cores,
            intermediate_data_path=self.intermediate_data_path,
            target=self.target,
            electrostatics_path=self.electrostatics_path,
            write_electrostatics_path=self.write_electrostatics_path,
        )
        self.grid_size = self.mapper.pqr_solver.grid_size
        self.modalities = self.mapper.pqr_solver.modalities
        with self.mapper as mapper:
            self.dataset = mapper.run(data)
        self.initialize_model()

        logger.info("Fitting CNN.")
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.fit(
                train_loader=self.dataset.prepare_data_loader(data, "train"),
                val_loader=self.dataset.prepare_data_loader(data, "valid"),
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                loss=self.loss,
                optimizer=self.optimizer,
                weight_decay=self.weight_decay,
                patience=self.patience,
            )
        else:
            self.model.fit(
                train_loader=self.dataset.prepare_data_loader(data, "train"),
                val_loader=self.dataset.prepare_data_loader(data, "valid"),
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                loss=self.loss,
                optimizer=self.optimizer,
                weight_decay=self.weight_decay,
                patience=self.patience,
            )

    def predict(self, data: pd.DataFrame):
        predictions = []
        data_loader = self.dataset.prepare_data_loader(data, inference=True)
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in tqdm(data_loader):
                predictions.append(self.model(inputs.to(self.device)))
        return self.scaler.inverse_transform(
            torch.cat(predictions).detach().cpu().numpy()
        ).flatten()

    def save(self, model_dir: Path):
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        with open(model_dir / "config.json", "w") as fp:
            json.dump(self.pipeline_config, fp)

    @classmethod
    def load(cls, model_dir: Path):
        with open(model_dir / "config.json", "r") as fp:
            config = json.load(fp)
        pipeline = cls(**config)
        model_weights = torch.load(model_dir / "model.pt")
        pipeline.model.load_state_dict(model_weights)
        pipeline.scaler = joblib.load(model_dir / "scaler.joblib")
        return pipeline


class APBSElectrostaticMapper:
    def __init__(
        self,
        pdb_file_path: str,
        reference_sequence: str,
        batch_size: int,
        mutagenesis_tool: str = "foldx",
        resolution: float = 0.5,
        use_ph: int = 0,
        use_temp: int = 0,
        pin_memory: bool = False,
        channel_configuration: str = "charge_density",
        n_cores: int = 1,
        intermediate_data_path: str = None,
        target: str = None,
        electrostatics_path: str = None,
        write_electrostatics_path: str = None,
    ):
        self.pipeline_config = {
            key: value for key, value in locals().items() if key != "self"
        }
        self.pdb_file_path = pdb_file_path
        self.reference_sequence = reference_sequence
        self.batch_size = batch_size
        self.mutagenesis_tool = mutagenesis_tool
        self.resolution = resolution
        self.use_ph = use_ph
        self.use_temp = use_temp
        self.pin_memory = pin_memory
        self.channel_configuration = channel_configuration
        self.n_cores = n_cores
        self.intermediate_data_path = intermediate_data_path
        self.target = target
        self.electrostatics_path = electrostatics_path
        self.write_electrostatics_path = write_electrostatics_path
        self.rm_intermediate_data_path = False
        self.molecule_name = str(Path(pdb_file_path).name).removesuffix(".pdb")
        self.condition_features = self.use_ph * ["ph"] + self.use_temp * ["temperature"]
        self.mutator = StructureMutator(
            Path(pdb_file_path),
            reference_sequence=self.reference_sequence,
            mutagenesis_tool=self.mutagenesis_tool,
        )
        self.pqr_solver = PQRSolver(
            self.pdb_file_path,
            self.molecule_name,
            self.resolution,
            self.condition_features,
            self.use_ph,
            self.channel_configuration,
        )

    def __enter__(self):
        # Needs to be created and deleted only if self.intermediate_data_path is None
        if self.intermediate_data_path is None:
            self.intermediate_data_path = Path(mkdtemp())
            self.rm_intermediate_data_path = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.rm_intermediate_data_path:
            shutil.rmtree(self.intermediate_data_path)

    @property
    def n_channels(self):
        return {
            "charge": 1,
            "charge_density": 1,
            "potential": 1,
            "charge_and_charge_density": 2,
            "charge_and_potential": 2,
            "charge_density_and_potential": 2,
            "all": 3,
        }[self.channel_configuration]

    def mutate_and_compute_electrostatics(self, data):
        logger.info(
            f"Mutating structures, generating electrostatic maps "
            f"and saving results to {self.intermediate_data_path}."
        )
        self.mutator.generate_structures(data, path=Path(self.intermediate_data_path))
        electrostatics = self.pqr_solver.compute_electrostatics(
            data, path=Path(self.intermediate_data_path)
        )
        return electrostatics

    def generate_electrostatics_distributed(self, chunks):
        with mp.Pool(self.n_cores) as pool:
            for result in pool.imap_unordered(
                self.mutate_and_compute_electrostatics, chunks
            ):
                yield result

    def run(self, data: pd.DataFrame):
        logger.info(
            ", ".join(
                [
                    f"{condition} found in dataset"
                    if condition in data.columns
                    else f"{condition} not found in dataset"
                    for condition in self.condition_features
                ]
            )
        )
        dataset = ElectrostaticDataset(
            data=data,
            molecule_name=self.molecule_name,
            grid_size=self.pqr_solver.psize.ngrid,
            n_channels=self.n_channels,
            batch_size=self.batch_size,
            n_workers=min(self.n_cores, 4),
            pin_memory=self.pin_memory,
            target=self.target,
        )
        logger.info("Populating dataset.")
        if self.intermediate_data_path is None:
            raise FileNotFoundError(
                "No context created or intermediate data directory specified."
            )
        if self.electrostatics_path is not None:
            electrostatics = self.load_electrostatics(Path(self.electrostatics_path))
            dataset.update_inputs(electrostatics)
        elif self.n_cores > 1:
            chunks = np.array_split(data, min(len(data), self.n_cores * 10))
            for result in self.generate_electrostatics_distributed(chunks):
                dataset.update_inputs(result)
        else:
            dataset.update_inputs(self.mutate_and_compute_electrostatics(data))
        if self.write_electrostatics_path is not None:
            dataset.write_electrostatics(Path(self.write_electrostatics_path))
        return dataset

    @staticmethod
    def load_electrostatics(electrostatics_path: Path):
        logger.info(f"Loading electrostatic maps from {electrostatics_path}.")
        electrostatics = {}
        with h5py.File(electrostatics_path / "electrostatics.h5", "r") as h5file:
            for key in h5file.keys():
                electrostatics[key] = h5file[key][:]
        return electrostatics


class DipoleRegressor:
    def __init__(
        self,
        pdb_file_path: str,
        target: str,
        molecule_name: str,
        reference_sequence: str,
        use_ph: int = 0,
        com: int = 0,
        n_cores: int = 1,
        intermediate_data_path: str = None,
    ):
        self.pipeline_config = {
            key: value for key, value in locals().items() if key != "self"
        }
        self.pdb_file_path = pdb_file_path
        self.target = target
        self.molecule_name = molecule_name
        self.reference_sequence = reference_sequence
        self.use_ph = use_ph
        self.com = com
        self.n_cores = n_cores
        self.intermediate_data_path = intermediate_data_path
        self.pqr_solver = PQRSolver(
            reference_pdb_path=self.pdb_file_path,
            molecule_name=self.molecule_name,
            use_ph=self.use_ph,
        )
        self.mutator = StructureMutator(
            Path(self.pdb_file_path), self.reference_sequence
        )
        self.scaler = MinMaxScaler()
        self.top_model = Ridge()

    def __enter__(self):
        # Needs to be created and deleted only if self.intermediate_data_path is None
        if self.intermediate_data_path is None:
            self.intermediate_data_path = Path(mkdtemp())
            self.rm_intermediate_data_path = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.rm_intermediate_data_path:
            shutil.rmtree(self.intermediate_data_path)

    def mutate_and_compute_dipole_moments(self, data: pd.DataFrame):
        logger.info(
            f"Mutating structures, generating electrostatic maps "
            f"and saving results to {self.intermediate_data_path}."
        )
        self.mutator.generate_structures(data, path=Path(self.intermediate_data_path))
        dipole_moments = self.pqr_solver.compute_dipole_moments(
            data,
            path=Path(self.intermediate_data_path),
            com=self.com,
        )
        return dipole_moments

    def run(self, data: pd.DataFrame):
        if self.intermediate_data_path is None:
            raise FileNotFoundError(
                "No context created or intermediate data directory specified."
            )
        if self.n_cores > 1:
            dipole_moments = {}
            chunks = np.array_split(data, min(len(data), self.n_cores * 10))
            with mp.Pool(self.n_cores) as pool:
                for result in pool.map(self.mutate_and_compute_dipole_moments, chunks):
                    dipole_moments.update(result)
        else:
            dipole_moments = self.mutate_and_compute_dipole_moments(data)
        raw_vectors = [
            dipole_moments[f"{self.molecule_name}_{idx}"] for idx in data.index
        ]
        inputs = self.scaler.fit_transform(raw_vectors)
        self.top_model.fit(inputs, data[self.target])

    def predict(self, data: pd.DataFrame):
        if self.intermediate_data_path is None:
            raise FileNotFoundError(
                "No context created or intermediate data directory specified."
            )
        if self.n_cores > 1:
            dipole_moments = {}
            chunks = np.array_split(data, min(len(data), self.n_cores * 10))
            with mp.Pool(self.n_cores) as pool:
                for result in pool.map(self.mutate_and_compute_dipole_moments, chunks):
                    dipole_moments.update(result)
        else:
            dipole_moments = self.mutate_and_compute_dipole_moments(data)
        raw_vectors = [
            dipole_moments[f"{self.molecule_name}_{idx}"] for idx in data.index
        ]
        inputs = self.scaler.transform(raw_vectors)
        self.top_model.predict(inputs)

    def save(self, model_dir: Path):
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(self.top_model, model_dir / "top_model.joblib")
        with open(model_dir / "config.json", "w") as fp:
            json.dump(self.pipeline_config, fp)

    @classmethod
    def load(cls, model_dir: Path):
        with open(model_dir / "config.json", "r") as fp:
            config = json.load(fp)
        pipeline = cls(**config)
        pipeline.scaler = joblib.load(model_dir / "scaler.joblib")
        pipeline.top_model = joblib.load(model_dir / "top_model.joblib")
        return pipeline
