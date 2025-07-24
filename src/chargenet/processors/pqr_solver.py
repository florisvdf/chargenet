import re
from loguru import logger
import subprocess
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from pdb2pqr.psize import Psize

from utilin.constants import CHEMICAL_ELEMENT_PATTERN, ATOMIC_MASSES

from chargenet.utils import (
    prepare_input_config,
    parse_apbs_outputs,
    extract_conditions,
)
from chargenet.constants import DEFAULT_CONDITIONS


class PQRSolver:
    def __init__(
        self,
        reference_pdb_path: str,
        molecule_name: str,
        resolution: float = 1.5,
        condition_features: list = None,
        use_ph: int = 0,
        channel_configuration: str = "charge",
    ):
        self.reference_pdb_path = reference_pdb_path
        self.molecule_name = molecule_name
        self.resolution = resolution
        self.condition_features = condition_features
        self.use_ph = use_ph
        self.channel_configuration = channel_configuration
        self.psize = Psize(space=self.resolution)
        self.grid_size = None
        self.initialize_psize()

    @property
    def modalities(self):
        return {
            "charge": ["charge"],
            "charge_density": ["charge_density"],
            "potential": ["potential"],
            "charge_and_charge_density": ["charge", "charge_density"],
            "charge_and_potential": ["charge", "potential"],
            "charge_density_and_potential": ["charge_density", "potential"],
            "all": ["charge", "charge_density", "potential"],
        }[self.channel_configuration]

    def initialize_psize(self):
        logger.info("Computing APBS grid settings.")
        self.convert_pdb_to_pqr(
            Path(self.reference_pdb_path).parent,
            self.molecule_name,
            conditions=DEFAULT_CONDITIONS,
        )
        self.psize.run_psize(
            str(Path(self.reference_pdb_path).parent / f"{self.molecule_name}.pqr")
        )
        self.grid_size = self.psize.ngrid
        # logger.info(f"Psize configuration set to:{self.psize.__str__()}.")

    def convert_pdb_to_pqr(self, path, variant_id, conditions):
        input_file_path = path / f"{variant_id}.pdb"
        with open(input_file_path, "r") as file:
            lines = file.readlines()
        if "FoldX" in lines[0]:
            with open(input_file_path, "w") as file:
                file.writelines(lines[3:])
        output_file_path = path / f"{variant_id}.pqr"
        command = [
            "pdb2pqr",
            "--ff=AMBER",
            str(input_file_path),
            str(output_file_path),
        ]
        if self.use_ph == 1:
            command.extend(
                ["--titration-state-method=propka", f"--with-ph={conditions['ph']}"]
            )
        try:
            _ = subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command '{e.cmd}' failed with return code {e.returncode}. Error: {e.stderr.decode().strip()}"
            )

    @staticmethod
    def pqr_to_dipole_moment(path: Union[str, Path], reference_point: str):
        """
        Does NOT take into account HETATOMS
        """
        atom_pattern = re.compile(CHEMICAL_ELEMENT_PATTERN)
        with open(path, "r") as fp:
            lines = fp.readlines()
            atom_entries = list(filter(lambda x: x.startswith("ATOM"), lines))
            coords = np.empty((len(atom_entries), 3))
            charges = np.empty((len(atom_entries), 1))
            masses = np.empty((len(atom_entries), 1))
            for i, entry in enumerate(atom_entries):
                coords[i] = entry.strip("\n").split()[-5:-2]
                charges[i] = entry.strip("\n").split()[-2]
                masses[i] = ATOMIC_MASSES[
                    re.findall(atom_pattern, entry.strip("\n").split()[2])[0]
                ]
        net_charge = np.sum(charges)
        charge_vectors = coords * charges
        total_mass = sum(masses)
        mass_vectors = coords * masses
        if reference_point == "com":
            reference_point = (np.sum(mass_vectors, axis=0) / total_mass).reshape(1, 3)
        elif reference_point == "coc":
            reference_point = (np.sum(coords, axis=0) / len(coords)).reshape(1, 3)
        elif reference_point == "origin":
            reference_point = np.zeros((1, 3))
        else:
            # This is most likely bugged
            reference_point = (np.sum(charge_vectors, axis=0) / net_charge).reshape(
                1, 3
            )
        distances_from_reference = coords - reference_point
        centered_charge_vectors = distances_from_reference * charges
        dipole_moment = np.sum(centered_charge_vectors, axis=0)
        return dipole_moment

    def compute_dipole_moments(self, data: pd.DataFrame, path: Path, com: int):
        dipole_moments = {}
        for idx, row in data.iterrows():
            conditions = extract_conditions(
                row, condition_features=self.condition_features
            )
            variant_id = f"{self.molecule_name}_{idx}"
            self.convert_pdb_to_pqr(path, variant_id, conditions)
            dipole_moments[variant_id] = self.pqr_to_dipole_moment(
                path / f"{variant_id}.pqr", com
            )
        return dipole_moments

    def compute_electrostatics(self, data: pd.DataFrame, path: Path):
        electrostatics = {}
        for idx, row in data.iterrows():
            conditions = extract_conditions(
                row, condition_features=self.condition_features
            )
            variant_id = f"{self.molecule_name}_{idx}"
            self.convert_pdb_to_pqr(path, variant_id, conditions)
            config = prepare_input_config(
                pqr_file=path / f"{variant_id}.pqr",
                center=self.psize.center,
                fine_grid_length=self.psize.fine_length,
                coarse_grid_length=self.psize.coarse_length,
                resolution=self.psize.ngrid,
                modalities=self.modalities,
                conditions=conditions,
            )
            with open(f"{path}/input_{idx}.in", "w") as f:
                f.write(config)
            command = ["apbs", f"{path}/input_{idx}.in"]
            try:
                _ = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Command '{e.cmd}' failed with return code {e.returncode}. Error: {e.stderr.decode().strip()}"
                )
            electrostatics[variant_id] = parse_apbs_outputs(
                path, variant_id, self.modalities, self.grid_size
            )
        return electrostatics
