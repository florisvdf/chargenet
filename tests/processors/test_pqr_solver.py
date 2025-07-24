import subprocess
import numpy as np
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from chargenet.processors import PQRSolver


class TestPQRSolver:
    @pytest.fixture(scope="session")
    def solver(self, normal_pdb_file_path):
        molecule_name = normal_pdb_file_path.name.strip(".pdb")
        return PQRSolver(str(normal_pdb_file_path), molecule_name)

    def test_pdb2pqr_is_affected_by_ph(self, normal_pdb_file_path):
        phs = [2, 13]
        molecule_name = normal_pdb_file_path.name.strip(".pdb")
        with TemporaryDirectory() as temp_dir:
            total_charges = []
            for ph in phs:
                pqr_output_path = Path(temp_dir) / f"{molecule_name}_{ph}.pqr"
                command = [
                    "pdb2pqr",
                    "--ff=AMBER",
                    str(normal_pdb_file_path),
                    str(pqr_output_path),
                    "--titration-state-method=propka",
                    f"--with-ph={ph}",
                ]
                _ = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                )
                with open(pqr_output_path, "r") as fp:
                    atom_charges = [
                        float(line.split(" ")[-2])
                        for line in fp.readlines()
                        if line.split(" ")[0] == "ATOM"
                    ]
                    total_charges.append(sum(atom_charges))
        assert not np.allclose(total_charges[0], total_charges[1])
