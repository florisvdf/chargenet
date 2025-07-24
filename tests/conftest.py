import pytest
import random
import shutil
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
from biopandas.pdb import PandasPdb

from chargenet.constants import THREE_TO_SINGLE_LETTER_CODES, AA_ALPHABET


@pytest.fixture(scope="session")
def tmpdir_session() -> Path:
    path = mkdtemp()
    yield Path(path)
    shutil.rmtree(path)


@pytest.fixture(scope="session")
def mini_pdb_file_path() -> Path:
    return Path(__file__).parent / "data/1N9U.pdb"


@pytest.fixture(scope="session")
def tiny_pdb_file_path() -> Path:
    return Path(__file__).parent / "data/2PM1.pdb"


@pytest.fixture(scope="session")
def small_pdb_file_path() -> Path:
    return Path(__file__).parent / "data/1EOD.pdb"


@pytest.fixture(scope="session")
def normal_pdb_file_path() -> Path:
    return Path(__file__).parent / "data/1AGY.pdb"


@pytest.fixture(scope="session")
def mini_non_canonical_pdb_file_path() -> Path:
    return Path(__file__).parent / "data/8GL4.pdb"


def protein_data_frame(pdb_file_path, n_samples, mut_prob) -> pd.DataFrame:
    reference_sequence = "".join(
        (
            PandasPdb()
            .read_pdb(str(pdb_file_path))
            .df["ATOM"][["residue_number", "residue_name"]]
            .drop_duplicates("residue_number")
        )["residue_name"]
        .map(lambda x: THREE_TO_SINGLE_LETTER_CODES[x])
        .to_list()
    )
    sequences = set()
    while len(sequences) < n_samples:
        sequences.add(
            "".join(
                [
                    random.choices(
                        [aa_ref, random.choice(AA_ALPHABET)], [1 - mut_prob, mut_prob]
                    )[0]
                    for aa_ref in reference_sequence
                ]
            )
        )

    return pd.DataFrame(
        dict(
            sequence=list(sequences),
            split=["train", "valid"] * int(n_samples / 2),
            a=[1] * n_samples,
            b=[2] * n_samples,
            c=[3, 4] * int(n_samples / 2),
            d=["a", "b"] * int(n_samples / 2),
        )
    )

@pytest.fixture(scope="session")
def mini_protein_data_frame(mini_pdb_file_path):
    return protein_data_frame(mini_pdb_file_path, n_samples=12, mut_prob=0.25)


@pytest.fixture(scope="session")
def tiny_protein_data_frame(tiny_pdb_file_path):
    return protein_data_frame(tiny_pdb_file_path, n_samples=12, mut_prob=0.1)


@pytest.fixture(scope="session")
def small_protein_data_frame(small_pdb_file_path):
    return protein_data_frame(small_pdb_file_path, n_samples=12, mut_prob=0.06)


@pytest.fixture(scope="session")
def normal_protein_data_frame(normal_pdb_file_path):
    return protein_data_frame(normal_pdb_file_path, n_samples=12, mut_prob=0.03)
