import contextlib
import itertools
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from Bio import PDB
from Bio.PDB import PDBIO, Chain, Model, Residue, Structure
from Bio.PDB.cealign import CEAligner
from Bio.PDB.PDBExceptions import PDBConstructionException
from biopandas.pdb import PandasPdb
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from loguru import logger
from polyleven import levenshtein

from sklearn.cluster import k_means
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from utilin.utils import read_fasta

from chargenet.constants import (
    DEFAULT_CONDITIONS,
    ROTABASE_LOCATION,
    THREE_TO_SINGLE_LETTER_CODES,
    UNIPROT_ACCESSION_PATTERN,
)

CURRENT_PATH = os.environ.get("PATH").removeprefix("$PATH:")


class ProteinGym:
    """
    Currently only support for substitutions.
    """

    def __init__(self, proteingym_location, meta_data_path):
        self.proteingym_location = proteingym_location
        self.meta_data_path = meta_data_path
        self.reference_information = pd.read_csv(self.meta_data_path, index_col=None)
        self.available_pdbs = {}

    def update_reference_information(self):
        logger.info("Updating reference information with structure information.")
        pdb_entry_pattern = r".*PDB; [A-Z0-9]{4};.*"
        region_pattern = r"A=(\d+-\d+)"
        for i, row in self.reference_information.iterrows():
            uniprot_id = row["UniProt_ID"]
            region_mutated = row["region_mutated"]
            response = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
            )
            if response.status_code == 400:
                continue
            body = response.text
            pdb_entries = list(
                filter(
                    lambda x: True if re.match(pdb_entry_pattern, x) else False,
                    body.split("\n"),
                )
            )
            reference_sequence = self.reference_information.loc[i, "target_seq"]
            uniprot_sequence = fetch_uniprot_sequence(uniprot_id)
            self.available_pdbs[uniprot_id] = pdb_entries
            self.reference_information.loc[i, "has_pdb_structure"] = (
                len(pdb_entries) > 0
            )
            self.reference_information.loc[i, "structure_covers_mutated_region"] = False
            self.reference_information.loc[i, "uniprot_sequence"] = uniprot_sequence
            self.reference_information.loc[i, "reference_distance_to_uniprot"] = (
                distance_of_reference_to_uniprot(reference_sequence, uniprot_sequence)
            )
            for entry in pdb_entries:
                match = re.search(region_pattern, entry)
                if match:
                    structure_region = match.group(1)
                    structure_covers_mutated_region = region_is_subregion(
                        region_mutated, structure_region
                    )
                    if structure_covers_mutated_region:
                        self.reference_information.loc[
                            i, "structure_covers_mutated_region"
                        ] = True
                        break

    def describe_dataset(self, dataset_name):
        return self.reference_information[
            self.reference_information["DMS_id"] == dataset_name
        ]

    def fetch_msa(self, dataset_name):
        uniprot_id = self.describe_dataset(dataset_name)["UniProt_ID"].values[0]
        matching_msa_paths = list(
            (Path(self.proteingym_location) / "MSA_files/DMS").rglob(f"{uniprot_id}*")
        )
        logger.info(f"Found {len(matching_msa_paths)} matching MSA files")
        return read_fasta(matching_msa_paths[0])

    def prepare_dataset(self, dataset_name):
        data = pd.read_csv(
            Path(self.proteingym_location)
            / f"cv_folds_singles_substitutions/{dataset_name}.csv"
        ).rename(columns={"mutated_sequence": "sequence"})
        return data


class MegaScaleProteinFoldingStability:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_csv(dataset_path)
        self.dataset["n_mutations"] = self.dataset["mut_type"].map(
            self.find_number_of_mutations
        )
        self.dataset["mutated_positions"] = self.dataset["mut_type"].map(
            self.find_mutated_positions
        )
        self.dataset["synonymous"] = self.dataset["mut_type"].map(self.is_synonymous)
        self.dataset["is_ins_or_del"] = self.dataset["mut_type"].map(
            self.is_insertion_or_deletion
        )
        self.dataset["sequence"] = self.dataset["aa_seq"]

    @property
    def dataset_names(self):
        return self.dataset["WT_name"].map(lambda x: x.split(".")[0]).unique()

    @staticmethod
    def find_number_of_mutations(mut_type):
        if mut_type in ["wt", ""]:
            return 0
        if re.search(r"ins|del", mut_type):
            return 1
        else:
            return len(mut_type.split(":"))

    @staticmethod
    def find_mutated_positions(mut_type):
        return (
            frozenset(int(mut[1:-1]) for mut in mut_type.split(":"))
            if not re.search(r"wt|ins|del", mut_type)
            else frozenset()
        )

    @staticmethod
    def is_synonymous(mut_type):
        return (
            all([mut[0] == mut[-1] for mut in mut_type.split(":")])
            if not re.search(r"wt|ins|del", mut_type)
            else False
        )

    @staticmethod
    def is_insertion_or_deletion(mut_type):
        return True if re.search(r"ins|del", mut_type) else False

    def fetch_dataset(
        self,
        pdb_id,
        keep_synonymous=False,
        keep_insertions_and_deletions=False,
        drop_duplicate_sequences=True,
        order="second",
    ):
        return self.dataset[
            (self.dataset["WT_name"] == f"{pdb_id}.pdb")
            & (self.dataset["synonymous"] == (False or keep_synonymous))
            & (
                self.dataset["is_ins_or_del"]
                == (False or keep_insertions_and_deletions)
            )
            & (
                self.dataset["n_mutations"].isin(
                    {"first": [1], "second": [2], "both": [1, 2]}[order]
                )
            )
        ].drop_duplicates(subset="sequence" if drop_duplicate_sequences else None)

    def get_reference_sequence(self, pdb_id):
        return self.dataset[self.dataset["name"] == f"{pdb_id}.pdb"]["sequence"]

    def describe_dataset(self, pdb_id):
        raise NotImplementedError


def slice_structure_sequence(reference_sequence: str, structure_sequence: str):
    mat = SubstitutionMatrix.std_protein_matrix()
    reference = ProteinSequence(reference_sequence)
    structure = ProteinSequence(structure_sequence)
    alignment = align_optimal(reference, structure, mat)[0]
    trace = alignment.trace
    mask = trace[:, 0] != -1
    relevant_structure_region = trace[mask][:, 1]
    relevant_structure_sequence = structure[relevant_structure_region]
    return relevant_structure_sequence.__str__()


def extract_sequence_from_structure(reference: str, structure: PandasPdb):
    sequence_data = structure.df["ATOM"][
        ["residue_number", "residue_name", "chain_id"]
    ].drop_duplicates("residue_number")
    chars = []
    for position in range(1, len(reference) + 1):
        residue_name = sequence_data.loc[
            sequence_data["residue_number"] == position, "residue_name"
        ]
        if residue_name.empty:
            chars.append("*")
        elif residue_name.values[0] not in THREE_TO_SINGLE_LETTER_CODES.keys():
            chars.append("X")
        else:
            chars.append(THREE_TO_SINGLE_LETTER_CODES[residue_name.values[0]])
    return "".join(chars)


class PDBChecker:
    def __init__(
        self,
        pdb_file_path: Union[str, Path],
        reference_sequence: str,
        verbosity: int = 1,
    ):
        self.pdb_file_path = pdb_file_path
        self.reference_sequence = reference_sequence
        self.verbosity = verbosity
        self.structure = PandasPdb().read_pdb(str(self.pdb_file_path))
        self.structure_sequence = extract_sequence_from_structure(
            self.reference_sequence, self.structure
        )
        self.sliced_structure_sequence = slice_structure_sequence(
            self.reference_sequence, self.structure_sequence
        )

    @staticmethod
    def reference_in_structure_sequence(
        reference_sequence: str, structure_sequence: str
    ) -> bool:
        return reference_sequence in structure_sequence

    def reference_sequence_matches_structure(
        self, reference_sequence: str, structure_sequence: str
    ):
        if "*" in structure_sequence:
            segments = structure_sequence.split("*")
            if not all([segment in reference_sequence for segment in segments]):
                if self.verbosity == 1:
                    logger.warning(
                        "Sequence corresponding to structure does not match reference sequence!"
                    )
                result = False
                return result
        elif "X" in structure_sequence:
            if self.verbosity == 1:
                logger.warning(
                    "Sequence corresponding to structure contains non canonical residues!"
                )
            result = False
            return result
        else:
            if structure_sequence != reference_sequence:
                if self.verbosity == 1:
                    logger.warning(
                        "Sequence corresponding to structure does not match reference sequence!"
                    )
                result = False
                return result
        if self.verbosity == 1:
            logger.info(
                "Sequence corresponding to structure matches reference sequence!"
            )
        result = True
        return result

    def structure_is_complete(self, reference_sequence: str, structure_sequence: str):
        self.missing_residues = []
        for i, (reference_residue, structure_residue) in enumerate(
            zip(reference_sequence, structure_sequence)
        ):
            position = i + 1
            if structure_residue == "*":
                self.missing_residues.append(f"{reference_residue}{position}")
        if self.missing_residues == []:
            if self.verbosity == 1:
                logger.info("Structure contains all residues.")
            result = True
        else:
            if self.verbosity == 1:
                logger.warning(f"Missing residues {self.missing_residues}")
            result = False
        return result

    def run_checks(self):
        match_result = self.reference_sequence_matches_structure(
            self.reference_sequence, self.sliced_structure_sequence
        )
        complete_result = self.structure_is_complete(
            self.reference_sequence, self.sliced_structure_sequence
        )
        if match_result and complete_result:
            return "Ok"
        else:
            return "Bad structure"


class PDBPrepper:
    def __init__(
        self,
        pdb_file_path: Union[str, Path],
        reference_sequence: str,
        uniprot_id: str,
        measurement_type: str = "X-ray",
        foldx_repair: bool = True,
    ):
        self.pdb_file_path = pdb_file_path
        self.reference_sequence = reference_sequence
        self.uniprot_id = uniprot_id
        self.measurement_type = measurement_type
        self.foldx_repair = foldx_repair
        self.missing_residues = None
        self.pdb_id = Path(self.pdb_file_path).name.split(".")[0]
        self.structure = PandasPdb().read_pdb(str(self.pdb_file_path))
        self.structure_sequence = extract_sequence_from_structure(
            self.reference_sequence, self.structure
        )
        self.sliced_structure_sequence = slice_structure_sequence(
            self.reference_sequence, self.structure_sequence
        )

    def reset_pdb_residue_positions(self, pdb_file_path: Union[str, Path]):
        structure = PandasPdb().read_pdb(str(pdb_file_path))
        structure.df["ATOM"] = structure.df["ATOM"][
            structure.df["ATOM"]["residue_number"] > 0
        ]
        sequence_data = structure.df["ATOM"][
            ["residue_number", "residue_name", "chain_id"]
        ].drop_duplicates("residue_number")
        consecutive_groups = (
            sequence_data["residue_number"] - sequence_data["residue_number"].shift(1)
            != 1
        ).cumsum()
        segments = [
            group
            for _, group in sequence_data.groupby(consecutive_groups, sort=False)
            if "".join(group["residue_name"].map(THREE_TO_SINGLE_LETTER_CODES))
            in self.reference_sequence
        ]
        largest_segment = list(sorted(segments, key=lambda segment: len(segment)))[0]
        position_of_segment_in_reference = (
            self.reference_sequence.find(
                "".join(
                    largest_segment["residue_name"].map(THREE_TO_SINGLE_LETTER_CODES)
                )
            )
            + 1
        )
        position_of_segment_in_structure = largest_segment["residue_number"].iloc[0]
        offset = position_of_segment_in_structure - position_of_segment_in_reference
        structure.df["ATOM"]["residue_number"] = (
            structure.df["ATOM"]["residue_number"] - offset
        )
        return structure

    def prepare_structure(self, path: Union[str, Path]):
        with TemporaryDirectory() as temp_dir:
            pdb_file_path = Path(temp_dir) / f"{self.pdb_id}.pdb"
            reindexed_structure = self.reset_pdb_residue_positions(self.pdb_file_path)
            reindexed_structure.to_pdb(pdb_file_path)
            target_parser = PDB.PDBParser(QUIET=True)
            target_structure = target_parser.get_structure(self.pdb_id, pdb_file_path)

            if self.missing_residues == []:
                residues_of_interest = []
                for position, amino_acid in enumerate(self.reference_sequence, start=1):
                    for residue in target_structure.get_residues():
                        if (
                            THREE_TO_SINGLE_LETTER_CODES.get(residue.resname)
                            == amino_acid
                        ) and (residue.id[1] == position):
                            residues_of_interest.append(residue)
                prepared_structure = self.new_structure_from_residues(
                    residues_of_interest
                )
            else:
                logger.info("Attempting to impute residues with alphafold structure.")
                download_alphafold_structure(
                    Path(temp_dir) / f"{self.uniprot_id}.pdb", self.uniprot_id
                )
                source_parser = PDB.PDBParser(QUIET=True)
                source_structure = source_parser.get_structure(
                    self.uniprot_id, Path(temp_dir) / f"{self.uniprot_id}.pdb"
                )
                atoms = [atom for atom in source_structure.get_atoms()]
                mean_plddt = np.mean([atom.bfactor for atom in atoms])
                if mean_plddt > 0.7:
                    logger.info(
                        f"Alphafold structure has a mean pLDDT of {mean_plddt:.2f}."
                    )
                else:
                    logger.warning(
                        f"Alphafold structure has a mean pLDDT of {mean_plddt:.2f}, "
                        f"this will likely cause errors during imputation!"
                    )
                aligner = CEAligner()
                aligner.set_reference(target_structure)
                aligner.align(source_structure)
                rmsd = aligner.rms
                if rmsd < 3:
                    logger.info(f"RMSD of superimposed structures: {rmsd:.2f}.")
                else:
                    logger.warning(
                        f"Superimposed structures have an RMSD of {rmsd:.2f}, "
                        f"this will likely cause errors during imputation!"
                    )
                prepared_structure = self.transfer_residues(
                    target_structure, source_structure
                )

        self.write_pdb(path, prepared_structure)
        if self.foldx_repair:
            logger.info("Repairing prepared structure with foldx.")
            cwd = str(Path(path).parent)
            structure_file_name = Path(path).name
            structure_name = structure_file_name.split(".")[0]
            command = [
                "foldx",
                "-c",
                "RepairPDB",
                "--pdb",
                structure_file_name,
                "--rotabase",
                str(ROTABASE_LOCATION),
            ]
            _ = subprocess.run(command, cwd=cwd)
            shutil.move(Path(cwd) / f"{structure_name}_Repair.pdb", path)

    def transfer_residues(
        self,
        target_structure: Structure.Structure,
        source_structure: Structure.Structure,
    ):
        residues_to_copy = []
        for residue_identifier in self.missing_residues:
            amino_acid, position = residue_identifier[0], int(residue_identifier[1:])
            for residue in source_structure.get_residues():
                if (
                    THREE_TO_SINGLE_LETTER_CODES.get(residue.resname) == amino_acid
                ) and (residue.id[1] == position):
                    residues_to_copy.append(residue)
        target_residues = [
            residue
            for residue in target_structure.get_residues()
            if residue not in residues_to_copy
            and residue.id[1] <= len(self.reference_sequence)
            and residue.full_id[3][0] == " "
        ]
        new_residues = residues_to_copy + target_residues
        sorted_residues = list(sorted(new_residues, key=lambda residue: residue.id[1]))
        imputed_structure = self.new_structure_from_residues(sorted_residues)
        return imputed_structure

    def new_structure_from_residues(self, residues: List[Residue.Residue]):
        structure = Structure.Structure(id=0)
        model = Model.Model(id="0")
        chain = Chain.Chain(id="0")
        for residue in residues:
            try:
                chain.add(residue)
            except PDBConstructionException:
                continue
        model.add(chain)
        structure.add(model)
        return structure

    def write_pdb(self, path: Union[str, Path], structure: Structure.Structure):
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(path))


def region_is_subregion(region1: str, region2: str):
    start1, end1 = tuple(int(value) for value in region1.split("-"))
    start2, end2 = tuple(int(value) for value in region2.split("-"))
    return (start1 >= start2) & (end1 <= end2)


def download_pdb(pdb_id: str, path: Union[str, Path]):
    response = requests.get(f"https://files.rcsb.org/view/{pdb_id}.pdb")
    with open(Path(path) / f"{pdb_id}.pdb", "w") as f:
        f.write(response.text)


def download_alphafold_structure(path: Union[str, Path], uniprot_id: str):
    alphafold_db_key = "AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"
    uniprot_response = requests.get(
        f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    )
    uniprot_body = uniprot_response.text
    accession_lines = [
        line for line in uniprot_body.split("\n") if line.startswith("AC")
    ]
    pattern = re.compile(UNIPROT_ACCESSION_PATTERN)
    match = re.search(pattern, accession_lines[0])
    if match:
        accession = match.group(0)
    else:
        logger.info(f"No accession found for uniprot id: {uniprot_id}")
    meta_data_url = (
        f"https://alphafold.ebi.ac.uk/api/prediction/{accession}?key={alphafold_db_key}"
    )
    meta_data_response = requests.get(meta_data_url)
    meta_data = meta_data_response.json()[0]
    pdb_url = meta_data["pdbUrl"]
    pdb_response = requests.get(pdb_url)
    pdb_body = pdb_response.text
    with open(path, "w") as f:
        f.write(pdb_body)


def fetch_uniprot_sequence(uniprot_id: str):
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
    body = response.text
    sequence = "".join(body.split("\n")[1:])
    return sequence


def distance_of_reference_to_uniprot(reference: str, uniprot: str):
    mat = SubstitutionMatrix.std_protein_matrix()
    reference_sequence = ProteinSequence(reference)
    uniprot_sequence = ProteinSequence(uniprot)
    alignment = align_optimal(reference_sequence, uniprot_sequence, mat)[0]
    trace = alignment.trace
    mask = trace[:, 0] != -1
    relevant_uniprot_region = trace[mask][:, 1]
    relevant_uniprot_sequence = uniprot_sequence[relevant_uniprot_region]
    edit_distance = levenshtein(reference, relevant_uniprot_sequence.__str__())
    return edit_distance


def prepare_input_config(
    pqr_file,
    center,
    fine_grid_length,
    coarse_grid_length,
    resolution,
    modalities,
    conditions,
):
    output_directory = pqr_file.parent
    molecule_name = str(pqr_file.name).split(".")[0]
    write_options = {
        "charge": "charge",
        "charge_density": "qdens",
        "potential": "pot",
    }
    write_arguments = [write_options[modality] for modality in modalities]
    write_lines = "\n\t".join(
        [
            f"write {write_argument} dx {output_directory}/{molecule_name}_{modality}"
            for write_argument, modality in zip(write_arguments, modalities)
        ]
    )
    return f"""read
    mol pqr {pqr_file}
end
elec
    mg-auto
    mol 1
    fgcent {", ".join(map(str, center))}
    cgcent {", ".join(map(str, center))}
    fglen {", ".join(map(str, fine_grid_length))}
    cglen {", ".join(map(str, coarse_grid_length))}
    dime {", ".join(map(str, resolution))}
    lpbe
    bcfl sdh
    pdie 2.0
    sdie 78.0
    chgm spl2
    srfm smol
    swin 0.3
    temp {conditions["temperature"]}
    sdens 10.0
    calcenergy no
    calcforce no
    srad 1.4
    ion charge +1 conc 0.15 radius 2.0
    ion charge -1 conc 0.15 radius 1.8
    {write_lines}
end
quit
    """


def assign_split_by_mutated_positions(
    positions: frozenset,
    train_positions: List[frozenset],
    val_and_test_positions: List[frozenset],
):
    if positions in train_positions:
        return "train"
    elif positions in val_and_test_positions:
        return np.random.choice(["valid", "test"], p=[0.5, 0.5])
    else:
        raise ValueError("Positions not in allowed train, valid or test positions!")


def mutations_to_sequence(mutations: Union[str, List[str]], reference: str):
    reference_residues = list(reference)
    if isinstance(mutations, list):
        for mut in mutations:
            pos = int(mut[1:-1])
            aa_var = mut[-1]
            reference_residues[pos - 1] = aa_var
    else:
        mut = mutations
        pos = int(mut[1:-1])
        aa_var = mut[-1]
        reference_residues[pos - 1] = aa_var
    return "".join(reference_residues)


def determine_offset(residue_positions_a: dict, residue_positions_b: dict):
    matching_pairs = []
    for idx_a, char_a in residue_positions_a.items():
        for idx_b, char_b in residue_positions_b.items():
            if char_a == char_b:
                matching_pairs.append((idx_a, idx_b))
    offsets = defaultdict(int)
    for idx_a, idx_b in matching_pairs:
        offset = idx_a - idx_b
        offsets[offset] += 1
    if offsets:
        most_frequent_offset = max(offsets, key=offsets.get)
        return most_frequent_offset
    else:
        return None


def extract_conditions(row: pd.Series, condition_features):
    if condition_features is None:
        return DEFAULT_CONDITIONS
    assumed_conditions = DEFAULT_CONDITIONS.copy()
    extracted_conditions = {
        feature: row.get(feature, DEFAULT_CONDITIONS[feature])
        for feature in condition_features
    }
    assumed_conditions.update(extracted_conditions)
    return assumed_conditions


def parse_dx_file(path, map_shape):
    n_grid_points = math.prod(map_shape)
    if n_grid_points % 3 != 0:
        pad_dx_file(path, map_shape)
    try:
        electrostatics = (
            np.genfromtxt(
                path,
                skip_header=11,
                skip_footer=5,
                dtype=np.float32,
            )
            .flatten()[:n_grid_points]
            .reshape(map_shape)
        )
    except ValueError as e:
        logger.error(e)
        logger.error(f"Could not read APBS output file {path}")
        logger.info("Dropping variant.")
        electrostatics = np.full(map_shape, fill_value=np.nan, dtype=np.float32)
    return electrostatics


def parse_apbs_outputs(path, variant_id, modalities, grid_size, remove_after=True):
    map_shape = (len(modalities), *grid_size)
    dx_file_paths = [f"{path}/{variant_id}_{modality}.dx" for modality in modalities]
    n_grid_points = math.prod(grid_size)
    electrostatics = np.empty(map_shape, dtype=np.float32)
    for i, fp in enumerate(dx_file_paths):
        if n_grid_points % 3 != 0:
            pad_dx_file(fp, grid_size)
        electrostatics[i] = (
            np.genfromtxt(
                fp,
                skip_header=11,
                skip_footer=5,
                dtype=np.float32,
            )
            .flatten()[:n_grid_points]
            .reshape(1, *grid_size)
        )
        if remove_after:
            os.remove(fp)
    return electrostatics


def pad_dx_file(path, grid_size, n_headers=11):
    n_grid_points = math.prod(grid_size)
    line_number_to_pad = ceildiv(n_grid_points, 3) + n_headers
    padding_length = 3 - n_grid_points % 3
    with open(path, "r") as f:
        lines = f.readlines()
        line_to_pad = lines[line_number_to_pad - 1]
        padded_line = (
            line_to_pad.strip("\n")
            + "".join(["0.0 " for _ in range(padding_length)])
            + "\n"
        )
    lines[line_number_to_pad - 1] = padded_line
    with open(path, "w") as f:
        f.writelines(lines)


def ceildiv(a, b):
    return -(a // -b)


def update_environment_variables(shell: str):
    home_dir = os.environ.get("HOME")
    with open(f"{home_dir}/.{shell}rc", "r") as file:
        bashrc_contents = file.read()
    pattern = r"export\s+(\w+)\s*=\s*(.*)"
    matches = re.findall(pattern, bashrc_contents)
    env_vars = {"PATH": CURRENT_PATH}
    for match in matches:
        key = match[0]
        value = match[1].strip("\"'")
        if key == "PATH":
            value = value.removeprefix("$PATH:").removesuffix("$PATH")
            env_vars[key] = env_vars[key] + value
        else:
            env_vars[key] = value
    os.environ.update(env_vars)


class KMerFeaturizer:
    def __init__(self, k: int = 3):
        self.k = k
        self.sequence_length = None
        self.kmers = None

    def transform(self, sequences: Iterable):
        sequence_matrix = np.array(list(map(list, sequences)))
        n_sequences, self.sequence_length = sequence_matrix.shape
        kmerized_sequences = np.array(
            [self.kmerize(sequence) for sequence in sequence_matrix]
        )
        self.kmers = np.unique(kmerized_sequences.reshape(-1, self.k), axis=0)
        return np.array(
            [
                self.compute_kmer_frequencies(kmerized_sequence)
                for kmerized_sequence in kmerized_sequences
            ]
        )

    def kmerize(self, sequence):
        return np.array(
            [sequence[i : i + self.k] for i in range(self.sequence_length - self.k + 1)]
        )

    def compute_kmer_frequencies(self, kmerized_sequence: np.array):
        counts = np.sum(
            np.all(kmerized_sequence[:, np.newaxis, :] == self.kmers, axis=2), axis=0
        )
        return counts / self.sequence_length

    def cluster_sequences(self, sequences: np.ndarray, n_clusters: int) -> np.ndarray:
        featurizer = KMerFeaturizer()
        logger.info("Transforming sequences to kmer frequencies.")
        featurized_sequences = featurizer.transform(sequences)
        logger.info("Clustering transformed sequences.")
        _, labels, _ = k_means(
            featurized_sequences,
            n_clusters,
            n_init="auto",
            random_state=self.random_seed,
        )
        return labels

    def assign_train_1_val_2(self, sequences: np.ndarray):
        labels = self.cluster_sequences(sequences, n_clusters=2)
        train_label, val_label = [label for label, _ in Counter(labels).most_common()]
        return (
            sequences[labels == train_label],
            sequences[labels == val_label],
            np.ndarray([]),
        )

    def assign_train_1_val_1_test_2(self, sequences: np.ndarray):
        if self.train_ratio + self.valid_ratio != 1:
            raise ValueError(
                "Train ratio and test ration must sum up to 1 when "
                "assigning the largest cluster to train and valid!"
            )
        labels = self.cluster_sequences(sequences, n_clusters=2)
        majority_label, minority_label = [
            label for label, _ in Counter(labels).most_common()
        ]
        if np.sum(labels == majority_label) >= 2 * np.sum(labels == minority_label):
            train_and_val_label, test_label = (majority_label, minority_label)
        else:
            train_and_val_label, test_label = (minority_label, majority_label)
        train_and_val_sequences = sequences[labels == train_and_val_label]
        sizes = self.n_train_valid_test(len(train_and_val_sequences))
        random.shuffle(train_and_val_sequences)
        return (
            train_and_val_sequences[: sizes.n_train],
            train_and_val_sequences[sizes.n_train :],
            sequences[labels == test_label],
        )

    def assign_train_1_val_2_test_2(self, sequences: np.ndarray):
        labels = self.cluster_sequences(sequences, n_clusters=2)
        train_label, val_and_test_label = [
            label for label, _ in Counter(labels).most_common()
        ]
        val_and_test_sequences = sequences[labels == val_and_test_label]
        random.shuffle(val_and_test_sequences)
        return (
            sequences[labels == train_label],
            val_and_test_sequences[: int(len(val_and_test_sequences) / 2)],
            val_and_test_sequences[int(len(val_and_test_sequences) / 2) :],
        )

    def assign_train_1_val_2_test_3(self, sequences: np.ndarray):
        labels = self.cluster_sequences(sequences, n_clusters=3)
        train_label, val_label, test_label = [
            label for label, _ in Counter(labels).most_common()
        ]
        return (
            sequences[labels == train_label],
            sequences[labels == val_label],
            sequences[labels == test_label],
        )


class PredictionStacker:
    def __init__(self, meta_regressor_name: str = "ridge"):
        self.meta_regressor_name = meta_regressor_name
        self.meta_regressor = {
            "ridge": Ridge(),
            "svm": SVR(),
        }[self.meta_regressor_name]

    def fit(self, data: pd.DataFrame, models: List[str], target: str):
        base_predictions = np.array(
            [data[data["source"] == model][target].values for model in models]
        ).transpose()
        self.meta_regressor.fit(
            base_predictions, data[data["source"] == "ground_truth"][target]
        )

    def predict(self, data: pd.DataFrame, models: List[str], target: str):
        base_predictions = np.array(
            [data[data["source"] == model][target].values for model in models]
        ).transpose()
        return self.meta_regressor.predict(base_predictions)


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def set_matplotlib_styles(font_size=12, title_size=16, label_size=18, tick_size=16):
    plt.rc("font", size=font_size)
    plt.rc("axes", titlesize=title_size)
    plt.rc("axes", labelsize=label_size)
    plt.rc("xtick", labelsize=tick_size)
    plt.rc("ytick", labelsize=tick_size)
    sns.set_style("whitegrid")


def split_sel(data, mutation_col="mutation", ratio=0.1, seed=42):
    """
    Ensures that every mutated position in validation is also observed in train
    """
    np.random.seed(seed)
    max_test_size = int(ratio * len(data))
    positions = data[mutation_col].map(lambda x: x[1:-1]).unique()
    np.random.shuffle(positions)
    data["split"] = "train"
    for position in positions:
        if len(data[data["split"] == "valid"]) >= max_test_size:
            break
        corresponding_rows = data[data[mutation_col].str.contains(position)]
        if len(corresponding_rows) <= 1:
            continue
        else:
            data.loc[data[mutation_col].str.contains(position), "split"] = (
                np.random.choice(
                    ["train", "valid"],
                    size=len(corresponding_rows),
                    p=[1 - ratio, ratio],
                )
            )
    return data


def assign_ssm_folds(
    data,
    position_col="residue_number",
    n_folds=10,
    fold_name: str = "test_fold",
    random_seed=None,
):
    df = data.copy().reset_index(drop=True)
    if random_seed is not None:
        np.random.seed(random_seed)
    positions = data[position_col].unique()
    np.random.shuffle(positions)
    df[fold_name] = pd.Series(dtype="Int64")
    for position in positions:
        matching_indices = np.argwhere(
            (data[position_col] == position).values
        ).flatten()
        for fold_number, index in zip(
            itertools.cycle(np.random.permutation(n_folds)), matching_indices
        ):
            df.loc[index, fold_name] = fold_number
    return df


def assign_random_folds(
    data: pd.DataFrame, n_folds: int = 5, seed: int = None
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    df = data.copy()
    fold_col = f"fold_random_{n_folds}"
    folds = np.tile(np.arange(n_folds), np.ceil(len(df) / n_folds).astype(int))[
        : len(df)
    ]
    np.random.shuffle(folds)
    df[fold_col] = folds
    return df


def assign_modulo_folds(
    data: pd.DataFrame,
    mutation_col: str = "mutant",
    n_folds: int = 5,
    fold_start: int = 0,
) -> pd.DataFrame:
    df = data.copy()
    fold_col = f"fold_modulo_{n_folds}"
    mutated_positions = df[mutation_col].map(lambda x: int(x[1:-1])).values
    folds = (mutated_positions - 1 + fold_start) % n_folds
    df[fold_col] = folds
    return df


def assign_contiguous_folds(
    data: pd.DataFrame,
    mutation_col: str = "mutant",
    n_folds: int = 5,
    span: int = 5,
    fold_start: int = 0,
) -> pd.DataFrame:
    sequence_length = len(data["sequence"][0])
    assert np.ceil(sequence_length / n_folds) >= span, (
        f"Span is too large for a sequence of length {sequence_length} and {n_folds} folds"
    )
    df = data.copy()
    fold_col = f"fold_contiguous_{n_folds}"
    n_regions_per_fold = np.ceil(sequence_length / (span * n_folds)).astype(int)
    mutated_positions = df[mutation_col].map(lambda x: int(x[1:-1])).values
    position_fold_ids = np.tile(
        np.repeat(np.arange(n_folds) + fold_start, span) % n_folds, n_regions_per_fold
    )[:sequence_length]
    folds = list(map(lambda x: position_fold_ids[x - 1], mutated_positions))
    df[fold_col] = folds
    return df


def sets_share_positions(set_1: tuple, set_2: tuple) -> bool:
    return any([x == y for x in set_1 for y in set_2])


def group_position_non_overlapping(combinations: List[tuple]) -> Dict[tuple, int]:
    initial_grouping = dict(Counter(combinations))
    combinations_have_overlap = True
    new_grouping = initial_grouping.copy()
    while combinations_have_overlap:
        unique_sets = list(new_grouping.keys())
        overlap_indicator = np.zeros(
            (len(unique_sets), len(unique_sets)), dtype=np.int32
        )
        for i, set_1 in enumerate(unique_sets):
            for j, set_2 in enumerate(unique_sets):
                overlap_indicator[i, j] = int(sets_share_positions(set_1, set_2))
        combinations_have_overlap = np.sum(np.triu(overlap_indicator, k=1)) > 0
        set_with_most_overlap_idx = np.argmax(np.sum(overlap_indicator, axis=0))
        sets_to_combine = [
            unique_sets[r]
            for r in np.where(overlap_indicator[set_with_most_overlap_idx])[0]
        ]
        new_combination = tuple(
            set(position for combination in sets_to_combine for position in combination)
        )
        counts = 0
        for old_set in sets_to_combine:
            counts += new_grouping[old_set]
            new_grouping.pop(old_set, None)
        new_grouping[new_combination] = counts
    return new_grouping
