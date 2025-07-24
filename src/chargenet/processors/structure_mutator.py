import shutil
import subprocess
from loguru import logger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, List

import pandas as pd
from biopandas.pdb import PandasPdb

try:
    from pymol import cmd

    PYMOL_AVAILABLE = True
except ModuleNotFoundError:
    PYMOL_AVAILABLE = False

from chargenet.utils import suppress_stdout_stderr, determine_offset
from chargenet.constants import (
    ROTABASE_LOCATION,
    THREE_TO_SINGLE_LETTER_CODES,
    SINGLE_TO_THREE_LETTER_CODES,
)


class StructureMutator:
    def __init__(
        self, pdb_file_path: Path, reference_sequence, mutagenesis_tool: str = "foldx"
    ):
        self.pdb_file_path = pdb_file_path
        self.reference_sequence = reference_sequence
        self.mutagenesis_tool = mutagenesis_tool
        self.structure_name = str(pdb_file_path.name).removesuffix(".pdb")
        self.structure_sequence_data = (
            PandasPdb()
            .read_pdb(str(self.pdb_file_path))
            .df["ATOM"][["residue_number", "residue_name", "chain_id"]]
            .drop_duplicates("residue_number")
        )
        self.structure_sequence = "".join(
            self.structure_sequence_data["residue_name"]
            .map(lambda x: THREE_TO_SINGLE_LETTER_CODES.get(x, ""))
            .to_list()
        )
        self.structure_offset_from_reference = (
            self.determine_structure_offset_from_reference()
        )

    def determine_structure_offset_from_reference(self):
        reference_residues_positions = {
            i + 1: aa_ref for i, aa_ref in enumerate(self.reference_sequence)
        }
        structure_residues_positions = {
            self.structure_sequence_data["residue_number"].iloc[
                i
            ]: THREE_TO_SINGLE_LETTER_CODES.get(
                self.structure_sequence_data["residue_name"].iloc[i]
            )
            for i in range(len(self.structure_sequence_data))
        }
        offset = determine_offset(
            structure_residues_positions, reference_residues_positions
        )
        return offset

    def generate_structures(self, data: pd.DataFrame, path: Path):
        all_mutations = self.sequences_to_mutations(data["sequence"])
        with TemporaryDirectory() as temp_dir:
            shutil.copy(self.pdb_file_path, temp_dir)
            output_paths = [
                path / f"{self.structure_name}_{idx}.pdb" for idx in data.index
            ]
            {"foldx": self.foldx_mutate, "pymol": self.pymol_mutate}[
                self.mutagenesis_tool
            ](temp_dir, all_mutations)
            for variant_suffix, output_path in enumerate(output_paths, start=1):
                shutil.move(
                    f"{temp_dir}/{self.structure_name}_{variant_suffix}.pdb",
                    output_path,
                )

    def sequences_to_mutations(self, sequences: Union[pd.DataFrame, list]):
        all_mutations = []
        for sequence in sequences:
            mutations = []
            for i, (aa_var, aa_ref) in enumerate(
                zip(
                    sequence,
                    self.structure_sequence[self.structure_offset_from_reference :],
                )
            ):
                if aa_ref != aa_var:
                    pdb_position, chain_id = self.structure_sequence_data[
                        ["residue_number", "chain_id"]
                    ].iloc[i + self.structure_offset_from_reference]
                    mutations.append(f"{aa_ref}{chain_id}{pdb_position}{aa_var}")
            if mutations == []:
                pdb_position, chain_id = self.structure_sequence_data[
                    ["residue_number", "chain_id"]
                ].iloc[0]
                aa_ref = aa_var = self.structure_sequence[0]
                mutations = [f"{aa_ref}{chain_id}{pdb_position}{aa_var}"]
            all_mutations.append(mutations)
        return all_mutations

    def foldx_mutate(self, cwd: Union[Path, str], all_mutations: List[List[str]]):
        mutant_file_path = Path(cwd) / "individual_list.txt"
        with open(mutant_file_path, "w") as fh:
            for mutations in all_mutations:
                mutant_file_body = ",".join(mutations) + ";\n"
                logger.info(
                    f"Using {self.mutagenesis_tool} to make mutations: {mutant_file_body}"
                )
                fh.write(mutant_file_body)
        command = [
            "foldx",
            "-c",
            "BuildModel",
            "--pdb",
            str(self.pdb_file_path.name),
            "--mutant-file",
            "individual_list.txt",
            "--rotabase",
            str(ROTABASE_LOCATION),
        ]
        try:
            _ = subprocess.run(
                command,
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command '{e.cmd}' failed with return code {e.returncode}. "
                f"Error: {e.stderr.decode().strip()}. "
                f"Full stdout: {e.stdout.decode().strip()}"
            )

    def pymol_mutate(self, cwd: Union[Path, str], all_mutations: List[List[str]]):
        if not PYMOL_AVAILABLE:
            raise RuntimeError("PyMol not available!")
        with suppress_stdout_stderr():
            cmd.set("pdb_reformat_names_mode", 2)
            cmd.load(Path(cwd) / self.pdb_file_path.name, "reference_structure")
            for i, mutations in enumerate(all_mutations, start=1):
                mutations_to_log = ",".join(mutations) + ";\n"
                logger.info(
                    f"Using {self.mutagenesis_tool} to make mutations: {mutations_to_log}"
                )
                cmd.copy("variant_structure", "reference_structure")
                for mutant in mutations:
                    _, chain_id, pos, aa_var = (
                        mutant[0],
                        mutant[1],
                        int(mutant[2:-1]),
                        mutant[-1],
                    )
                    cmd.wizard("mutagenesis")
                    cmd.refresh_wizard()
                    cmd.get_wizard().set_mode(SINGLE_TO_THREE_LETTER_CODES[aa_var])
                    cmd.get_wizard().do_select(f"variant_structure//{chain_id}/{pos}/")
                    cmd.frame(1)
                    cmd.get_wizard().apply()
                    cmd.set_wizard()
                cmd.save(
                    Path(cwd) / f"{self.structure_name}_{i}.pdb",
                    "variant_structure",
                    state=-1,
                    format="pdb",
                )
                cmd.delete("variant_structure")
            cmd.reinitialize()
