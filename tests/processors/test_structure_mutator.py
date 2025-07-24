import re
from loguru import logger
import pytest

from biopandas.pdb import PandasPdb

from chargenet.processors import StructureMutator
from chargenet.constants import THREE_TO_SINGLE_LETTER_CODES


def extract_mutations(strings):
    mutation_lists = []
    for s in strings:
        match = re.search(r"Using foldx to make mutations: (.*?);", s)
        if match:
            mutations = [mutation.strip() for mutation in match.group(1).split(",")]
            mutation_lists.append(mutations)
    return mutation_lists


class TestStructureMutator:
    @pytest.fixture(scope="session")
    def solver(self, mini_pdb_file_path, reference_sequence):
        return StructureMutator(
            pdb_file_path=mini_pdb_file_path,
            reference_sequence=reference_sequence,
            mutagenesis_tool="foldx",
        )

    @pytest.fixture(scope="session")
    def structure_offset_from_reference(self):
        return 2

    @pytest.fixture(scope="session")
    def structure_sequence_data(self, mini_pdb_file_path):
        return (
            PandasPdb()
            .read_pdb(str(mini_pdb_file_path))
            .df["ATOM"][["residue_number", "residue_name", "chain_id"]]
            .drop_duplicates("residue_number")
        )

    @pytest.fixture(scope="session")
    def reference_sequence(
        self, structure_sequence_data, structure_offset_from_reference
    ):
        return "".join(
            structure_sequence_data["residue_name"]
            .map(lambda x: THREE_TO_SINGLE_LETTER_CODES[x] if x != "HOH" else "")
            .to_list()
        )[structure_offset_from_reference:]

    @pytest.fixture(scope="session")
    def chain_id(self, structure_sequence_data):
        return structure_sequence_data["chain_id"].iloc[0]

    @pytest.fixture(scope="session")
    def loguru_caplog(self):
        class LoguruCaplog:
            def __init__(self):
                self.messages = []

            def __enter__(self):
                self.handler_id = logger.add(self._log_handler, format="{message}")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                logger.remove(self.handler_id)

            def _log_handler(self, message):
                self.messages.append(message)

        return LoguruCaplog()

    def test_correct_mutagenesis_when_leading_residues_in_structure(
        self,
        solver,
        mini_pdb_file_path,
        mini_protein_data_frame,
        reference_sequence,
        chain_id,
        structure_offset_from_reference,
        loguru_caplog,
        tmpdir_session,
    ):
        data = mini_protein_data_frame[:3]
        data["sequence"] = data["sequence"].map(
            lambda x: x[structure_offset_from_reference:]
        )
        actual_mutations = []
        for variant_sequence in data["sequence"]:
            actual_mutations.append(
                [
                    f"{aa_ref}{chain_id}{i + 1 + structure_offset_from_reference}{aa_var}"
                    for i, (aa_ref, aa_var) in enumerate(
                        zip(reference_sequence, variant_sequence)
                    )
                    if aa_ref != aa_var
                ]
            )
        with loguru_caplog as caplog:
            solver.generate_structures(data=data, path=tmpdir_session)
        logged_mutations = extract_mutations(caplog.messages)
        for actual_variant_mutations, logged_variant_mutations in zip(
            actual_mutations, logged_mutations
        ):
            if logged_variant_mutations[0][0] == logged_variant_mutations[0][-1]:
                logged_variant_mutations = []
            assert set(actual_variant_mutations) == set(logged_variant_mutations)
