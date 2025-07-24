import pytest
import shutil

from biopandas.pdb import PandasPdb

from chargenet.pipelines import APBSElectrostaticMapper, ChargeNet
from chargenet.constants import THREE_TO_SINGLE_LETTER_CODES


FOLDX_AVAILABLE = True if shutil.which("foldx") is not None else False


class TestPQRSolver:
    @pytest.fixture(scope="session")
    def structure_sequence_data(self, mini_pdb_file_path):
        return (
            PandasPdb()
            .read_pdb(str(mini_pdb_file_path))
            .df["ATOM"][["residue_number", "residue_name", "chain_id"]]
            .drop_duplicates("residue_number")
        )

    @pytest.fixture(scope="session")
    def reference_sequence(self, structure_sequence_data):
        return "".join(
            structure_sequence_data["residue_name"]
            .map(lambda x: THREE_TO_SINGLE_LETTER_CODES[x] if x != "HOH" else "")
            .to_list()
        )

    @pytest.fixture(scope="session")
    def electrostatic_mapper(self, mini_pdb_file_path, reference_sequence):
        return APBSElectrostaticMapper(
            pdb_file_path=str(mini_pdb_file_path),
            reference_sequence=reference_sequence,
            batch_size=8,
            channel_configuration="all",
            n_cores=4,
        )

    def test_pqr_solver_logs_stderr(
        self, electrostatic_mapper, mini_protein_data_frame
    ):
        data = mini_protein_data_frame
        electrostatic_mapper.pqr_solver.psize.center = ["error"]
        with electrostatic_mapper as mapper:
            with pytest.raises(RuntimeError) as exc_info:
                mapper.run(data)
            exception = exc_info.value
            print(str(exception))
            assert "Error: " in str(exception)

    @pytest.fixture(scope="session")
    def chargenet(self, mini_pdb_file_path, reference_sequence):
        chargenet = ChargeNet(
            pdb_file_path=str(mini_pdb_file_path),
            reference_sequence=reference_sequence,
            target="a",
            mutagenesis_tool="foldx" if FOLDX_AVAILABLE else "pymol",
            epochs=2,
        )
        return chargenet

    def test_chargenet_can_run(self, chargenet, mini_protein_data_frame):
        data = mini_protein_data_frame
        chargenet.run(data)

    def test_chargenet_can_predict(self, chargenet, mini_protein_data_frame):
        data = mini_protein_data_frame
        predictions = chargenet.predict(data)
        assert predictions.shape == (len(mini_protein_data_frame),)
