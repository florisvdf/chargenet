import os
import re
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("CHARGENET_ROOT", Path(__file__).resolve().parent.parent.parent)
)
DATA_ROOT = (
    Path(
        os.environ.get("CHARGENET_ROOT", Path(__file__).resolve().parent.parent.parent)
    )
    / "data"
)
ROTABASE_LOCATION = Path(
    os.environ.get("ROTABASE_LOCATION", PROJECT_ROOT / "software/rotabase.txt")
)
AA_GAP_PATTERN = re.compile("^[-ACDEFGHIKLMNPQRSTVWY]+$", flags=re.IGNORECASE)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREE_TO_SINGLE_LETTER_CODES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
SINGLE_TO_THREE_LETTER_CODES = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
UNIPROT_ACCESSION_PATTERN = (
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
)
CONDITION_COL_NAMES = ["temperature", "ph"]
DEFAULT_TEMPERATURE = 298
DEFAULT_PH = 7
DEFAULT_CONDITIONS = {
    "temperature": DEFAULT_TEMPERATURE,
    "ph": DEFAULT_PH,
}
