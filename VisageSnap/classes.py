from dataclasses import dataclass
import numpy as np

@dataclass
class Face():
    label: str
    encodings: np.ndarray
    filenames: list

@dataclass
class From():
    LABEL = "Label"
    FILENAME = "Filename"

@dataclass
class To():
    NAME = "Name"
    NUMBER = "Number"

@dataclass
class As():
    LABELED = True
    UNLABELED = False