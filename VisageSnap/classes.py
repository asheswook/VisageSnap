from dataclasses import dataclass
import numpy as np
from sklearn.semi_supervised import LabelPropagation

@dataclass
class Face():
    label: str
    encodings: np.ndarray
    filenames: list

@dataclass
class GlobalState:
    faces: list[Face]
    label: dict
    model: LabelPropagation

@dataclass
class Directory:
    labeled: str
    unlabeled: str
    model: str
    predict: str

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