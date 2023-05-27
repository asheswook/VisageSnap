import numpy as np
import os

def gen(target: list[any]) -> any:
    """
    This function is a generator.
    
    Parameters
    ----------
    target (list) : target list.
    """
    assert isinstance(target, list | np.ndarray), "target must be a list or numpy.ndarray."
    for i in target:
        yield i

def absp(value: str) -> str:
    return os.path.join(os.getcwd(), value)

def isimage(filename: str) -> bool:
    """
    This function checks if the file is an image file.

    Parameters
    ----------
    filename (str) : target filename.
    """
    assert isinstance(filename, str), "filename must be a string."

    list = [
        ".jpg",
        ".png",
        ".jpeg"
        ]

    for i in list:
        if filename.endswith(i):
            return True
    return False