import numpy as np

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