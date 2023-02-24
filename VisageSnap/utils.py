def gen(target: list[any]) -> any:
    """
    This function is a generator.
    
    Parameters
    ----------
    target (list) : target list.
    """
    assert isinstance(target, list), "target must be a list."
    for i in target:
        yield i