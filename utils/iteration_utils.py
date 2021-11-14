def check_equal_lengths(*args):
    "Checks whether all args are iterables and have the same length"
    length = None
    for arg in args:
        if not hasattr(arg, "__iter__"):
            raise Exception("All args should be iterables")
        if length is not None:
            if len(arg) != length:
                return False
        else:
            length = len(arg)
    return True
