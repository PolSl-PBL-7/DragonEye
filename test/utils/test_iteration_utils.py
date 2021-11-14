import pytest

from utils.iteration_utils import check_equal_lengths


def test_correct():
    if not check_equal_lengths((1, 2, 3), (4, 5, 6)):
        pytest.fail()


def test_incorrect_lengths():
    result = False
    try:
        result = check_equal_lengths((1, 2, 3), (2, 3))
    except Exception() as ex:
        pass
    if result:
        pytest.fail("Not equal lengths of input args but function did not return error")


def test_not_iterables():
    try:
        check_equal_lengths((1, 2, 3), 4)
    except:
        return
    pytest.fail("All args have to be iterables")
