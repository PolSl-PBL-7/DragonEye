from utils.range_types import ClosedRange, OpenRange, Range


def test_left_of_open_range():
    a = 2
    assert a not in OpenRange(3, 5)


def test_on_left_boundary_open_range():
    a = 2
    assert a not in OpenRange(2, 5)


def test_right_of_open_range():
    a = 6
    assert a not in OpenRange(3, 5)


def test_on_right_boundary_open_range():
    a = 5
    assert a not in OpenRange(2, 5)


def test_in_open_range():
    a = 4
    assert a in OpenRange(2, 5)


def test_left_of_closed_range():
    a = 2
    assert a not in ClosedRange(3, 5)


def test_on_left_boundary_closed_range():
    a = 2
    assert a in ClosedRange(2, 5)


def test_right_of_closed_range():
    a = 6
    assert a not in ClosedRange(3, 5)


def test_on_right_boundary_closed_range():
    a = 5
    assert a in ClosedRange(2, 5)


def test_in_closed_range():
    a = 4
    assert a in ClosedRange(2, 5)


def test_left_of_default_range():
    a = 2
    assert a not in Range(3, 5)


def test_on_left_boundary_default_range():
    a = 2
    assert a in Range(2, 5)


def test_right_of_default_range():
    a = 6
    assert a not in Range(3, 5)


def test_on_right_boundary_default_range():
    a = 5
    assert a not in Range(2, 5)


def test_in_default_range():
    a = 4
    assert a in Range(2, 5)


###
def test_on_left_boundary_left_exclusive_right_inclusive_range():
    a = 3
    assert a not in Range(3, 5, left_inclusive=False, right_inclusive=True)


def test_on_right_boundary_left_exclusive_right_inclusive_range():
    a = 5
    assert a in Range(3, 5, left_inclusive=False, right_inclusive=True)


def test_in_left_exclusive_right_inclusive_range():
    a = 4
    assert a in Range(3, 5, left_inclusive=False, right_inclusive=True)


def test_right_of_exclusive_right_inclusive_range():
    a = 6
    assert a not in Range(3, 5, left_inclusive=False, right_inclusive=True)


def test_left_of_exclusive_right_inclusive_range():
    a = 2
    assert a not in Range(3, 5, left_inclusive=False, right_inclusive=True)
