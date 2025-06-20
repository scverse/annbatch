import numpy as np
from arrayloaders.io.utils import sample_rows


def test_sample_rows_basic():
    """
    This test checks the sample_rows function without shuffling.

    Verifies that the function yields the expected (x, y) pairs
    when given lists of arrays and labels, and shuffle is set to False.
    """
    x_list = [np.arange(6).reshape(3, 2), np.arange(8, 16).reshape(4, 2)]
    y_list = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]
    # Test without shuffling
    result = list(sample_rows(x_list, y_list, shuffle=False))
    expected = [
        (np.array([0, 1]), 0),
        (np.array([2, 3]), 1),
        (np.array([4, 5]), 2),
        (np.array([8, 9]), 3),
        (np.array([10, 11]), 4),
        (np.array([12, 13]), 5),
        (np.array([14, 15]), 6),
    ]
    for (x, y), (ex, ey) in zip(result, expected):
        np.testing.assert_array_equal(x, ex)
        assert y == ey


def test_sample_rows_shuffle():
    """
    This test checks the sample_rows function with shuffling enabled.

    Ensures that all unique (x, y) pairs are present in the result,
    regardless of order, when shuffle is set to True.
    """
    x_list = [np.arange(6).reshape(3, 2), np.arange(8, 16).reshape(4, 2)]
    y_list = [np.array([0, 1, 2]), np.array([3, 4, 5, 6])]
    result = list(sample_rows(x_list, y_list, shuffle=True))
    # Should have all unique pairs, order may differ
    assert sorted([tuple(x) + (y,) for x, y in result]) == [
        (0, 1, 0),
        (2, 3, 1),
        (4, 5, 2),
        (8, 9, 3),
        (10, 11, 4),
        (12, 13, 5),
        (14, 15, 6),
    ]
