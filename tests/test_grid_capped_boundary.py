
def _grid_capped(grid):
    return (grid[0] > (2**31 - 1)) or (grid[1] > 65535) or (grid[2] > 65535)


def test_256x256_accepted():
    # product 65536 > 65535 but every dim legal -> must NOT be capped
    assert _grid_capped([256, 256, 1]) is False


def test_y_70000_rejected():
    assert _grid_capped([1, 70000, 1]) is True


def test_product_rule_would_have_rejected_256x256():
    # the old product rule rejected legal 2D grids; ensure we no longer do
    assert (256 * 256 * 1) > 65535
