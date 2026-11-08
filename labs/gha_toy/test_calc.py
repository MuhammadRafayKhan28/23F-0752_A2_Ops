from calc import add, is_ai


def test_add():
    assert add(2, 3) == 5
    # RED run: change the line above to `assert add(2, 3) == 6`, push, watch it fail; revert -> GREEN


def test_threshold():
    assert is_ai(0.7) and not is_ai(0.3)
