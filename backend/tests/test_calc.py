import pytest

from app.calc import CalcError, calculate


def test_growth_rate():
    assert calculate("(5082 - 4941) / 4941 * 100") == pytest.approx(2.853673, rel=1e-6)


def test_strips_commas_and_dollars():
    assert calculate("$5,082 + 1,000") == 6082


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "a + 1", "2 ** 999", "1/0", "[1,2]"])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalcError):
        calculate(expr)
