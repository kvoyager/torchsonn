import pytest

from torchsonn.types import CriterionType, LayerCreationError, RefFunctionType


class TestRefFunctionType:
    @pytest.mark.parametrize(
        "value,name",
        [
            (RefFunctionType.rfUnknown, "Unknown"),
            (RefFunctionType.rfLinear, "Linear"),
            (RefFunctionType.rfLinearCov, "LinearCov"),
            (RefFunctionType.rfQuadratic, "Quadratic"),
            (RefFunctionType.rfCubic, "Cubic"),
            (RefFunctionType.rfPolyQuadratic, "PolyQuadratic"),
            (RefFunctionType.rfLegendre, "Legendre"),
            (RefFunctionType.rfChebyshev, "Chebyshev"),
        ],
    )
    def test_get_name(self, value, name):
        assert RefFunctionType.get_name(value) == name

    def test_get_name_invalid_returns_unknown(self):
        # passing an unrelated enum-like value should fall through to "Unknown"
        class Fake:
            pass

        assert RefFunctionType.get_name(Fake()) == "Unknown"

    @pytest.mark.parametrize(
        "arg,expected",
        [
            ("linear", RefFunctionType.rfLinear),
            ("linear_cov", RefFunctionType.rfLinearCov),
            ("quadratic", RefFunctionType.rfQuadratic),
            ("cubic", RefFunctionType.rfCubic),
            ("polyquad", RefFunctionType.rfPolyQuadratic),
            ("legendre", RefFunctionType.rfLegendre),
            ("chebyshev", RefFunctionType.rfChebyshev),
            ("rbf", RefFunctionType.rfRBF),
        ],
    )
    def test_get_from_string(self, arg, expected):
        assert RefFunctionType.get(arg) == expected

    def test_get_passthrough(self):
        assert RefFunctionType.get(RefFunctionType.rfLinear) is RefFunctionType.rfLinear

    @pytest.mark.parametrize("arg", ["not-a-name", "lcov", "quad", "leg", "cheb", "gauss"])
    def test_get_invalid_raises(self, arg):
        with pytest.raises(ValueError):
            RefFunctionType.get(arg)


class TestCriterionType:
    @pytest.mark.parametrize(
        "value,name",
        [
            (CriterionType.cmpValidate, "validate error comparison"),
            (CriterionType.cmpBias, "bias error comparison"),
            (CriterionType.cmpComb_validate_bias, "bias and validate error comparison"),
            (CriterionType.cmpComb_bias_retrain, "bias error comparison with retrain"),
        ],
    )
    def test_get_name(self, value, name):
        assert CriterionType.get_name(value) == name

    def test_get_name_unknown(self):
        class Fake:
            pass

        assert CriterionType.get_name(Fake()) == "Unknown"

    @pytest.mark.parametrize(
        "arg,expected",
        [
            ("validate", CriterionType.cmpValidate),
            ("bias", CriterionType.cmpBias),
            ("validate_bias", CriterionType.cmpComb_validate_bias),
            ("bias_retrain", CriterionType.cmpComb_bias_retrain),
        ],
    )
    def test_get_from_string(self, arg, expected):
        assert CriterionType.get(arg) == expected

    def test_get_passthrough(self):
        c = CriterionType.cmpBias
        assert CriterionType.get(c) is c

    @pytest.mark.parametrize("arg", ["nope", "bias_refit"])
    def test_get_invalid_raises(self, arg):
        with pytest.raises(ValueError):
            CriterionType.get(arg)


def test_layer_creation_error_carries_index():
    err = LayerCreationError("boom", layer_index=7)
    assert err.layer_index == 7
    assert "boom" in str(err)
    assert isinstance(err, Exception)
