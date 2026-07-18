import numpy as np
import pandas as pd
import pytest
import torch

from torchsonn.data.dataset import SONNDataset
from torchsonn.data.preprocessing import (
    DataSetType,
    SequenceTypeSet,
    predict_preprocessing,
    set_split_types,
    split_dataset,
    train_preprocessing,
)


class TestSONNDataset:
    def test_len_no_split(self):
        x = torch.arange(10).reshape(10, 1).float()
        y = torch.arange(10).float()
        ds = SONNDataset(x, y, split=None)
        assert len(ds) == 10
        a, b = ds[3]
        assert int(a.item()) == 3
        assert int(b.item()) == 3

    def test_split_zero_takes_even_rows(self):
        x = np.arange(10).reshape(10, 1)
        y = np.arange(10)
        ds = SONNDataset(x, y, split=0)
        assert len(ds) == 5
        # index i → 2i
        first_x, _ = ds[2]
        assert int(first_x.item() if hasattr(first_x, "item") else first_x) == 4

    def test_split_one_takes_odd_rows(self):
        x = np.arange(9).reshape(9, 1)
        y = np.arange(9)
        ds = SONNDataset(x, y, split=1)
        assert len(ds) == 4
        first_x, _ = ds[1]
        # index 1 → 2*1+1 = 3
        assert int(first_x.item() if hasattr(first_x, "item") else first_x) == 3

    def test_target_none_returns_none(self):
        x = np.arange(6).reshape(6, 1)
        ds = SONNDataset(x, None)
        _, y = ds[0]
        assert y is None

    def test_invalid_split(self):
        with pytest.raises(AssertionError):
            SONNDataset(np.zeros((4, 1)), None, split=2)


class TestTrainPreprocessing:
    def test_numpy_passthrough(self):
        x = np.random.rand(10, 3)
        y = np.random.rand(10)
        x2, y2 = train_preprocessing(x, y, feature_names=None)
        assert x2.shape == x.shape
        assert y2.shape == y.shape

    def test_pandas_inputs(self):
        x = pd.DataFrame(np.random.rand(8, 2), columns=["a", "b"])
        y = pd.Series(np.random.rand(8))
        x2, y2 = train_preprocessing(x, y, feature_names=["a", "b"])
        assert isinstance(x2, np.ndarray)
        assert isinstance(y2, np.ndarray)
        assert y2.shape == (8,)

    def test_pandas_dataframe_target(self):
        x = pd.DataFrame(np.random.rand(8, 2))
        y_df = pd.DataFrame(np.random.rand(8, 1))
        _, y2 = train_preprocessing(x, y_df, feature_names=None)
        assert y2.shape == (8,)

    def test_list_target_converted(self):
        x = np.random.rand(5, 2)
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, y2 = train_preprocessing(x, y, feature_names=None)
        assert isinstance(y2, np.ndarray)
        assert y2.shape == (5,)

    def test_target_row_vector_flattens(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(1, 5)
        _, y2 = train_preprocessing(x, y, feature_names=None)
        assert y2.shape == (5,)

    def test_target_column_vector_flattens(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 1)
        _, y2 = train_preprocessing(x, y, feature_names=None)
        assert y2.shape == (5,)

    def test_target_2d_invalid(self):
        x = np.random.rand(5, 2)
        y = np.random.rand(5, 2)
        with pytest.raises(ValueError, match="data_y dimension"):
            train_preprocessing(x, y, feature_names=None)

    def test_x_wrong_ndim(self):
        x = np.random.rand(5)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="data_x dimension"):
            train_preprocessing(x, y, feature_names=None)

    def test_x_transposed(self):
        # (features, samples) instead of (samples, features) — should auto-transpose
        x = np.random.rand(2, 7)
        y = np.random.rand(7)
        x2, _ = train_preprocessing(x, y, feature_names=None)
        assert x2.shape == (7, 2)

    def test_mismatched_lengths(self):
        x = np.random.rand(4, 3)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="not equal"):
            train_preprocessing(x, y, feature_names=None)

    def test_too_few_features(self):
        x = np.random.rand(5, 1)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="not less than two"):
            train_preprocessing(x, y, feature_names=None)

    def test_too_few_samples(self):
        x = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        with pytest.raises(ValueError, match="number of samples"):
            train_preprocessing(x, y, feature_names=None)

    def test_x_from_list(self):
        # exercise the "data_x not ndarray" path: a plain list of lists
        x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        y = [1.0, 2.0, 3.0]
        x2, _ = train_preprocessing(x, y, feature_names=None)
        assert isinstance(x2, np.ndarray)

    def test_feature_names_mismatch(self):
        x = np.random.rand(5, 3)
        y = np.random.rand(5)
        with pytest.raises(ValueError, match="feature_names"):
            train_preprocessing(x, y, feature_names=["a", "b"])  # 2 != 3


class TestPredictPreprocessing:
    def test_basic(self):
        x = np.random.rand(6, 4)
        x2, n = predict_preprocessing(x, n_features=4)
        assert x2.shape == x.shape
        assert n == 6

    def test_pandas_dataframe(self):
        x = pd.DataFrame(np.random.rand(6, 3))
        x2, n = predict_preprocessing(x, n_features=3)
        assert isinstance(x2, np.ndarray)
        assert n == 6

    def test_transpose(self):
        x = np.random.rand(3, 6)  # (n_features=3, samples=6)
        x2, n = predict_preprocessing(x, n_features=3)
        assert x2.shape == (6, 3)
        assert n == 6

    def test_x_from_list(self):
        x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        x2, n = predict_preprocessing(x, n_features=3)
        assert isinstance(x2, np.ndarray)
        assert n == 2

    def test_bad_ndim(self):
        with pytest.raises(ValueError, match="data_x dimension"):
            predict_preprocessing(np.random.rand(5), n_features=2)

    def test_feature_mismatch(self):
        x = np.random.rand(5, 4)
        with pytest.raises(ValueError, match="number of features"):
            predict_preprocessing(x, n_features=3)


class TestSequenceTypeSet:
    def test_mode1_helpers(self):
        assert SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode1)
        assert SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode3_1)
        assert SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode4_1)
        assert not SequenceTypeSet.is_mode1_type(SequenceTypeSet.sqMode2)

    def test_mode2_helpers(self):
        assert SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode2)
        assert SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode3_2)
        assert SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode4_2)
        assert not SequenceTypeSet.is_mode2_type(SequenceTypeSet.sqMode1)


class TestSetSplitTypes:
    def test_random(self):
        types = set_split_types(SequenceTypeSet.sqRandom, 100)
        assert types.shape == (100,)
        assert set(np.unique(types).tolist()).issubset(
            {int(DataSetType.dsTrain), int(DataSetType.dsValidate)}
        )

    @pytest.mark.parametrize(
        "seq_type",
        [
            SequenceTypeSet.sqMode1,
            SequenceTypeSet.sqMode3_1,
            SequenceTypeSet.sqMode4_1,
            SequenceTypeSet.sqMode2,
            SequenceTypeSet.sqMode3_2,
            SequenceTypeSet.sqMode4_2,
        ],
    )
    def test_deterministic_modes(self, seq_type):
        types = set_split_types(seq_type, 12)
        assert types.shape == (12,)
        # Both labels appear in a 12-row run
        assert int(DataSetType.dsTrain) in types
        assert int(DataSetType.dsValidate) in types

    def test_unknown_mode_raises(self):
        class Fake:
            value = 9999

        with pytest.raises(ValueError):
            set_split_types(Fake(), 10)  # type: ignore[arg-type]


class TestSplitDataset:
    def test_partitions_cover_all_rows(self):
        x = np.arange(40).reshape(20, 2)
        y = np.arange(20)
        tx, ty, vx, vy = split_dataset(x, y, SequenceTypeSet.sqMode1)
        # together the partitions should account for the full dataset
        assert tx.shape[0] + vx.shape[0] == 20
        assert ty.shape[0] + vy.shape[0] == 20
        # feature dim preserved
        assert tx.shape[1] == 2 and vx.shape[1] == 2
