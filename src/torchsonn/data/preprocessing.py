from enum import IntEnum
from typing import Sequence

import numpy as np
import pandas as pd


class DataSetType(IntEnum):
    """Tag attached to each row of a dataset by `set_split_types`."""
    dsTrain = 0
    dsValidate = 1


class SequenceTypeSet(IntEnum):
    """How rows are assigned to train vs. validate.

    sqRandom    — random Bernoulli per row.
    sqMode1     — every 2nd row is validate.
    sqMode3_1   — every 3rd row is validate.
    sqMode4_1   — every 4th row is validate.
    sqMode2     — every 2nd row is train (inverse of sqMode1).
    sqMode3_2   — every 3rd row is train (inverse of sqMode3_1).
    sqMode4_2   — every 4th row is train (inverse of sqMode4_1).
    """
    sqRandom = 0
    sqMode1 = 1
    sqMode3_1 = 2
    sqMode4_1 = 3
    sqMode2 = 4
    sqMode3_2 = 5
    sqMode4_2 = 6

    @classmethod
    def is_mode1_type(cls, seq_type: "SequenceTypeSet") -> bool:
        return seq_type in (cls.sqMode1, cls.sqMode3_1, cls.sqMode4_1)

    @classmethod
    def is_mode2_type(cls, seq_type: "SequenceTypeSet") -> bool:
        return seq_type in (cls.sqMode2, cls.sqMode3_2, cls.sqMode4_2)


def train_preprocessing(
    data_x: np.ndarray | pd.DataFrame,
    data_y: np.ndarray | pd.DataFrame | pd.Series,
    feature_names: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """process of train input data: transform to numpy matrix, transpose etc"""

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()

    if isinstance(data_y, pd.DataFrame) or isinstance(data_y, pd.Series):
        data_y = data_y.to_numpy()

    if not isinstance(data_y, (np.ndarray, np.generic)):
        data_y = np.asarray(data_y)
    if len(data_y.shape) != 1:
        if data_y.shape[0] == 1:
            data_y = data_y[0, :]
        elif data_y.shape[1] == 1:
            data_y = data_y[:, 0]
        else:
            raise ValueError('data_y dimension should be 1 or (n, 1) or (1, n)')
    data_len = data_y.shape[0]

    if not isinstance(data_x, (np.ndarray, np.generic)):
        data_x = np.asarray(data_x)

    if len(data_x.shape) != 2:
        raise ValueError('data_x dimension has to be 2. it has to be a 2D numpy array: number of features x '
                         'number of observations')

    if data_x.shape[0] != data_len:
        # try to check if transpose matrix is suitable
        if data_x.shape[1] == data_len:
            # ok, need to transpose data_x
            data_x = data_x.transpose()
        else:
            raise ValueError('number of examples in data_x is not equal to number of examples in data_y')

    if data_x.shape[1] < 2:
        raise ValueError('Error: number of features should be not less than two')

    if data_x.shape[0] < 2:
        raise ValueError('Error: number of samples should be not less than two')

    if feature_names is not None:
        feature_names_len = len(feature_names)
        if feature_names_len > 0 and feature_names_len != data_x.shape[1]:
            raise ValueError('Error: size of feature_names list is not equal to number of features')

    return data_x, data_y


def predict_preprocessing(
    data_x: np.ndarray | pd.DataFrame,
    n_features: int,
) -> tuple[np.ndarray, int]:
    """process of predict input data: transform to numpy matrix, transpose etc"""

    if isinstance(data_x, pd.DataFrame):
        data_x = data_x.to_numpy()

    if not isinstance(data_x, (np.ndarray, np.generic)):
        data_x = np.asarray(data_x)

    if len(data_x.shape) != 2:
        raise ValueError('data_x dimension has to be 2. it has to be a 2D numpy array: number of features x '
                         'number of observations')

    if data_x.shape[1] != n_features:
        # try to check if transpose matrix is suitable
        if data_x.shape[0] == n_features:
            # ok, need to transpose data_x
            data_x = data_x.transpose()
        else:
            raise ValueError('number of features in data_x is not equal to number of features in trained model')

    data_len = data_x.shape[0]
    return data_x, data_len


def set_split_types(seq_type: SequenceTypeSet, data_len: int) -> np.ndarray:
    """
    Set seq_types array that will be used to divide data set to train and validate ones
    """
    # DataSetType is IntEnum, so numeric comparison works against int8 storage.
    seq_types = np.empty((data_len,), dtype=np.int8)

    if seq_type == SequenceTypeSet.sqRandom:
        r = np.random.uniform(-1, 1, data_len)
        seq_types[:] = np.where(r > 0, DataSetType.dsTrain, DataSetType.dsValidate)
        return seq_types

    elif seq_type == SequenceTypeSet.sqMode1:
        n = 2
    elif seq_type == SequenceTypeSet.sqMode3_1:
        n = 3
    elif seq_type == SequenceTypeSet.sqMode4_1:
        n = 4
    elif seq_type == SequenceTypeSet.sqMode2:
        n = 2
    elif seq_type == SequenceTypeSet.sqMode3_2:
        n = 3
    elif seq_type == SequenceTypeSet.sqMode4_2:
        n = 4
    else:
        raise ValueError('Unknown type of data division into train and validate sequences')

    if SequenceTypeSet.is_mode1_type(seq_type):
        for i in range(data_len, 0, -1):
            if (data_len-i) % n == 0:
                seq_types[i-1] = DataSetType.dsValidate
            else:
                seq_types[i-1] = DataSetType.dsTrain

    if SequenceTypeSet.is_mode2_type(seq_type):
        for i in range(data_len, 0, -1):
            if (data_len-i) % n == 0:
                seq_types[i-1] = DataSetType.dsTrain
            else:
                seq_types[i-1] = DataSetType.dsValidate
    return seq_types


def split_dataset(
    data_x: np.ndarray,
    data_y: np.ndarray,
    seq_type: SequenceTypeSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train and validate data sets from input data set and target
    """
    data_len = data_x.shape[0]

    seq_types = set_split_types(seq_type, data_len)

    idx_train = np.extract(DataSetType.dsTrain == seq_types, np.arange(data_len))
    input_train_x = data_x[idx_train, :]
    train_y = data_y[idx_train]

    idx_validate = np.extract(DataSetType.dsValidate == seq_types, np.arange(data_len))
    input_validate_x = data_x[idx_validate, :]
    validate_y = data_y[idx_validate]

    return input_train_x, train_y, input_validate_x, validate_y