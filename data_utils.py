import numpy as np
import random
import os

"""
    taken from: https://github.com/GokuMohandas/fast-weights/blob/master/fw/data_utils.py
    Creating the data set for fast weights implementation.
    Data will minmic the synthetic dataset created in
    https://arxiv.org/abs/1610.06258 Ba et al.
    Ex.
    c6a7s4??a = 7 (it asking for the value for the key a)
    This is a very interesting dataset because
    it requires the model to retrieve and use temporary memory
    in order to accurately predict the proper value for the key.
"""


def get_letters(num_letters):
    """
    Retrieve three random letters (a-z)
    without replacement.
    """
    return np.random.choice(range(0, 26), num_letters, replace=False)


def get_numbers(num_letters):
    """
    Retrieve three random numbers (0-9)
    with replacement.
    """
    return np.random.choice(range(26, 26+10), num_letters, replace=True)


def create_sequence(num_letters):
    """
    Concatenate keys and values with
    ?? and one of the keys.
    Returns the input and output.
    """
    letters = get_letters(num_letters)
    numbers = get_numbers(num_letters)
    X = np.zeros((9))
    y = np.zeros((1))
    for i in range(0, 5, 2):
        X[i] = letters[i//2]
        X[i+1] = numbers[i//2]

    # append ??
    X[6] = 26+10
    X[7] = 26+10

    # last key and respective value (y)
    index = np.random.choice(range(0, 3), 1, replace=False)
    X[8] = letters[index]
    # y = numbers[index] - 26
    y = numbers[index]

    # one hot encode X and y
    X_one_hot = np.eye(26+10+1)[np.array(X).astype('int')]
    # y_one_hot = np.eye(26+10+1)[y][0]

    # return X_one_hot, y_one_hot
    return X_one_hot, y


def ordinal_to_alpha(sequence):
    """
    Convert from ordinal to alpha-numeric representations.
    Just for funsies :)
    """
    corpus = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, '?'
    ]

    conversion = ""
    for item in sequence:
        conversion += str(corpus[int(item)])
    return conversion


def create_data(num_samples, num_letters):
    """
    Create a num_samples long set of X and y.
    """
    X = np.zeros([num_samples, 9, 26 + 10 + 1], dtype=np.int32)
    # X = np.zeros([num_samples, 9], dtype=np.int32)
    y = np.zeros([num_samples], dtype=np.int32)
    for i in range(num_samples):
        X[i], y[i] = create_sequence(num_letters)
    return X, y
