import pytest
import numpy as np
import pandas as pd

from musicparser.data_loading import get_metrical_strength


def test_metrical_strength_68():
    time_signature = (6,8,8)
    total_div = 12
    rel_onsets = np.array([0,6,8,9,10])
    m_strength = get_metrical_strength(rel_onsets, time_signature, total_div)
    expected = np.array([5,4,3,2,3])
    assert np.array_equal(m_strength, expected)

def test_metrical_strength_24():
    time_signature = (2,4,2)
    total_div = 24
    rel_onsets = np.array([0,12,16,20])
    m_strength = get_metrical_strength(rel_onsets, time_signature, total_div)
    expected = np.array([5,4,0,0])
    assert np.array_equal(m_strength, expected)

