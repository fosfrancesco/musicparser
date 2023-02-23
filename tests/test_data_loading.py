import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from musicparser.data_loading import get_metrical_strength
from musicparser.data_loading import get_note_features_and_dep_arcs


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

def test_metrical_strength_piece40():
    score_file = str(Path(r"data/40/score.xml"))
    ts_file = str(Path(r"data/40/TS.xml"))
    n_feat, dep_arcs, gttm_style_dep = get_note_features_and_dep_arcs(score_file, ts_file)
    metrical = n_feat[:,3]
    m1_expected = [5,0,0,4,3,4,2]
    m2_expected = [5,2,4]
    m3_expected = [5,0,0,4,3,4,2]
    m4_expected = [5,4]
    m5_expected = [5,0,0,4,3,4,2]
    m6_expected = [5,2,4]
    m7_expected = [5,0,0,4,3,4,2]
    m8_expected = [5,4]
    assert np.array_equal(metrical[:7], m1_expected)
    assert np.array_equal(metrical[7:10], m2_expected)
    assert np.array_equal(metrical[10:17], m3_expected)
    assert np.array_equal(metrical[17:19], m4_expected)
    assert np.array_equal(metrical[19:26], m5_expected)
    assert np.array_equal(metrical[26:29], m6_expected)
    assert np.array_equal(metrical[29:36], m7_expected)
    assert np.array_equal(metrical[36:38], m8_expected)


def test_ts_tree_piece40():
    score_file = str(Path(r"data/40/score.xml"))
    ts_file = str(Path(r"data/40/TS.xml"))
    n_feat, dep_arcs, gttm_style_dep = get_note_features_and_dep_arcs(score_file, ts_file)
    expected_dep_arcs = [(0,1),(1,2),(0,3),(0,9),(9,7),(9,8),(7,6),(6,5),(0,17),(17,10),(10,13),(17,16),(16,15),(10,11),(11,12),(0,36),(36,19),(36,29),(36,35),(35,34),(29,32),(29,30),(30,31),(19,28),(19,22),(19,20),(20,21),(28,26),(28,27),(26,25),(25,24)]
    for i,e in enumerate(expected_dep_arcs):
        assert(e in dep_arcs)
    for i,e in enumerate(dep_arcs):
        assert(e in expected_dep_arcs)