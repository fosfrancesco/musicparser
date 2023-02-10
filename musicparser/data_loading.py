from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import numpy as np
import partitura as pt


class TSDataset(Dataset):
    """Dataset for the TS trees."""

    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the folder containing the data.
        """
        self.data_df = get_data_df(data_folder)
        self.note_features = []
        self.dep_arcs = []
        for title, score_file, ts_xml_file in self.data_df.values:
            print("Processing", title)
            n_feat, d_arc = get_note_features_and_dep_arcs(score_file, ts_xml_file)
            self.note_features.append(n_feat)
            self.dep_arcs.append(d_arc)

    def __len__(self):
        return len(self.score_files)

    def __getitem__(self, idx):
        return self.note_features[idx], self.dep_arcs[idx]


def get_score_path(folder):
    score_files = [file for file in folder.iterdir() if file.name.startswith("score")]
    assert(len(score_files) == 1)
    return str(score_files[0])

def get_ts_path(folder):
    ts_file = [file for file in folder.iterdir() if file.name.startswith("TS")]
    assert(len(ts_file) == 1)
    return str(ts_file[0])

def get_title(folder):
    title_file = [file for file in folder.iterdir() if file.suffix == ".txt"]
    assert(len(title_file) == 1)
    return title_file[0].name

def get_data_df(data_folder):
    list_of_tuples = [
        (get_title(folder), get_score_path(folder), get_ts_path(folder))
        for folder in Path(data_folder).iterdir()
    ]
    return pd.DataFrame(list_of_tuples, columns=["title", "score", "ts"])


def ts_xml_to_dependency_tree(xml_file):
    """Converts a ts tree xml file to a dependency tree

    Args:
        xml_file (string): the path to the xml file

    Returns:
        list: a list of dependencies, each dependency is a tuple of the form (source, destination) with gttm-style ids
    """
    tree = ET.parse(str(xml_file))
    root = tree.getroot()
    return _iterative_parse(root)[0]


def gttm_style_to_id_dependency_ts(gttm_ts_dependency, measure_mapping, nra_untied, na):
    """Converts a dependency tree from gttm-style ids to ids in the noteaarray.

    Args:
        ts (list): a list of dependencies, each dependency is a tuple of the form (source, destination) with gttm-style ids

    Returns:
        list: a list of dependencies, each dependency is a tuple of the form (source, destination) with ids in the notearray
    """
    return [
        (
            note_id_to_note_array_index(
                gttm_id_to_pt_id(dep[0], measure_mapping, nra_untied),na
            ),
            note_id_to_note_array_index(
                gttm_id_to_pt_id(dep[1], measure_mapping, nra_untied),na
            ),
        )
        for dep in gttm_ts_dependency
    ]


def _iterative_parse(xml_elem):
    """Iterative function to parse the ts xml file"""
    primary_children = xml_elem.find("ts").find("primary")
    secondary_children = xml_elem.find("ts").find("secondary")
    if primary_children is None:  # recursion ending condition
        assert secondary_children is None
        return (
            [],
            xml_elem.find("ts").find("head").find("chord").find("note").attrib["id"],
        )
    else:  # recursive call
        assert secondary_children is not None
        out_list = []  # dependency list
        iterative_result_primary = _iterative_parse(primary_children)
        iterative_result_secondary = _iterative_parse(secondary_children)
        # merge the dependencies lists computed deeper
        out_list.extend(iterative_result_primary[0])
        out_list.extend(iterative_result_secondary[0])
        # append the dependency for the current node
        out_list.append((iterative_result_primary[1], iterative_result_secondary[1]))
        # return the dependency list, and the id of the current node, i.e., the primary
        return out_list, iterative_result_primary[1]


def get_note_features(score):
    """Extracts the score features from a partitura score.

    Args:
        score (partitura.score.Score): A monophonic musical score with one part.

    Returns:
        list: a list of notes, each note is a tuple with features
    """
    na = pt.utils.music.ensure_notearray(
        score,
        include_metrical_position=True,
        include_time_signature=True,
    )
    # correct the metrical information
    clean_na, real_ts, real_measure_duration = correct_metrical_information(na)
    # get the note features
    note_feature = get_features_from_na(clean_na)
    return note_feature


def correct_metrical_information(na):
    """Corrects the metrical information in the note array. This removes wrong ts and metrical info for pickup notes and ending measures."""
    time_signatures = (
        np.char.array(na["ts_beats"].astype(str))
        + np.char.array(["/"] * na.shape[0])
        + np.char.array(na["ts_beat_type"].astype(str))
    )
    # get real time signature and measure duration, discarding pickup and ending measure ts
    real_ts = np.unique(time_signatures)[-1]
    real_measure_duration = na[time_signatures == real_ts][0]["tot_measure_div"]
    # find pickup notes
    pickup_note_indices = np.where(
        (na["tot_measure_div"] != real_measure_duration)
        * (na["onset_div"] < real_measure_duration)
    )[0]
    # set the pickup note to the correct metrical position
    na["rel_onset_div"][pickup_note_indices] = (
        na["onset_div"][pickup_note_indices]
        + real_measure_duration
        - na["tot_measure_div"][pickup_note_indices]
    )
    # set the measure duration to correct value
    na["tot_measure_div"] = real_measure_duration
    # set the correct time signature
    na["ts_beats"] = real_ts.split("/")[0]
    na["ts_mus_beats"] = real_ts.split("/")[0]
    na["ts_beat_type"] = real_ts.split("/")[1]
    return na, real_ts, real_measure_duration


def get_features_from_na(na):
    """Extracts the features from the note array. It must contains the metrical information."""
    pitch = na["pitch"]
    metrical = na["is_downbeat"]  # TODO: add more metrical info
    duration = na["duration_div"] / na["tot_measure_div"]
    # TODO: consider rests as lag from previous note
    return np.vstack((pitch, metrical, duration)).T


def gttm_id_to_pt_id(gttm_id, measure_mapping, nra_untied):
    """Translate the gttm-style ids, e.g., 'P1-3-1', to indices in partitura note array.
    We need both the untied notes and the rest, because the gttm notation takes both into account

    Args:
        gttm_id (string): gttm-style id, e.g., 'P1-3-1'
        measure_mapping (list[int]): a list with the measure number for each (untied) note and rest in the partitura note array
        nra_untied (np.array): a numpy structured array of (untied) notes and rest. It must contains the "id" field

    Returns:
        int: id of the note in the partitura nra_untied
    """
    measure_number = int(gttm_id.split("-")[1])
    note_number = int(gttm_id.split("-")[2])
    # notes_in_measure = np.where(measure_mapping == measure_number)[0]
    nra_index = np.where(measure_mapping == measure_number)[0][int(note_number) - 1]
    return nra_untied[nra_index]["id"]
    

def note_id_to_note_array_index(id, na):
    """Translate the note id to the index in the note array. This work because annotations in gttm database are only on tied notes.

    Args:
        id (int): id of the note
        na (np.array): a numpy structured array of (tied) notes and rest. It must contains the "id" field

    Returns:
        int: index of the note in the partitura note array
    """
    potential_indices = np.where(na["id"] == id)[0]
    if len(potential_indices) == 1:
        return np.where(na["id"] == id)[0][0]
    else:
        print("Problem with note id: ", id)
        return 0


def get_dependency_arcs(ts_xml_file, score):
    na = pt.utils.music.ensure_notearray(
        score,  
    )
    gttm_ts = ts_xml_to_dependency_tree(ts_xml_file)
    # compute untied (untied) notes and rests array. We need it because the gttm notation takes both into account in its counting
    na_untied = pt.utils.music.note_array_from_note_list(score.parts[0].notes)
    ra_untied = pt.utils.music.rest_array_from_rest_list(score.parts[0].rests)
    ra__untied_fields = list(ra_untied.dtype.names)
    nra_untied = np.hstack([na_untied[ra__untied_fields],ra_untied])
    nra_untied.sort(order="onset_div")
    # # keep only one rest row if there are consecutive rests, to comply with gttm notation
    # rest_mask = nra_untied["id"].astype('U1') == "r"
    # consecutive_mask = ~np.insert(np.diff(rest_mask), 0, True)
    # combined_mask = rest_mask * consecutive_mask
    # nra_untied = nra_untied[~combined_mask]
    # get the gttm-style dependency tree
    m_map = score.parts[0].measure_number_map(nra_untied["onset_div"])
    return gttm_style_to_id_dependency_ts(gttm_ts, m_map, nra_untied, na)

def get_note_features_and_dep_arcs(score_file, ts_xml_file):
    score = pt.load_musicxml(score_file, force_note_ids=True)
    # get the note array
    note_features = get_note_features(score)
    dep_arcs = get_dependency_arcs(ts_xml_file, score)
    return note_features, dep_arcs
