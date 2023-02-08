from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import numpy as np
import partitura as pt

class TSDataset(Dataset):
    """Dataset for the TS trees.
    """
    def __init__(self, data_folder):
        """
        Args:
            data_folder (string): Path to the folder containing the data.
        """
        data_df = get_data_df(data_folder)
        self.score_files = data_df["score"]
        self.ts_files = data_df["ts"]
        self.note_features = [get_score_features(score) for score in self.score_files]
        self.tss = [ts_xml_to_dependency_tree(ts) for ts in self.ts_files]
        
    def __len__(self):
        return len(self.score_files)

    def __getitem__(self, idx):
        return self.note_features[idx], self.tss[idx] 

def get_score_path(piece_folder):
    """Retrieve the path to the score file from a piece folder.
    """
    # pieces can either be named MSC or start with a number + the title
    msc_file = [file for file in piece_folder.iterdir() if file.name.startswith("MSC")]
    numbered_file = [file for file in piece_folder.iterdir() if file.name.startswith(piece_folder.name[:2])]
    assert(len(msc_file)+len(numbered_file) == 1)
    score_file = msc_file[0] if len(msc_file) == 1 else numbered_file[0]
    return str(score_file)

def get_ts_path(piece_folder):
    ts_file = [file for file in piece_folder.iterdir() if file.name.startswith("TS")]
    assert(len(ts_file) == 1)
    return str(ts_file[0])

def get_data_df(data_folder):
    list_of_tuples = [(get_score_path(folder), get_ts_path(folder)) for folder in Path(data_folder).iterdir()]
    return pd.DataFrame(list_of_tuples, columns = ['score', 'ts'])

def ts_xml_to_dependency_tree(xml_file):
    """Converts a ts tree xml file to a dependency tree

    Args:
        xml_file (string): the path to the xml file

    Returns:
        list: a list of dependencies, each dependency is a tuple of the form (source, destination)
    """
    tree = ET.parse(str(xml_file))
    root = tree.getroot()
    return _iterative_parse(root)[0]

def _iterative_parse(xml_elem):
    """Iterative function to parse the ts xml file"""
    primary_children = xml_elem.find("ts").find("primary")
    secondary_children = xml_elem.find("ts").find("secondary")
    if primary_children is None: # recursion ending condition
        assert secondary_children is None
        return [], xml_elem.find("ts").find("head").find("chord").find("note").attrib["id"]
    else: # recursive call
        assert secondary_children is not None
        out_list = [] # dependency list
        iterative_result_primary = _iterative_parse(primary_children)
        iterative_result_secondary = _iterative_parse(secondary_children)
        # merge the dependencies lists computed deeper
        out_list.extend(iterative_result_primary[0])
        out_list.extend(iterative_result_secondary[0])
        # append the dependency for the current node
        out_list.append((iterative_result_primary[1], iterative_result_secondary[1]))
        # return the dependency list, and the id of the current node, i.e., the primary
        return out_list, iterative_result_primary[1]

def get_score_features(score_file):
    """Extracts the score features from a score file.

    Args:
        score_file (string): the path to the score file

    Returns:
        list: a list of notes, each note is a tuple with features
    """
    score = pt.load_musicxml(score_file)
    # get the note array
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
    time_signatures = np.char.array(na["ts_beats"].astype(str)) + np.char.array(["/"]*na.shape[0])+ np.char.array(na["ts_beat_type"].astype(str))
    # get real time signature and measure duration, discarding pickup and ending measure ts
    real_ts = np.unique(time_signatures)[-1]
    real_measure_duration = na[time_signatures == real_ts][0]["tot_measure_div"]
    # find pickup notes
    pickup_note_indices = np.where((na["tot_measure_div"]!= real_measure_duration) * (na["onset_div"] < real_measure_duration ))[0]
    # set the pickup note to the correct metrical position
    na["rel_onset_div"][pickup_note_indices] = na["onset_div"][pickup_note_indices] + real_measure_duration - na["tot_measure_div"][pickup_note_indices]
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
    metrical =  na["is_downbeat"] #TODO: add more metrical info
    duration = na["duration_div"]/na["tot_measure_div"]
    # TODO: consider rests as lag from previous note
    return np.vstack((pitch, metrical, duration)).T

