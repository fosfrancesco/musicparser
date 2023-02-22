from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import numpy as np
import partitura as pt
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from collections import defaultdict

MINIMUM_OCTAVE = 4
MAXIMUM_OCTAVE = 9
DURATIONS = [0.0312, 0.0357, 0.0417, 0.0500, 0.0556, 0.0625, 0.0833, 0.1111, 0.1250,
        0.1667, 0.1750, 0.1875, 0.2083, 0.2222, 0.2500, 0.2917, 0.3125, 0.3333,
        0.3750, 0.4062, 0.4167, 0.4375, 0.4444, 0.5000, 0.5556, 0.5625, 0.5833,
        0.6250, 0.6667, 0.6875, 0.7500, 0.8333, 0.8750, 0.9167, 0.9375, 1.0000,
        1.1250, 1.1667, 1.2500, 1.3333, 1.5000, 1.6667, 1.7500, 2.0000, 4.0000]
METRICAL_DIVISIONS = {
    12 : [4,3,2,2],
    9 : [3,3,2,2],
    8 : [2,2,2,2],
    6 : [2,3,2,2],
    4 : [2,2,2,2],
    3 : [3,2,2,2],
    2 : [2,2,2,2],
}

class TSDataModule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, will_use_embeddings=False, data_augmentation=True):
        super(TSDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # instatiate 3 different datasets for augmentation
        self.dataset = TSDataset(Path("data"), will_use_embeddings=will_use_embeddings, data_augmentation=data_augmentation)
        self.positive_weight = self.dataset.get_positive_weight()
        # self.features = self.dataset.features
        # self.test_collection = test_collection

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        idxs = range(len(self.dataset))
        trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, random_state=0)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=0)
        # create the datasets
        self.dataset_train = torch.utils.data.Subset(self.dataset,train_idx)
        self.dataset_val = torch.utils.data.Subset(self.dataset,val_idx)
        self.dataset_test = torch.utils.data.Subset(self.dataset,test_idx)
        # set the data augmentation idx in the dataset
        to_aug_dict = defaultdict(bool)
        for idx in train_idx:
            to_aug_dict[idx] = True
        self.dataset.to_augment_dict = to_aug_dict
        # self.dataset_predict = self.dataset[test_idx[:5]]
        print(f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}")
        # compute the positive weight to be used to balance the loss
            
    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) :
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.dataset_predict, batch_size=self.batch_size, num_workers=self.num_workers)


class TSDataset(Dataset):
    """Dataset for the TS trees."""

    def __init__(self, data_folder, data_augmentation=False, will_use_embeddings=False):
        """
        Args:
            data_folder (string): Path to the folder containing the data.
        """
        self.data_augmentation = data_augmentation
        self.will_use_embeddings = will_use_embeddings
        self.data_df = get_data_df(data_folder)
        self.note_features = []
        self.dep_arcs = []
        self.truth_masks = []
        self.pot_arcs = []
        self.score_files = []
        self.to_augment_dict = {}
        print("Loading data...")
        for title, score_file, ts_xml_file in self.data_df.values:
            try:
                n_feat, d_arc, gttm_style_feat = get_note_features_and_dep_arcs(score_file, ts_xml_file)
                n_feat= torch.tensor(n_feat)
                d_arc = torch.tensor(d_arc)
                # compute potential arcs, i.e., all arcs minus self loops and rests connections
                indices = torch.arange(len(n_feat))
                cart_prod = torch.cartesian_prod(indices,indices) # all possible pairs
                pot_arcs = cart_prod[cart_prod[:,0]!=cart_prod[:,1]] # remove self loops
                starting_rest_mask = n_feat[pot_arcs[:,0]][:,1]
                ending_rest_mask = n_feat[pot_arcs[:,1]][:,1]
                pot_arcs = pot_arcs[~np.logical_or(starting_rest_mask, ending_rest_mask)]
                # compute the ground truth mask over the pot arcs
                truth_mask = get_edges_mask(d_arc, pot_arcs)
                # add everything to the dataset
                self.score_files.append(score_file)
                self.note_features.append(n_feat)
                self.dep_arcs.append(d_arc)
                self.pot_arcs.append(pot_arcs)
                self.truth_masks.append(truth_mask)
            except Exception as e:
                print(f"!!!!! Error with {title}", e)

    def __len__(self):
        return len(self.truth_masks)

    def __getitem__(self, idx):
        # return [(self.note_features[i], self.dep_arcs[i], self.truth_masks[i], self.pot_arcs[i]) for i in idx]
        if not self.data_augmentation:
            return data_preparation(self.note_features[idx], self.will_use_embeddings), self.truth_masks[idx], self.pot_arcs[idx]
        else:
            if not self.to_augment_dict[idx]: # don't augment, for example for test and validation
                return data_preparation(self.note_features[idx], self.will_use_embeddings), self.truth_masks[idx], self.pot_arcs[idx]
            else: # augment
                return data_preparation(data_augmentation(self.note_features[idx]), self.will_use_embeddings), self.truth_masks[idx], self.pot_arcs[idx]

           
       

    def get_positive_weight(self):
        return sum([len(truth_mask)/torch.sum(truth_mask) for truth_mask in self.truth_masks])/len(self.truth_masks)


def get_edges_mask(subset_edges, total_edges, transpose=False, check_strict_subset=True):
    """Get a mask of edges to use for training.
    Parameters
    ----------
    subset_edges : np.array
        A subset of total_edges.
    total_edges : np.array
        Total edges.
    transpose : bool, optional.
        Whether to transpose the subset_edges, by default True.
        This is necessary if the input arrays are (2, n) instead of (n, 2)
    check_strict_subset : bool, optional
        Whether to check that the subset_edges are a strict subset of total_edges.
    Returns
    -------
    edges_mask : np.array
        Mask that identifies subset edges from total_edges.
    dropped_edges : np.array
        Truth edges that are not in potential edges.
        This is only returned if check_strict_subset is True.
    """
    # convert to numpy, custom types are not supported by torch
    total_edges = total_edges.numpy() if not isinstance(total_edges, np.ndarray) else total_edges
    subset_edges = subset_edges.numpy() if not isinstance(subset_edges, np.ndarray) else subset_edges
    # transpose if r; contiguous is required for the type conversion step later
    if transpose:
        total_edges = np.ascontiguousarray(total_edges.T)
        subset_edges = np.ascontiguousarray(subset_edges.T)
    # convert (n, 2) array to an n array of bytes, in order to use isin, that only works with 1d arrays
    # view_total = total_edges.view(np.dtype((np.void, total_edges.dtype.itemsize * total_edges.shape[-1])))
    # view_subset = subset_edges.view(np.dtype((np.void, subset_edges.dtype.itemsize * subset_edges.shape[-1])))
    view_total = np.char.array(total_edges.astype(str))
    view_subset = np.char.array(subset_edges.astype(str))
    view_total = view_total[:, 0] + "-" + view_total[:, 1]
    view_subset = view_subset[:, 0] + "-" + view_subset[:, 1]
    if check_strict_subset:
        dropped_edges = subset_edges[(~np.isin(view_subset, view_total))]
        assert(len(dropped_edges) == 0)
    return torch.from_numpy(np.isin(view_total, view_subset)).squeeze()

        
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
    xml_root = tree.getroot()
    dep_arcs, root = _iterative_parse(xml_root)
    # # add the root
    # dep_arcs.append(("ROOT", -1))
    return  dep_arcs, root


def gttm_style_to_id_dependency_ts(gttm_ts_dependency, measure_mapping, nra_untied, nra_tied):
    """Converts a dependency tree from gttm-style ids to ids in the noteaarray.
    We need both the an array of untied notes and rests (what gttm notation reference to) and the notearray (what we will use) to convert the ids.

    Args:
        gttm_ts_dependency (list): a list of dependencies, each dependency is a tuple of the form (source, destination) with gttm-style ids
        measure_mapping (list): a list of measures for each note in the nra_untied
        nra_untied (np.array): a structured array of untied notes and rests
        nra (np.array): a structured array of (tied) notes and rests
    Returns:
        list: a list of dependencies, each dependency is a tuple of the form (source, destination) with ids in the notearray
    """
    return [
        (
            note_id_to_note_array_index(
                gttm_id_to_pt_id(dep[0], measure_mapping, nra_untied),nra_tied
            ),
            note_id_to_note_array_index(
                gttm_id_to_pt_id(dep[1], measure_mapping, nra_untied),nra_tied
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


def get_note_features(score, nra):
    """Extracts the score features from a partitura score.

    Args:
        score (partitura.score.Score): A monophonic musical score with one part.

    Returns:
        list: a list of notes, each note is a tuple with features
    """
    # add the time signature
    time_signatures = score.parts[0].time_signature_map(nra["onset_div"])
    # add metrical information
    metrical_info = score.parts[0].metrical_position_map(nra["onset_div"])
    # correct the metrical information
    nra, time_signatures, rel_onset_div, total_measure_div = correct_metrical_information(nra,time_signatures,metrical_info)
    # get the note features
    note_feature = get_features_from_nra(nra,time_signatures,rel_onset_div,total_measure_div)
    return note_feature


def correct_metrical_information(nra, time_signatures,metrical_info):
    """Corrects the metrical information in the note array. This removes wrong ts and metrical info for pickup notes and ending measures."""
    rel_onset_div = metrical_info[:,0]
    total_measure_div = metrical_info[:,1]
    time_signature_strings = (
        np.char.array(time_signatures[:,0].astype(str))
        + np.char.array(["/"] * nra.shape[0])
        + time_signatures[:,1].astype(str)
    )
    # get real time signature and measure duration, discarding pickup and ending measure ts
    real_ts = np.unique(time_signature_strings)[-1] #pick the one with biggest numerator
    real_measure_duration = total_measure_div[time_signature_strings == real_ts][0]
    # find pickup notes
    pickup_note_indices = np.where(
        (total_measure_div != real_measure_duration)
        * (nra["onset_div"] < real_measure_duration)
    )[0]
    # set the pickup note to the correct metrical position
    rel_onset_div[pickup_note_indices] = (
        nra["onset_div"][pickup_note_indices]
        + real_measure_duration
        - total_measure_div[pickup_note_indices]
    )
    # set the measure duration to correct value
    total_measure_div = real_measure_duration
    # set the correct time signature
    time_signatures[:,0] = real_ts.split("/")[0]
    time_signatures[:,2] = real_ts.split("/")[0]
    time_signatures[:,1] = real_ts.split("/")[1]
    return nra, time_signatures, rel_onset_div, total_measure_div

def get_pc_one_hot(pitch):
    return F.one_hot(torch.remainder(pitch, 12).to(torch.int64), num_classes=12)

def get_octave_one_hot(pitch):
    return F.one_hot(torch.floor_divide(pitch, 12).to(torch.int64), num_classes=MAXIMUM_OCTAVE)

def get_duration_one_hot(duration):
    return F.one_hot(duration.to(torch.int64), num_classes=len(DURATIONS))

def get_metrical_one_hot(metrical):
    return F.one_hot(metrical.to(torch.int64), num_classes=6)

def get_feats_one_hot(n_feats):
    pitch = n_feats[:,0]
    is_rest = n_feats[:,1]
    duration = n_feats[:,2]
    metrical = n_feats[:,3]
    # compute one hot encoding for pitch and octave
    pc_oh = get_pc_one_hot(pitch)
    octave_oh = get_octave_one_hot(pitch)
    # remove pitch info for rests
    pc_oh[is_rest.to(torch.int64),:] = 0
    octave_oh[is_rest.to(torch.int64),:] = 0
    # truncate octave to MAX_OCTAVE - MIN_OCTAVE values
    octave_oh = octave_oh[:,MINIMUM_OCTAVE:MAXIMUM_OCTAVE]
    # compute one hot encoding for duration
    # duration_oh = get_duration_one_hot(duration)
    duration = torch.tanh(torch.tensor(DURATIONS).to(duration.device)[duration])
    # compute one hot encoding for metrical position
    metrical_oh = get_metrical_one_hot(metrical)
    return torch.hstack((torch.unsqueeze(is_rest,1),pc_oh, octave_oh, torch.unsqueeze(duration,1), metrical_oh))

def data_preparation(n_feats, will_use_embeddings=False):
    n_feats = torch.tensor(n_feats)
    return n_feats

def data_augmentation(n_feats):
    random_transp_int = int(torch.randint(low=-12, high = 13, size=(1,))[0])
    transpose_mask = (n_feats[:,1] == 0) * random_transp_int # this is to not transpose rests
    n_feats[:,0] = n_feats[:,0] + transpose_mask
    return n_feats


def get_features_from_nra(nra, time_signatures, rel_onset_div, total_measure_div):
    """Extracts the features from the (tied) note rest array."""
    duration = nra["duration_div"] / total_measure_div
    duration_indices = [DURATIONS.index(round(d,4)) for d in duration]
    pitch = nra["pitch"]
    is_rest = np.char.startswith(nra["id"],"r")
    # metrical = rel_onset_div == 0
    assert np.all(time_signatures == time_signatures[0])
    metrical = get_metrical_strength(rel_onset_div, time_signatures[0], total_measure_div)
    # octave_oh = get_octave_one_hot(nra)
    # pc_oh = get_pc_one_hot(nra)
    # duration_feature = np.expand_dims(1- np.tanh(duration), 1)
    # metrical = np.expand_dims( rel_onset_div == 0,1)  # TODO: add more metrical info
    # out = np.hstack((pc_oh, octave_oh,duration_feature, metrical))
    return np.vstack((pitch, is_rest, duration_indices, metrical)).T


def get_metrical_strength(rel_onsets, time_signature, total_measure_div):
    """Computes the metrical strength of the onsets in a given time signature and total measure duration.
    """
    num_beats = time_signature[0]
    if num_beats not in METRICAL_DIVISIONS.keys():
        raise ValueError(f"The number of beats {num_beats} is not supported.")
    metrical_divisions = np.array([1] + METRICAL_DIVISIONS[num_beats]) # added 1 for the strongest downbeat position
    divisors = total_measure_div / np.cumprod(metrical_divisions)
    # compute the metrical strength
    metrical_strength = np.remainder(np.expand_dims(rel_onsets,1), divisors) == 0
    return np.sum(metrical_strength, axis=1)

        



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
    

def note_id_to_note_array_index(id, nra):
    """Translate the note id to the index in the note array. This work because annotations in gttm database are only on tied notes.

    Args:
        id (int): id of the note
        na (np.array): a numpy structured array of (tied) notes and rest. It must contains the "id" field

    Returns:
        int: index of the note in the partitura note array
    """
    if id[0] == "r":
        raise ValueError("Trying to build an arc from a rest")
    if id[0] == "ROOT":
        return "ROOT"
    potential_indices = np.where(nra["id"] == id)[0]
    if len(potential_indices) == 1:
        return np.where(nra["id"] == id)[0][0]
    else:
        raise Exception("Problem with note id: ", id)


def get_dependency_arcs(ts_xml_file, score, nra_tied):
    gttm_ts, gttm_root = ts_xml_to_dependency_tree(ts_xml_file)
    # compute untied (untied) notes and rests array. We need it because the gttm notation takes both into account in its counting
    na_untied = pt.utils.music.note_array_from_note_list(score.parts[0].notes)
    ra_untied = pt.utils.music.rest_array_from_rest_list(score.parts[0].rests)
    ra__untied_fields = list(ra_untied.dtype.names)
    nra_untied = np.hstack([na_untied[ra__untied_fields],ra_untied])
    nra_untied.sort(order="onset_div")
    # keep only one rest row if there are consecutive rests, to comply with gttm notation
    # rest_mask = nra_untied["id"].astype('U1') == "r"
    # consecutive_mask = ~np.insert(np.diff(rest_mask), 0, True)
    # combined_mask = rest_mask * consecutive_mask
    # nra_untied = nra_untied[~combined_mask]
    # get the gttm-style dependency tree
    m_map = score.parts[0].measure_number_map(nra_untied["onset_div"])
    try:
        return gttm_style_to_id_dependency_ts(gttm_ts, m_map, nra_untied, nra_tied), gttm_ts
    except: # there is a common error in pickup measures where the first rest is not counted
        print("Trying to solve first measure error in: ", ts_xml_file)
        try:
            return  gttm_style_to_id_dependency_ts(gttm_ts, m_map[1:], nra_untied[1:], nra_tied), gttm_ts
        except:
            try:
                return  gttm_style_to_id_dependency_ts(gttm_ts, m_map[2:], nra_untied[2:], nra_tied), gttm_ts
            except:
                raise ValueError("Can't assign ids in: ", ts_xml_file)

def get_note_features_and_dep_arcs(score_file, ts_xml_file):
    score = pt.load_musicxml(score_file, force_note_ids=True)
    nra = get_nra(score)
    # remove grace notes
    nra = nra[nra["duration_div"] != 0]
    # compute features
    note_features = get_note_features(score,nra)
    # compute dependency arcs
    dep_arcs, gttm_style_dep = get_dependency_arcs(ts_xml_file, score, nra)
    return note_features, dep_arcs, gttm_style_dep

def get_nra(score):
    # get tied note array
    na = pt.utils.music.ensure_notearray(
        score
    )[["onset_div","duration_div","pitch","id"]]
    # get rest array
    ra = pt.utils.music.ensure_rest_array(
        score.parts[0]
    )[["onset_div","duration_div","pitch","id"]]
    # merge the two and sort by onset
    nra = np.hstack([na,ra])
    nra.sort(order="onset_div")
    return nra