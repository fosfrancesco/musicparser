from spacy import displacy
import numpy as np

def display_JHT_svg(piece_data,type = "head_predicted_postp", jupyter=True):
    """ 
    Display a piece of music from the JHT dataset as a dependency tree.
        piece_data: a dictionary containing the piece data, loaded from the predicted_JHT.json file
        type: the type of dependency tree to display. Can be one of "head_predicted_postp", "head_predicted", "head_truth"
        jupyter: whether to display the tree in a jupyter notebook or not

    Returns:
        An svg string containing the dependency tree (only if jupyter=False)
    """
    spacy_words = [{"text": chord, "tag": ""} for chord in piece_data["chords"]]
    spacy_arcs = []
    for start_ix, end_ix in enumerate(piece_data[type]):
        if end_ix == 0:
            continue # skip root
        if start_ix < end_ix-1:
            spacy_arcs.append({"start": start_ix, "end": end_ix-1, "label": "", "dir": "right"})
        else:
            spacy_arcs.append({"start": end_ix-1, "end": start_ix, "label": "", "dir": "left"})

    spacy_dict = {"words": spacy_words, "arcs": spacy_arcs}
    return displacy.render(spacy_dict, style='dep', jupyter=jupyter, manual=True, options={"compact": False, "distance": 80, "arrow_stroke":1, "arrow_width":6, "word_spacing":20})

def display_GTTM_svg(piece_data,type = "head_predicted_postp", jupyter=True):
    """ 
    Display a piece of music from the GTTM dataset as a dependency tree.
        piece_data: a dictionary containing the piece data loaded from the predicted_GTTM.json file
        type: the type of dependency tree to display. Can be one of "head_predicted_postp", "head_predicted", "head_truth"
        jupyter: whether to display the tree in a jupyter notebook or not

    Returns:
        An svg string containing the dependency tree (only if jupyter=False)
    """
    head_seq = piece_data[type]
    head_seq = [0]+head_seq # add the root
    ## remove rests from the head_seq, and reduce the end index accordingly
    # to do this we pass through an arc matrix
    is_rest = np.array([False]+[note == "C0" for note in piece_data["notes"]]) # rests are marked as C0
    arc_matrix = np.zeros((len(head_seq), len(head_seq)))
    for start_ix, end_ix in enumerate(head_seq):
        if end_ix != 0:
            arc_matrix[start_ix, end_ix] = 1
    # remove rows and columns correponding to rests (all zeros)
    arc_matrix = arc_matrix[~is_rest,:][:,~is_rest]
    # argmax to get the new head sequence
    new_head_seq = arc_matrix.argmax(axis=1)
    new_head_seq = new_head_seq.tolist()[1:] # remove the root

    spacy_words = [{"text": note, "tag": ""} for note in piece_data["notes"] if note != "C0"]
    print(piece_data["notes"])
    spacy_arcs = []
    for start_ix, end_ix in enumerate(new_head_seq):
        if end_ix == 0:
            continue # skip root
        if start_ix < end_ix-1:
            spacy_arcs.append({"start": start_ix, "end": end_ix-1, "label": "", "dir": "right"})
        else:
            spacy_arcs.append({"start": end_ix-1, "end": start_ix, "label": "", "dir": "left"})

    spacy_dict = {"words": spacy_words, "arcs": spacy_arcs}
    render_options = {"compact": False, "distance": 80, "arrow_stroke":1, "arrow_width":6, "word_spacing":20}
    return displacy.render(spacy_dict, style='dep', jupyter=True, manual=True, options=render_options)


def MIDInumber_to_note_name(number: int) -> str:
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    OCTAVES = list(range(11))
    NOTES_IN_OCTAVE = len(NOTES)

    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES
    assert 0 <= number <= 127
    note = NOTES[number % NOTES_IN_OCTAVE]

    return f"{note}{octave}"