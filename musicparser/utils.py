import spacy
from spacy import displacy

def display_JHT_svg(piece_data,type = "head_predicted_postp", jupyter=True):
    """ 
    Display a piece of music from the JHT dataset as a dependency tree.
        piece_data: a dictionary containing the piece data, as returned by the JHTDataModule
        type: the type of dependency tree to display. Can be one of "head_predicted_postp", "head_predicted", "head_truth"
        jupyter: whether to display the tree in a jupyter notebook or not
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