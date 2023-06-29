# Dependency Parsing of Music Sequences

<img src="./results/rendered_JHT/dependency_trees/ground_truth/Equinox.svg"> 

This repository accompanies the publication: Francesco Foscarin, Daniel Harasim, and Gerhard Widmer, 
"Predicting Music Hierarchies with a Graph-Based Neural Decoder", published in Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR), Milan, 2023.

We present a data-driven framework to parse musical sequences into dependency trees, which are hierarchical structures used in music cognition research and music analysis. The parsing involves two steps:
1. the input sequence is passed through a transformer encoder to enrich it with contextual information;
2. a classifier filters the graph of all possible dependency arcs to produce the dependency tree.

This system does not rely on any particular symbolic grammar, therefore it can consider multiple musical features simultaneously, make use of sequential context information, and produce partial results for noisy inputs. 

## Datasets
The code in this repo considers two datasets of musical trees: time-span trees of monophonic note sequences and harmonic trees of jazz chord sequences.
All data for training and evaluation were obtained from the [Jazz Harmony Dataset](https://github.com/DCMLab/JazzHarmonyTreebank) and [GTTM database](https://gttm.jp/gttm/database/) and we invite you to cite the corresponding papers if you use these data.

## Running the system
The easiest way of running the system is to call a launch script from the terminal. The required packages are specified in the [requirements.txt](requirements.txt) file. Two different scripts are available, one for the JHT dataset and one for the GTTM dataset. For example, to start an experiment with the JHT datasets you can write:
```
python launch_scripts/launch_script_jtb.py 
```
A big number of parameters are available, which you can explore by calling ```python launch_scripts/launch_script_jtb.py --help```. If no extra parameters are specified, the experiment will be run with the network hyperparameters specified in the paper, by using 90% of the dataset as training and 10% as testing (no validation set).
You can log the metrics on [Weight and Bias](https://wandb.ai/site) by using the ```--wandb_log``` parameter, but this will require an account.

## Reproduce paper results
The results in the paper are computed by using leave-one-out cross-validation, i.e., for a dataset of N pieces, the system is run N times, and each time it uses N-1 pieces for training and 1 piece for testing. The cross-validation is handled by a Weight and Bias sweep and will require an account. There are two configuration files, [one](launch_scripts/sweep_loo_jtb.py) for the JHT and [one](launch_scripts/sweep_train_ts.py) for the GTTM experiments. To see how to initialize a sweep with weight and bias, see [this](https://docs.wandb.ai/guides/sweeps/initialize-sweeps) guide. For example for the JHT experiment, the CLI command is:
```
wandb sweep --project name_of_wandb_proj sweep/sweep_jtb_loo.yml
```
This can take some time since the system will need to be trained 150 times for the JHT dataset, or 296 for the GTTM dataset. A faster way to explore the paper results is to use the precomputed metrics and outputs.

## Precomputed Results
### Metrics
The detailed metrics computed from the results of our experiments are available in the [results](./results) folder. All reported metrics were obtained with leave-one-out cross-validation, i.e., for a given piece name, it is used to compute the test metrics, after the system is trained with all the other pieces. The metrics are separated into two files for the
- [GTTM dataset](./results/GTTM%20result%20table.csv): time-span trees over monophonic melodies;
- [JHT dataset](./results/JHT%20result%20table.csv): harmonic analyses over jazz chord sequences.

The entire metric collection and the plots reported in the paper are contained in this [sheet document](./results/All%20result%20table.xlsx). 

### Predicted trees
All predicted trees (with leave-one-out cross-validation) are saved in a JSON format in [this](results/predicted_JHT.json) and [this](results/predicted_GTTM.json) files.
For each JSON file, the first-level keys are the name of the pieces, and the values are other dictionaries with the following keys:
- "head_predicted": the predicted head sequence from the neural decoder;
- "head_predicted_postp": the predicted head sequence after the postprocessing algorithm;
- "head_truth": the ground truth head sequence;
- "ctree_predicted": a parenthetical representation of the predicted tree after conversion into a constituent tree;
- "ctree_truth": a parenthetical representation of the ground truth constituent tree;
- "notes" or "chords" (depending on the dataset): the element of the input sequence.


### Graphical Rendition of Dependency Trees
Graphical rendering of the trees, such as this one for the JHT dataset (piece Equinox),
<img src="./results/rendered_JHT/dependency_trees/ground_truth/Equinox.svg"> 
or this one for the GTTM dataset (piece Waves of the Danube)
<img src="./results/rendered_GTTM/dependency_trees/predicted_postprocessing/57_Waves of the Danube.txt.svg"> 
are available in the results folder. There are two markdown files which collect and organize all rendered images and enable faster visual comparison of the ground truth, predicted, and predicted-with-postprocessing trees.

A small analysis of the graphical rendering and the code to produce it from the json files is available in [this](data_analysis.ipynb) notebook.

## Citing
If you use this system in any research, please cite the relevant paper:

```
@inproceedings{foscarin23,
  title={Predicting Music Hierarchies with a Graph-Based Neural Decoder},
  author={Foscarin, Francesco and Harasim, Daniel and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference {(ISMIR)}},
  year={2023},
}
```

## License
This work is made available under a [Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

