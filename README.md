# Dependency Parsing of Music Sequences

This repository accompanies the publication: Francesco Foscarin, Daniel Harasim, and Gerhard Widmer, 
"Predicting Music Hierarchies with a Graph-Based Neural Decoder", published in Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR), Milan, 2023.

We present a data-driven framework to parse musical sequences into dependency trees, which are hierarchical structures used in music cognition research and music analysis. The parsing involves two steps:
1. the input sequence is passed through a transformer encoder to enrich it with contextual information;
2. a classifier filters the graph of all possible dependency arcs to produce the dependency tree.

This system does not rely on any particular symbolic grammar, therefore it can consider multiple musical features simultaneously, make use of sequential context information, and produce partial results for noisy inputs. 


## Datasets
The code in this repo considers two datasets of musical trees: time-span trees of monophonic note sequences and harmonic trees of jazz chord sequences.
All data for training and evaluation were obtained from the [Jazz Harmony Dataset](https://github.com/DCMLab/JazzHarmonyTreebank) and [GTTM database](https://gttm.jp/gttm/database/) and we invite you to cite the corresponding papers if you use these data.

## Metrics
The detailed metrics computed from the results of our experiments are available in the [results](./results) folder. All reported metrics were obtained with leave-one-out cross-validation, i.e., for a given piece name, it is used to compute the test metrics, after the system is trained with all the other pieces. The metrics are separated into two files for the
- [GTTM dataset](./results/GTTM%20result%20table.csv): time-span trees over monophonic melodies;
- [JHT dataset](./results/JHT%20result%20table.csv): harmonic analyses over jazz chord sequences.

The entire metric collection and the plots reported in the paper are contained in this [sheet document](./results/All%20result%20table.xlsx). 

## Graphical Rendition of Dependency Trees
Graphical rendering of the trees, such as this one for the JHT dataset (piece Equinox),
<img src="./results/rendered_JHT/dependency_trees/ground_truth/Equinox.svg"> 
or this one for the GTTM dataset (piece Waves of the Danube)
<img src="./results/rendered_GTTM/dependency_trees/predicted_postprocessing/57_Waves of the Danube.txt.svg"> 
are available in the results folder. There are two markdown files which collect and organize all rendered images.









