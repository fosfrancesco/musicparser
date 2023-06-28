# Dependency Parsing of Music Sequences

This repository accompanies the publication: Francesco Foscarin, Daniel Harasim, and Gerhard Widmer, 
"Predicting Music Hierarchies with a Graph-Based Neural Decoder", published in Proceedings of the 24rd ISMIR 2023.

All data for training and evaluation were obtained from the [Jazz Harmony Dataset](https://github.com/DCMLab/JazzHarmonyTreebank) and [GTTM database](https://gttm.jp/gttm/database/) and we invite you to cite the corresponding papers if you use these data.

The complete statistics of our results are available in the the [results](./results) folder. All reported metrics were obtained with leave-one-out cross validation, i.e., all the other pieces were used for training, and the selected piece for testing. They are separated in the two files for the
- [GTTM dataset](./results/GTTM%20result%20table.csv): time-span trees over monophonic melodies;
- [JHT dataset](./results/JHT%20result%20table.csv): harmonic analyses over jazz chord sequences.

Graphical rendering of the trees, such as this one for the JHT dataset (piece Equinox),
<img src="./results/rendered_JHT/dependency_trees/ground_truth/Equinox.svg"> 
or this one for the GTTM dataset (piece Waves of the Danube)
<img src="./results/rendered_GTTM/dependency_trees/predicted_postprocessing/57_Waves of the Danube.txt.svg"> 
are also available in the results folder. 









