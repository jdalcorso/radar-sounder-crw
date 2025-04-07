# Semantic Segmentation of Radargrams via Unsupervised Random Walks and User-Guided Label Propagation

This is the repository of the work [An Approach to Semantic Segmentation of Radar Sounder Data Based on Unsupervised Random Walks and User-Guided Label Propagation](https://doi.org/10.1109/TGRS.2024.3458188).
The project can be easily replicated using the associated Docker container. To create and launch the container use:
```
bash launch_docker.sh <name> <tag>
```
and choose a name and a tag for the container.
Within the running container, you can use:
```
python train.py <args>
```
to launch the unsupervised training step described in the paper above. Be sure to set all the paths to datasets and create the datasets in needed. Mind that this is a developer container, hence one would likely modify part of the source code to use a new dataset.

Part of the tests and plots used for the paper can be found within the `\test\` folder. One would slightly modify the scripts to obtain a personalized script for performing inference with a trained model.

For comments, clarifications and issues, please write to [jordy.dalcorso@unitn.it](mailto:jordy.dalcorso@unitn.it)

For citations, use the following: 
```
@ARTICLE{10677400,
  author={Dal Corso, Jordy and Bruzzone, Lorenzo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={An Approach to Semantic Segmentation of Radar Sounder Data Based on Unsupervised Random Walks and User-Guided Label Propagation}, 
  year={2024},
  volume={62},
  number={},
  pages={1-19},
  keywords={Radar;Semantic segmentation;Instruments;Feature extraction;Training;Measurement;Deep learning;Label propagation;multichannel coherent radar depth sounder (MCoRDS);radar sounder (RS);random walks;shallow radar (SHARAD);unsupervised learning},
  doi={10.1109/TGRS.2024.3458188}}
```

For an earlier work addressing horizontal correlation within radargrams see also [Radargrams as Sequences: A Method for The Semantic Segmentation of Radar Sounder Data](https://doi.org/10.1109/IGARSS53475.2024.10641860).