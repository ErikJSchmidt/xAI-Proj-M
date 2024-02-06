# xAI-Proj-M
This repository is part of our participation in the xAI-Proj-M module of the Otto-Friedrich-University Bamberg in the winter semester 2023/24.
The goal of this module is to design and implement CNNs for image classifiactaion, get familiar with PyTorch and follow our own ideas/research questions in the realm of knn classification in the embedding space.
**Group members**: Johannes Miran Langer, Erik Jonathan Schmidt

## Repository content
- custom_cnn: Here we designed our own models and craeted a pipeline for training those models, while tracking configurations and results.
- embeddings: Here we extract embeddings from allready trained models and evaluate their knn-classification performance
- lowdim: Here we adjusted one of our CNNs to only have two features in the embedding layer. Consequently the produced mebeddings can be plotted in a plane which allows for some trouble shooting. 

### Sources
- https://github.com/abhishek-kathuria/CIFAR100-Image-Classification/blob/main/CNN.ipynb
- https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch

#### Center loss papers
- https://ieeexplore.ieee.org/document/9521556
