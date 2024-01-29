## KNN Classification Accuracy
Test different models for embeddings and knn accuracy on them.

### Files
- contains notebooks that use trained models to populate a chroma vector database
- notebook that reads labeled embeddings from chroma and calculates its knn classification accuracy

## Results

### Pretrained ResNet18 from torch
Loaded the model pretrained on ImageNet.
Test accuracy on cifar test set(10000 samples) with k=10: **34.32%**  


### Skipped18LayerForEmbedding 
Our own adoption of resnet18 with the fc-layer separated for later embedding extraction.
Trained on training subset of cifar10.
Test accuracy on cifar test set(10000 samples) with k=10: **79.97%** 

Loss: Cross Entropy  
Epochs: 40

=> We guess that the pretrained model was trained on ImageNet. Our model was trained on

###
L()
