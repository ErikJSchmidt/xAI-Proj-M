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

### Skipped18LayerForEmbedding (divergence loss)
- Remove fc layer and train with divergence loss to optimize distance between class centroid embedding
- Add untrained fc layer
- Safe state dict
- Load state dict and train with cross entropy
- Assess knn classification accuracy

Model weights after 2 epochs divergence_loss are stored in Skipped18LayerForEmbbeding_20240114_2152.
These weights are then loaded and trained with cross entropy for further 38 epochs and finally stored in
Skipped18LayerForEmbbeding_20240114_2315.
Test accuracy on cifar test set(10000 samples) with k=10: **48.93**

Loss: Divergence Loss, Cross Entropy
Epochs 2, 38

