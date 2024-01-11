## KNN Classification Accuracy
Test different models for embeddings and knn accuracy on them.

### Pretrained ResNet18 from torch
Loaded the model pretrained on Cifar10.
Test accuracy on cifar test set(10000 samples) with k=10: **34.32%**  


### Skipped18LayerForEmbedding 
Our own adoption of resnet18 with the fc-layer separated for later embedding extraction.

Loss: Cross Entropy  
Epochs: 40
