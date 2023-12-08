import torch
import torch.nn as nn
import torch.nn.functional as F

class KNNLoss():
    '''
    Wrapper for embedding-space loss functions.
    '''
    def __init__(self, classes, diff = 'euclidean'):
        self.centroids = None
        # self.diff = self.euclidean_difference() # Currently only uses euclidean
        self.classes = classes

        if diff == 'euclidian':
            self.diff = self.euclidian_difference()


    def divergence_loss(self, forward_pass, labels):
        '''
        Divergence Loss function, with the aim of maximizing the difference between the class's centroids.
        This loss function assumes, that a randomly initialized network will randomly embed inputs, such that the centroid of each class is roughly the same.

        PARAMETERS
        ----------
        forward_pass : tensor of shape[0] = batch_size containing the embeddings for the samples
        labels : tensor of same shape[0] as forward_pass containing the corresponding labels

        RETURNS
        -------
        The average distance between the class's centroids in the embedding space in a 1D tensor.
        '''

        batch_size = forward_pass.shape[0]

        # Calculate Current Centroids.
        centroids = []
        for _class in self.classes:
            class_tensors = []
            for i in range(batch_size):
                if labels[i] == _class:
                    class_tensors.append(forward_pass[i])
            stacked_tensor = torch.stack(class_tensors)
            centroids.append(torch.mean(stacked_tensor, dim = 0))

        self.centroids = torch.stack(centroids)

        # Calculate average distance between centroids.
        # This part is combinatorial with respect to the number of classes.
        distances = []
        for centroid_a in self.centroids:
            local_distances = []
            for centroid_b in self.centroids:
                if not torch.equal(centroid_a, centroid_b):
                    local_distances.append(self.euclidean_distance(centroid_a, centroid_b))
            distances.append(torch.mean(torch.tensor(local_distances)))


        return torch.mean(torch.tensor(distances).pow(-0.1)).item()


    def convergence_loss(self, forward_pass, labels):
        '''
        Convergence Loss function, with the aim of minimizing the average (summed?) difference between the instances and corresponding class centroid. This function assumes that the object's centroids attribute contains embeddings that are sufficiently different from each other.
        
        PARAMETERS
        ----------
        foward_pass: tensor of shape[0] = batch_size containing the embeddings for the samples
        labels: tensor of same shape[0] as forward_pass containing the corresponding labels
        
        RETURNS
        -------
        The average distance between the instances and the corresponding class's centroids.
        '''

        batch_size = forward_pass.shape[0]

        distances = []
        for i in range(batch_size):
            instance = forward_pass[i]
            label = int(labels[i].item())
            corresponding_centroid = self.centroids[label]
            dist = self.euclidean_distance(instance, corresponding_centroid)
            distances.append(dist)

        return torch.mean(torch.tensor(distances)).item()


    def get_centroids(self):
        return self.centroids

    def euclidean_distance(self, tensor_a, tensor_b, dim = 0):
        return (tensor_b - tensor_a).pow(2).sum(dim).sqrt()

