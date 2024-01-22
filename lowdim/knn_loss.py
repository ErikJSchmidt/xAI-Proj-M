import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utility_functions

class KNNLoss():
    '''
    Wrapper for embedding-space loss functions.
    '''
    def __init__(self, classes, diff = 'euclidean'):
        self.centroids = None
        # self.diff = self.euclidean_difference() # Currently only uses euclidean
        self.classes = classes
        self.device = utility_functions.get_default_device()

        if diff == 'euclidian':
            self.diff = self.euclidian_difference


    def divergence_loss(self, input: torch.Tensor, target: torch.Tensor):
        '''
        Divergence Loss function, with the aim of maximizing the difference between the class's centroids.
        This loss function assumes, that a randomly initialized network will randomly embed inputs, such that the
        centroid of each class is roughly the same.

        PARAMETERS
        ----------
        input : tensor of shape[0] = batch_size containing the embeddings for the samples
        target : tensor of same shape[0] as input containing the corresponding labels

        RETURNS
        -------
        The average distance between the class's centroids in the embedding space in a 1D tensor.
        '''

        # DONE: dis one work

        distances = self.prepare_divergence(input, target)

        # sample_losses = []
        # for inp, tar in zip(input, target):
        #     sample_losses.append(distances[int(tar)])
        
        return torch.mean(torch.stack(distances)).mul(-1)


    # Try to make this loss function closer to F.cross_entropy, so that we can train a model with cross entropy for some
    # epochs and then switch to this loss function


    def divergence_loss_adjusted(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ):
        '''
        Divergence Loss function, with the aim of maximizing the difference between the class's centroids.
        This loss function assumes, that a randomly initialized network will randomly embed inputs, such that the
        centroid of each class is roughly the same.

        PARAMETERS
        ----------
        forward_pass : tensor of shape[0] = batch_size containing the embeddings for the samples
        labels : tensor of same shape[0] as forward_pass containing the corresponding labels

        RETURNS
        -------
        The average distance between the class's centroids in the embedding space in a 1D tensor.
        '''

        batch_size = input.shape[0]

        if input.shape[1] != len(self.classes):
            print("Number of classes in input does not match number of classes the KNNloss was initialized for")

        # Calculate Current Centroids.
        centroids = []
        for _class in self.classes:
            class_tensors = []
            for i in range(batch_size):
                target_class_porbs = target[i]
                target_class = np.argmax(target_class_porbs)
                if target_class == _class:
                    class_tensors.append(input[i])
            stacked_tensor = torch.stack(class_tensors)
            centroids.append(torch.mean(stacked_tensor, dim = 0))

        self.centroids = torch.stack(centroids)

        # Calculate average distance between centroids.
        # This part is combinatorial with respect to the number of classes.
        distances = []
        for class_a, centroid_a in zip(self.classes,self.centroids):
            local_distances = []
            for class_b, centroid_b in zip(self.classes, self.centroids):
                if not class_a == class_b:
                    local_distances.append(self.euclidean_distance(centroid_a, centroid_b))
            distances.append(torch.mean(torch.tensor(local_distances)))

        return torch.mean(torch.tensor(distances).pow(-1)).item()


    def convergence_loss(self, input, target):
        '''
        Convergence Loss function, with the aim of minimizing the average (summed?) difference between the instances and
        corresponding class centroid. This function assumes that the object's centroids attribute contains embeddings
        that are sufficiently different from each other.
        
        PARAMETERS
        ----------
        foward_pass: tensor of shape[0] = batch_size containing the embeddings for the samples
        labels: tensor of same shape[0] as forward_pass containing the corresponding labels
        
        RETURNS
        -------
        The average distance between the instances and the corresponding class's centroids.
        '''

        # batch_size = forward_pass.shape[0]

        #TODO: dis one does not work yet

        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label]
            dist = self.euclidean_distance(inp, corresponding_centroid)
            distances.append(dist)

        return torch.sum(torch.tensor(distances, requires_grad=True))
        # return torch.sum(torch.tensor(distances)).item()


    def convergence_loss_adjusted(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ):
        '''
        Convergence Loss function, with the aim of minimizing the average (summed?) difference between the instances and
        corresponding class centroid. This function assumes that the object's centroids attribute contains embeddings
        that are sufficiently different from each other.

        PARAMETERS
        ----------
        foward_pass: tensor of shape[0] = batch_size containing the embeddings for the samples
        labels: tensor of same shape[0] as forward_pass containing the corresponding labels

        RETURNS
        -------
        The average distance between the instances and the corresponding class's centroids.
        '''

        batch_size = input.shape[0]
        if input.shape[1] != len(self.classes):
            print("Number of classes in input does not match number of classes the KNNloss was initialized for")


        distances = []
        for i in range(batch_size):
            instance = input[i]
            label = int(np.argmax(target[i]))
            corresponding_centroid = self.centroids[label]
            dist = self.euclidean_distance(instance, corresponding_centroid)
            distances.append(dist)

        return torch.mean(torch.tensor(distances)).item()

    def combined_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ):
        return self.divergence_loss_adjusted(input, target) + self.convergence_loss_adjusted(input, target)


    def get_centroids(self):
        return self.centroids

    def euclidean_distance(self, tensor_a, tensor_b, dim = 0):
        return (tensor_b - tensor_a).pow(2).sum(dim).sqrt()

    def prepare_divergence(self, input, target):
        '''
        '''
        centroids = []
        for _class in self.classes:
            class_tensors = []
            for inp, tar in zip(input, target):
                if int(tar) == _class:
                    class_tensors.append(inp)
            stacked_tensor = torch.stack(class_tensors)
            centroids.append(torch.mean(stacked_tensor, dim = 0))
        
        self.centroids = torch.stack(centroids)

        distances = []
        for centroid_a in centroids:
            local_distances = []
            for centroid_b in centroids:
                if not torch.equal(centroid_a, centroid_b):
                    local_distances.append(self.euclidean_distance(centroid_a, centroid_b))
            local_tensor = torch.stack(local_distances)
            distances.append(torch.mean(local_tensor))

        return distances
