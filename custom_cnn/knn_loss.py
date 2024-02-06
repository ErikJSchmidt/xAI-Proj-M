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
        self.mean_centroid_distance = 0
        
        self.cos = nn.CosineSimilarity(dim = 0)

        if diff == 'euclidian':
            self.diff = self.euclidean_distance


    def loss_renewed(self, input: torch.Tensor, target: torch.Tensor):
        '''
        This is a renewed approach for a combined loss. It does not work, because it optimizes towards cramming all instances into a really small space.
        '''

        self.prep_divergence_renewed(input, target)
        # After calling this method, self.centroids will always have shape[0]==len(self.classes)
        
        # Calculate mean cosine similarities between centroids
        d = []
        # for centroid_a in self.centroids:
        #     local_d = []    # for each centroid collect distances to other centroids in a list. This list should be ALWAYS be 9 elements long
        #     for centroid_b in self.centroids:
        #         if not torch.equal(centroid_a, centroid_b):
        #             local_d.append(self.cosine_similarity(centroid_a, centroid_b))
        #     local_t = torch.stack(local_d)
        #     d.append(torch.mean(local_t))
        for _class in self.classes:
            centroid_a = self.centroids[int(_class)]
            local_d = []
            for __class in self.classes:
                centroid_b = self.centroids[int(__class)]
                if not _class == __class:
                    local_d.append(self.cosine_similarity(centroid_a, centroid_b))
            local_t = torch.stack(local_d)
            d.append(torch.mean(local_t))
        
        self.mean_centroid_distance = torch.stack(d).abs().mean()

        # Calculate mean euclidean distance between instances and their centroids
        instance_d = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            instance_d.append(self.euclidean_distance(inp, corresponding_centroid))

        d = torch.mul(torch.stack(instance_d), 1)

        convergence = torch.mean(d)
        uniformity = torch.std(d)
        divergence = torch.div(1, torch.sub(1, self.mean_centroid_distance))    # Cosine Similarity maxes out at 

        # We don't need a ratio with cosine similarity (since it is between -1 and 1, or 0 and 1 as we take the abs)

        return torch.mul(divergence, convergence.add(uniformity))


    def combined_loss(self, input: torch.Tensor, target: torch.Tensor):
        '''
        '''

        centroid_distances = self.prepare_divergence(input, target)
        self.mean_centroid_distance = torch.mean(torch.stack(centroid_distances))

        instance_distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            instance_distances.append(self.euclidean_distance(inp, corresponding_centroid))

        dists = torch.mul(torch.stack(instance_distances), 0.5)

        convergence = torch.mean(dists)
        uniformity = torch.std(dists)
        divergence = torch.div(1, self.mean_centroid_distance).mul(1000)
        scaler = torch.div(
            torch.add(convergence, uniformity),
            self.mean_centroid_distance
        )

        return torch.mul(scaler, torch.add(convergence, torch.add(uniformity, divergence)))


    def combined_loss_optim(self, input: torch.Tensor, target: torch.Tensor):
        '''
        '''

        if self.centroids is None:
            self.prepare_divergence(input, target)

        centroid_distances = self.prepare_divergence(input, target)
        self.mean_centroid_distance = torch.mean(torch.stack(centroid_distances))

        instance_distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            instance_distances.append(self.euclidean_distance(inp, corresponding_centroid))

        dists = torch.mul(torch.stack(instance_distances), 1)

        convergence = torch.mean(dists)
        uniformity = torch.std(dists)
        divergence = torch.div(1, self.mean_centroid_distance).mul(1000)

        ratio = torch.pow(
            torch.div(
                torch.add(convergence, uniformity),
                torch.mul(self.mean_centroid_distance, 2)
            ),
            10
        )

        return torch.mul(ratio, torch.add(convergence, torch.add(uniformity, divergence)))


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

        distances = self.prepare_divergence(input, target)
        self.mean_centroid_distance = torch.mean(torch.stack(distances))
        
        return torch.div(1, self.mean_centroid_distance)
    

    def divergence_diagnostic(self):
        '''
        Getter for self.mean_centroid_distance
        '''

        return self.mean_centroid_distance
    

    def divergence_renewed(self, input : torch.Tensor, target : torch.Tensor):
        '''
        Renewed Divergence loss, working towards generating spread-out centroids.

        TODO: Test this in the lowdim branch. For some reason, training on these centroids does not work, while training on fixed centroids does...
        '''

        self.prep_divergence_renewed(input, target)

        d = []
        for _class in self.classes:
            centroid_a = self.centroids[int(_class)]
            local_d = []
            for __class in self.classes:
                centroid_b = self.centroids[int(__class)]
                if not _class == __class:
                    local_d.append(self.cosine_similarity(centroid_a, centroid_b))
            local_t = torch.stack(local_d)
            d.append(torch.mean(local_t))
        
        self.mean_centroid_distance = torch.stack(d).abs().mean()
        return torch.div(1, torch.sub(1, self.mean_centroid_distance))


    def convergence_loss(self, input : torch.Tensor, target : torch.Tensor):
        '''
        Convergence Loss function, with the aim of minimizing the average (summed?) difference between the instances and corresponding class centroid. This function assumes that the object's centroids attribute contains embeddings that are sufficiently different from each other.
        
        PARAMETERS
        ----------
        input: tensor of shape[0] = batch_size containing the embeddings for the samples
        target: tensor of same shape[0] as input containing the corresponding labels
        
        RETURNS
        -------
        The average distance between the instances and the corresponding class's centroids.
        '''

        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            distances.append(self.euclidean_distance(inp, corresponding_centroid))

        return torch.mean(torch.stack(distances))
    

    def combined_convergence(self, input: torch.Tensor, target: torch.Tensor):
        '''
        '''
        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            distances.append(self.euclidean_distance(inp, corresponding_centroid))
        
        dists = torch.mul(torch.stack(distances), 1.0) # Reduce sizes to remove overflows. Probably unnecessary when not using an exponent. 

        # weighed_con = torch.mul(torch.mean(dists), 10.0)
        # weighed_uni = torch.mul(torch.std(dists), 5.0)
        # weighed_max = torch.mul(torch.max(dists), 1.0)

        return torch.mul(torch.mul(torch.mean(dists), 2.0), torch.std(dists))
    

    def convergence_range(self, input : torch.Tensor, target : torch.Tensor):
        '''
        Performs the Convergence Loss function, but does not return the average, but the max and min. This is for debugging purposes, to see whether there is extreme variance in the distances.

        PARAMETERS
        ----------
        input: tensor of shape[0] = batch_size containing the embeddings for the samples
        target: tensor of same shape[0] as input containing the corresponding labels
        
        RETURNS
        -------
        tuple: (max_dist, min_dist)
        '''

        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            distances.append(self.euclidean_distance(inp, corresponding_centroid))
        
        dists = torch.stack(distances)

        return (dists.max(), dists.min())
    

    def max_convergence_loss(self, input : torch.Tensor, target : torch.Tensor):
        '''
        Max Convergence Loss function, with the aim of minimizing the MAXIMUM deviation from the class centroid in any batch.

        PARAMETERS
        ----------
        input: tensor of shape[0] = batch_size containing the embeddings for the samples
        target: tensor of same shape[0] as input containing the corresponding labels

        RETURNS
        -------
        tensor: max distance from class centroid
        '''

        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            distances.append(self.euclidean_distance(inp, corresponding_centroid))

        return torch.max(torch.stack(distances))


    def uniformity_loss(self, input : torch.Tensor, target : torch.Tensor):
        '''
        Uniformity Loss function, with the aim of minimizing the standard deviation of the individual instance-to-centroid distances.

        PARAMETERS
        ----------
        input: tensor of shape[0] = batch_size containing the embeddings for the samples
        target: tensor of same shape[0] as input containing the corresponding labels

        RETURNS
        -------
        tensor: std of distances
        '''

        distances = []
        for inp, tar in zip(input, target):
            label = int(tar)
            corresponding_centroid = self.centroids[label].detach()
            distances.append(self.euclidean_distance(inp, corresponding_centroid))

        return torch.std(torch.stack(distances))

###################
##### HELPERS #####
###################

    def get_centroids(self):
        return self.centroids

    def euclidean_distance(self, tensor_a : torch.Tensor, tensor_b : torch.Tensor, dim = 0):
        '''
        Returns euclidean distance between two tensors.

        PARAMETERS
        ----------
        tensor_a, tensor_b : torch.Tensor of which the euclidean distance is to be calculated.
        dim : int along which dimension the tensor will be summed. This can most likely be left at the default. If not, you will probably know.

        RETURNS
        -------
        torch.Tensor containing a single scalar, the euclidean distance. This tensor remains attached to the computational graph.
        '''
        return (tensor_b - tensor_a).pow(2).sum(dim).sqrt()

    
    def cosine_similarity(self, tensor_a, tensor_b):
        return self.cos(tensor_a, tensor_b)

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
    
    def prep_divergence_renewed(self, input, target):
        '''
        This is a function that calculates the class centroids of a batch and updates self.centroids. It takes the current values of self.centroids into account to allow for two things: It a) ensures that the self.centroids tensor is always of len(self.classes) and b) that there is some random initiation, i.e. some initial scattering of the centroids of the hyper-diagonal (is that even a word)?
        This method does not return anything. It only updates self.centroids.
        '''

        # If there are no centroids, randomly generate some
        if self.centroids is None:
            self.centroids = utility_functions.to_device(torch.rand(len(self.classes), input.shape[1]), self.device)

        centroids = []
        for _class in self.classes:
            class_tensors = [self.centroids[_class].detach()]                # The first value in the class tensors is the old centroid. This has very little impact but makes sure that the list is never empty.
            for inp, tar in zip(input, target):
                if int(tar) == _class:
                    class_tensors.append(inp)
            stacked_tensor = torch.stack(class_tensors)
            centroids.append(torch.mean(stacked_tensor, dim = 0))

        self.centroids = torch.stack(centroids)

    
    def predefine_centroids(self, dimensionality : int):
        '''
        '''

        centroids = []
        for _class in self.classes:
            centroid = []
            for i in range(dimensionality):
                if i % len(self.classes) == _class:
                    centroid.append(1)
                else:
                    centroid.append(0)
            centroids.append(utility_functions.to_device(torch.tensor(centroid), self.device))

        self.centroids = torch.stack(centroids)

