import torch
import torch.nn as nn
from utility_functions import get_default_device, to_device
from sklearn.metrics import accuracy_score
from custom_cnn.networks.model_skipped_18_layer_for_low_dim_embedding import Skipped18LayerForLowDimEmbbeding


class LowDimModelWrapper:
    """
    The idea of a training run performed with this model wrapper is to train the model in different phases.
    The goal is to spread out the embeddings of, so that the centroids of different classes are further away.

    Training phases:
     1. Train with cross entropy
     2. Train with combination of centroid maximizing and class coherence loss

    """

    def __init__(self):
        self.model = to_device(Skipped18LayerForLowDimEmbbeding(), get_default_device())


    def training_step(self, images):
        self.model.train()
        embeddings, out = self.model(images)

        return embeddings, out

    def validation_step(self, images):
        self.model.eval()
        embeddings, out = self.model(images)

        return embeddings, out

    def save_model(self, dir_path):
        torch.save(
            self.model.network.state_dict(),
            dir_path + "/model_state_dict"
        )

    def load_model_weights(self, model_path):
        self.model.network.load_state_dict(
            torch.load(model_path, map_location=torch.device(get_default_device())))