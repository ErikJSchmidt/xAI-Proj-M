"""
Within the lowdim directory the low_dim_model_trainer stores all artifacts produced during the training run.
This file contain function that help to evaluate these artifacts, e.g. calculate the accuracy from the produced
predictions and the according labels
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def describe_epoch_result(epoch_result_subfolder_path):
    """
    Goes through the different lists of embeddings, predictions, ... stored per epoch and returns a string that describes
    content of each list

    :param epoch_result_subfolder_path:
    :return:
    """
    description = "Describe contents in " + epoch_result_subfolder_path

    for filename in ['train_embeddings', 'train_labels', 'train_predictions', 'val_embeddings', 'val_labels',
                     'val_predictions']:
        data = torch.load(f"{epoch_result_subfolder_path}/{filename}.pt", map_location=torch.device('cpu'))
        description += f"\n------{filename}-------"
        description += f"\npath: {epoch_result_subfolder_path}/{filename}.pt"
        description += f"\ntype of data: {type(data)}"
        description += f"\nlength of data: {len(data)}"
        description += f"\ntype of elements: {type(data[0])}"
        description += f"\nfirst element: {data[0]}"
        try:
            description += f"\nlength of single sample: {len(data[0])}"
        except:
            pass
        try:
            description += f"\nshape of single sample: {data[0].shape}"
        except:
            pass

    return description


def plot_2d_embeddings(epoch_result_subfolder_path):
    embedding_filename = "train_embeddings"
    label_filename = "train_labels"
    embeddings = torch.load(f"{epoch_result_subfolder_path}/{embedding_filename}.pt", map_location=torch.device('cpu'))
    labels = torch.load(f"{epoch_result_subfolder_path}/{label_filename}.pt", map_location=torch.device('cpu'))

    embeddings_by_class = []
    for class_nr in range(0, 10):
        embeddings_by_class.append([])

    for embedding, class_nr in zip(embeddings, labels):
        embeddings_by_class[class_nr].append(embedding)

    colors = cm.rainbow(np.linspace(0, 1, len(embeddings_by_class)))

    for class_embeddings, color in list(zip(embeddings_by_class, colors)):
        x_values = [embedding[0] for embedding in class_embeddings]
        print(f"max x value:{np.max(x_values)}")
        y_values = [embedding[1] for embedding in class_embeddings]
        print(f"max y value:{np.max(y_values)}")
        plt.scatter(
            x=x_values,
            y=y_values,
            c=color
        )

    plt.show()


def plot_2d_class_centroids(epoch_result_subfolder_path):
    embedding_filename = "train_embeddings"
    label_filename = "train_labels"
    embeddings = torch.load(f"{epoch_result_subfolder_path}/{embedding_filename}.pt", map_location=torch.device('cpu'))
    labels = torch.load(f"{epoch_result_subfolder_path}/{label_filename}.pt", map_location=torch.device('cpu'))

    embeddings_by_class = []
    for class_nr in range(0, 10):
        embeddings_by_class.append([])

    for embedding, class_nr in zip(embeddings, labels):
        embeddings_by_class[class_nr].append(embedding)

    centroids = []
    for class_embeddings in embeddings_by_class:
        x_values = [embedding[0] for embedding in class_embeddings]
        y_values = [embedding[1] for embedding in class_embeddings]

        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)

        centroids.append([x_mean, y_mean])

    colors = cm.rainbow(np.linspace(0, 1, len(embeddings_by_class)))

    for centroid, color in zip(centroids, colors):
        plt.scatter(
            x=[centroid[0]],
            y=[centroid[1]],
            c=color
        )

    plt.show()
