"""
Within the lowdim directory the low_dim_model_trainer stores all artifacts produced during the training run.
This file contain function that help to evaluate these artifacts, e.g. calculate the accuracy from the produced
predictions and the according labels
"""
import torch

def describe_epoch_result(epoch_result_subfolder_path):
    """
    Goes through the different lists of embeddings, predictions, ... stored per epoch and returns a string that describes
    content of each list

    :param epoch_result_subfolder_path:
    :return:
    """
    description = "Describe contents in " + epoch_result_subfolder_path

    for filename in ['train_embeddings', 'train_labels', 'train_predictions', 'val_embeddings', 'val_labels', 'val_predictions']:
        data = torch.load(f"{epoch_result_subfolder_path}/{filename}.pt", map_location=torch.device('cpu'))
        description +=f"\n------{filename}-------"
        description +=f"\npath: {epoch_result_subfolder_path}/{filename}.pt"
        description +=f"\ntype of data: {type(data)}"
        description +=f"\nlength of data: {len(data)}"
        description +=f"\ntype of elements: {type(data[0])}"
        description +=f"\nfirst element: {data[0]}"
        try:
            description +=f"\nlength of single sample: {len(data[0])}"
        except:
            pass
        try:
            description +=f"\nshape of single sample: {data[0].shape}"
        except:
            pass

    return description