import sys
import time
from torchvision.datasets.utils import download_url
import tarfile



# --- Downloading Cifar10 ----
# Download the dataset to data directory
def download_and_unpack_dataset(data_dir):
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, data_dir)
    print("downloaded the compressed dataset")
    time.sleep(5)
    print("start extracting the dataset")
    # Extract dataset from compressed file
    with tarfile.open(data_dir + '/cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')


if __name__ == "__main__":
    data_dir = sys.argv[1]
    print("download cifar10 dataset to " + data_dir)
    download_and_unpack_dataset(data_dir)

