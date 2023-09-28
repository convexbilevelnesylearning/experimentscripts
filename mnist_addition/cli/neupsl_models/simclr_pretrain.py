"""
SimCLRPreTrain.py
Code for pre-training a model using SimCLR on MNIST.
Adapted from: <https://github.com/sthalles/SimCLR>
"""

import os
import pandas as pd
import sys
import torch
import torchvision
import tqdm

from torch.utils.data import Dataset

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

from mnist_addition.cli.neupsl_models.mlp import MLP


class SimCLR(object):
    def __init__(self, model, optimizer, scheduler, temperature, size, device):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        labels = torch.cat([torch.arange(size, device=self.device) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        self.mask = mask
        self.labels = labels.bool()
        self.contrastive_output_labels = torch.zeros((2 * size),
                                                     dtype=torch.long, device=self.device)

    def info_nce_loss(self, features):
        features = torch.nn.functional.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix[~self.mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[self.labels].view(self.labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~self.labels].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        logits = logits / self.temperature
        return logits

    def train(self, train_loader, epochs):
        for epoch_counter in range(epochs):
            with tqdm.tqdm(train_loader) as tq:
                tq.set_description("Epoch:{}".format(epoch_counter))
                for images in tq:
                    images = torch.cat(images, dim=0)
                    images = images.to(self.device)

                    features = self.model(images)
                    logits = self.info_nce_loss(features)
                    loss = self.criterion(logits, self.contrastive_output_labels)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_value = loss.item()

                    self.optimizer.step()

                    tq.set_postfix(loss=loss_value)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()


def get_simclr_pipeline_transform():
    """Return a set of data augmentation transformations fitting the MNIST Dataset."""
    data_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(degrees=(0, 90)),
        torchvision.transforms.ElasticTransform(alpha=125.0),
        torchvision.transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])

    return data_transforms


class ContrastiveLearningMNISTDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.transform = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform())

    def __getitem__(self, index):
        return self.transform(self.features[index])

    def __len__(self):
        return len(self.features)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(2)]


def simclr_pretrain(data_folder):
    """Pre-train a model using SimCLR on MNIST."""
    device = get_torch_device()

    training_indices = torch.tensor(
        pd.read_csv(f"{data_folder}/image-target-train.txt", sep="\t", header=None)[0].unique())
    all_features = pd.read_csv(f"{data_folder}/entity-data-map.txt", sep="\t", header=None, index_col=0)
    training_features = torch.tensor(all_features.loc[training_indices].iloc[:, :-1].values, dtype=torch.float32)
    training_features = training_features.reshape(-1, 1, 28, 28)
    training_features = training_features.expand(-1, 3, -1, -1)

    mnist_dataset = ContrastiveLearningMNISTDataset(training_features)

    resnet18_ = torchvision.models.resnet18()
    resnet18_.conv1 = torch.nn.Conv2d(1, resnet18_.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = torchvision.models.resnet18(num_classes=128)

    projection_head = MLP(128, 128, 128, 2)

    model = torch.nn.Sequential(backbone, projection_head)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.9)

    simclr_pretrain = SimCLR(model.to(device), optimizer, scheduler,
                             temperature=0.07, size=min(1024, training_features.shape[0]), device=device)

    simclr_pretrain.train(
        torch.utils.data.DataLoader(mnist_dataset,
                                    batch_size=min(1024, training_features.shape[0]), shuffle=True,
                                    num_workers=0, pin_memory=True, drop_last=True),
        epochs=250)

    os.makedirs(f"{data_folder}/saved-networks", exist_ok=True)

    torch.save(backbone.state_dict(), f"{data_folder}/saved-networks/simclr-pretrained-backbone.pt")
    torch.save(projection_head.state_dict(), f"{data_folder}/saved-networks/simclr-pretrained-projection-head.pt")


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    for experiment in ["mnist-1", "mnist-2"]:
        for split in [0, 1, 2, 3, 4]:
            for train_size in ["00600", "06000", "50000"]:
                if os.path.exists(f"{THIS_DIR}/../../data/experiment::{experiment}/split::{split}/train-size::{train_size}/overlap::0.00/saved-networks/simclr-pretrained-backbone.pt"):
                    print(f"Found pretrained models for experiment::{experiment}/split::{split}/train-size::{train_size}/overlap::0.00. Skipping...")
                    continue

                simclr_pretrain(f"{THIS_DIR}/../../data/experiment::{experiment}/split::{split}/train-size::{train_size}/overlap::0.00")


if __name__ == "__main__":
    main()
