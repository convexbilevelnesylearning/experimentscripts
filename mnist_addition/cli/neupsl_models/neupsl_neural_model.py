#!/usr/bin/env python3
import importlib
import os
import sys

import numpy as np
import torch
import torchvision

import pslpython.deeppsl.model

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', 'scripts'))
util = importlib.import_module("util")

from mnist_addition.cli.neupsl_models.mnist_classifier import MNIST_Classifier
from mnist_addition.cli.neupsl_models.mlp import MLP


class MNISTAdditionModel(pslpython.deeppsl.model.DeepModel):
    def __init__(self):
        super().__init__()
        self._application = None

        self._model = None
        self._predictions = None

        self._training_transforms = None
        self._optimizer = None
        self._scheduler = None

        self._epoch = 0
        self._iteration = 0

        self._features = None
        self._digit_labels = None

        self._device = get_torch_device()

    def internal_init_model(self, application, options={}):
        self._application = application

        options = options.copy()
        self._model = self._create_model(options=options).to(self._device)

        if self._application == 'learning':
            self._epoch = 0
            self._iteration = 0
            if options['freeze_resnet'] == 'true':
                self._optimizer = torch.optim.Adam(self._model.mlp.parameters(), lr=float(options['neural_learning_rate']), weight_decay=float(options['weight_decay']))
            else:
                self._optimizer = torch.optim.Adam(self._model.parameters(), lr=float(options['neural_learning_rate']), weight_decay=float(options['weight_decay']))

            self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=int(options['learning_rate_decay_step']), gamma=float(options['learning_rate_decay']))

            if options['transforms'] == 'true':
                self._training_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomRotation(degrees=30),
                    torchvision.transforms.ElasticTransform(alpha=50.0)
                ])
            else:
                self._training_transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomRotation(degrees=0)
                ])
        elif self._application == 'inference':
            self._model.load_state_dict(torch.load(options['save-path'], map_location=self._device))
            self._model.temperature = 0.5

        return {}

    def internal_fit(self, data, gradients, options={}):
        self._prepare_data(data, options=options)

        structured_gradients = torch.tensor(gradients.astype(np.float32), dtype=torch.float32, device=self._device)

        self._optimizer.zero_grad()

        self._predictions.backward(structured_gradients, retain_graph=True)

        total_gradient_norm = 0.0
        for p in self._model.mlp.parameters():
            param_norm = p.grad.data.norm(2)
            total_gradient_norm += param_norm.item() ** 2
        total_gradient_norm = total_gradient_norm ** (1. / 2)

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 3.0)

        self._optimizer.step()

        # Compute the new training loss.
        new_output = self._model(self._features)

        loss = torch.nn.functional.cross_entropy(new_output, self._digit_labels).item()

        results = {"training_classification_loss": loss,
                   "gradient_norm_2": total_gradient_norm,
                   "struct gradient_norm_2": torch.norm(structured_gradients, 2).item(),
                   "struct gradient_norm_infty": torch.norm(structured_gradients, torch.inf).item()}

        self._iteration += 1

        return results

    def internal_predict(self, data, options={}):
        self._prepare_data(data, options=options)

        results = {}

        if options["learn"]:
            results['mode'] = 'learning'
            self._model.train()
            self._predictions = self._model(self._training_transforms(self._features))
        else:
            results['mode'] = 'inference'
            self._model.eval()
            self._predictions = self._model(self._features)

        return self._predictions.cpu().detach(), results

    def internal_epoch_end(self, options={}):
        if self._application == 'learning':
            self._epoch += 1
            self._scheduler.step()
            self._model.temperature_step(self._epoch, rate=float(options['temperature_decay_rate']))

        return {}

    def internal_eval(self, data, options={}):
        self._prepare_data(data, options=options)
        options['learn'] = False
        predictions, _ = self.internal_predict(data, options=options)
        results = {'training_loss': torch.nn.functional.cross_entropy(predictions.to(self._device), self._digit_labels).item()}

        return results

    def internal_save(self, options={}):
        os.makedirs(os.path.dirname(options['save-path']), exist_ok=True)
        torch.save(self._model.state_dict(), options['save-path'])

        return {}

    def _create_model(self, options={}):
        resnet18_ = torchvision.models.resnet18()
        resnet18_.conv1 = torch.nn.Conv2d(1, resnet18_.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        backbone = torchvision.models.resnet18(num_classes=128)

        backbone.load_state_dict(torch.load(
            f"{THIS_DIR}/../../data/experiment::{options['experiment']}/split::{options['split']}/train-size::{options['train_size']}/overlap::0.00/saved-networks/simclr-pretrained-backbone.pt",
            map_location=self._device
        ))

        return MNIST_Classifier(
            backbone,
            MLP(128, 64, int(options['class-size']), 2, float(options["dropout"])),
            device=self._device)

    def _prepare_data(self, data, options={}):
        self._features = torch.tensor(np.asarray(data[:, :-1], dtype=np.float32), dtype=torch.float32, device=self._device)
        self._features = self._features.reshape(self._features.shape[0], 1, 28, 28)
        self._digit_labels = torch.tensor(np.asarray([util.one_hot_encoding(int(label), int(options['class-size'])) for label in data[:, -1]], dtype=np.float32), dtype=torch.float32, device=self._device)


def get_torch_device():
    return torch.device("cpu")
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    # else:
    #     return torch.device("cpu")
