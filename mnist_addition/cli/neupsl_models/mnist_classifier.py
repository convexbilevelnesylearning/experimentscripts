import torch
import numpy as np


class MNIST_Classifier(torch.nn.Module):
    def __init__(self, backbone, mlp, initial_temperature=1.0, device="cpu"):
        super(MNIST_Classifier, self).__init__()

        self.backbone = backbone
        self.mlp = mlp
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.center = 0.0

        self.device = device

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        x = self.mlp(x)

        x = x - self.center

        return self.gumbel_softmax(x, temperature=self.temperature, hard=False)

    def sample_gumbel(self, shape, eps=1e-12):
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return torch.nn.functional.softmax(y / temperature, dim=1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumble-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def temperature_step(self, epoch, rate=1.0e-4):
        self.temperature = max(0.5, self.initial_temperature * np.exp(-1.0 * rate * epoch))
