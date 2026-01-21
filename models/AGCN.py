import torch.nn as nn
import torch

# Implements adaptive graph convolution to handle dynamic graph structures.
class AGCN(nn.Module):
    def __init__(self, num_joint, features,):
        super(AGCN, self).__init__()
        self.fc = nn.Linear(in_features=features, out_features=features)
        self.adj = nn.Parameter(torch.eye(num_joint).float(), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L
    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out
class GCNfction(nn.Module):
    def __init__(self, num_joint, features):
        super(GCNfction, self).__init__()
        self.out = nn.Sequential(
            torch.nn.LayerNorm(features),
            AGCN(num_joint, features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            AGCN(num_joint, features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        out = self.out(x)+x
        return out
