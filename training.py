from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unet import ContextUnet
from diffusion_utilities import *
import os

# ======================================================================================

# diffusion hyperparameters
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

# network hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 # 64 hidden dimension feature
n_cfeat = 5 # context vector is of size 5
height = 16 # 16x16 image
save_dir = './weights'

# training hyperparameters
batch_size = 100
n_epoch = 10
lrate=0.1

# ======================================================================================

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device)
a_t = 1 - b_t
ab_t = torch.sum(a_t.log(), dim=0)
ab_t[0] = 1

# helper function: perturbs an image to a specified noise level
def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * noise + (1 - ab_t[t, None, None, None]) * x

# ======================================================================================

# construct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# load dataset and construct optimizer
dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

# set into train mode
nn_model.train()

for ep in range(n_epoch):
    print(f'epoch {ep}')

    pbar = tqdm(dataloader, mininterval=2)
    for x, c in pbar:   # x: images  c: context
        optim.zero_grad()
        x = x.to(device)
        c = c.to(x)

        # randomly mask out c
        context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
        c = c * context_mask.unsqueeze(1)

        # perturb data
        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
        x_pert = perturb_input(x, t, noise)

        # use network to recover noise
        pred_noise = nn_model(x_pert, t / timesteps, c=c)

        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()

        optim.step()

torch.save(nn_model.state_dict(), save_dir + f"/context_model.pth")
print('saved model at ' + save_dir + f"/new_context_model.pth")
