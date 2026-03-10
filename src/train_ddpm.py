import torch
import functools
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np

from models import UNet_Tranformer
from utils import marginal_prob_std, diffusion_coeff, train_diffusion_model, MyDataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define noise fns, params
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

    # Init model
    print("initialize new score model...")
    score_model = torch.nn.DataParallel(UNet_Tranformer(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)

    # Define training params
    n_epochs = 10
    batch_size = 16
    lr = 10e-4

    #Preparing STI dataset
    label_path = r'data/master_labels.npy'
    labels = np.load(label_path)
    data_path = r'data/master_data_labeled.npy'
    data = np.load(data_path)
    
    labels = torch.from_numpy(labels)
    data = torch.from_numpy(data)

    data = data/255

    labels = labels.long()
    data = data.float()

    dataset = MyDataset(data, labels)

    # Run
    train_diffusion_model(dataset,
                          score_model,
                          marginal_prob_std_fn,
                          n_epochs=n_epochs,
                          batch_size=batch_size,
                          lr=lr,
                          model_name=f"STI_ddpm_mse_{n_epochs}e")