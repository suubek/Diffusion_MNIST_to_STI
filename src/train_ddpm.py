import torch
import functools
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from models import UNet_Tranformer
from utils import marginal_prob_std, diffusion_coeff, train_diffusion_model

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
    n_epochs = 100
    batch_size = 16
    lr = 10e-4

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 32x32 (or your desired size)
        transforms.ToTensor(),        # Convert to tensor
        # Optional: Add normalization if needed
        # transforms.Normalize((0.5,), (0.5,))  # For single channel
    ])
    dataset = MNIST('.', train=True, transform=transform, download=True)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Run
    train_diffusion_model(dataset,
                          score_model,
                          marginal_prob_std_fn,
                          n_epochs=n_epochs,
                          batch_size=batch_size,
                          lr=lr,
                          model_name=f"mnist_ddpm_mse_{n_epochs}e")