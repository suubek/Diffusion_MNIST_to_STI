from lpips import LPIPS
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import trange
from torch.optim import Adam
from models import AutoEncoder
from utils import MyDataset

if __name__ == "__main__":
    # Define the loss function, MSE and LPIPS
    #lpips = LPIPS(net="squeeze").cuda()
    loss_fn_ae = lambda x,xhat: nn.functional.mse_loss(x, xhat)# + lpips(x.repeat(1,3,1,1), x_hat.repeat(1,3,1,1)).mean()

    ae_model = AutoEncoder([4, 8, 16]).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs =   100  #@param {'type':'integer'}
    ## size of a mini-batch
    batch_size =  128   #@param {'type':'integer'}
    ## learning rate
    lr=10e-4 #@param {'type':'number'}

    # dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)    
    

    total_params = sum(p.numel() for p in ae_model.parameters())
    trainable_params = sum(p.numel() for p in ae_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("-----------------------------")

    optimizer = Adam(ae_model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            z = ae_model.encoder(x)
            x_hat = ae_model.decoder(z)
            loss = loss_fn_ae(x, x_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
    print('{} Average Loss: {:5f}'.format(epoch, avg_loss / num_items))
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(ae_model.state_dict(), f'ckpt_STI_mse_{n_epochs}e.pth')