import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from tqdm import tqdm
from models import AutoEncoder
from utils import MyDataset

if __name__=="__main__":
    # Prepare dataloader
    batch_size = 128
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
        # Load model
    device = 'cuda'
    ckpt = torch.load('model_objects/ckpt_rounded_STI_mse_100e.pth', map_location=device)
    ae_model = AutoEncoder([4, 8, 16]).cuda()
    ae_model = ae_model.to(device)
    ae_model.load_state_dict(ckpt)
    ae_model.requires_grad_(False)
    ae_model.eval()

    # Run
    zs = []
    ys = []
    for x, y in tqdm(data_loader):
        z = ae_model.encoder(x.to(device)).cpu()
        zs.append(z)
        ys.append(y)

    zdata = torch.cat(zs, )
    ydata = torch.cat(ys, )

    
    # Save original
    latent_dataset = TensorDataset(zdata, ydata)
    torch.save(latent_dataset, 'STI_latent_16d.pt')
    print("TensorDataset saved.")

