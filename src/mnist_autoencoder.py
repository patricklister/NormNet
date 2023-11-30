import torchvision.models as models
import torch.optim as opt
from prodigyopt import Prodigy
from torch.utils.data import DataLoader
import functions
import torch
import mnist_model
from torchvision.datasets import MNIST
from torchvision import transforms

batch_size, n_epochs, weight_file, loss_image, device, weight_decay = functions.parse_args()

n_epochs = 100
loss_image = './loss_graph/loss_euclid_autoencoder_negative_1over2_V7.png'
device = 'cuda'


train_transform = transforms.Compose([transforms.ToTensor(), torch.flatten])

train_set = MNIST('./data/mnist', train=True, download=True,
transform=train_transform)
validation_set = MNIST('./data/mnist', train=False, download=True,
transform=train_transform)

device = 'cuda'

model = mnist_model.mnistautoencoder()
#model.load_state_dict(torch.load(f"./saved_train/euclid_autoencoder_v5_epoch_99.pth"))
model.to(device)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=512, shuffle=True)

optimizer = opt.Adam(model.parameters(), lr=1e-3)
scheduler = opt.lr_scheduler.ExponentialLR(optimizer, gamma=1)

functions.train(model, n_epochs, train_loader, val_loader, device, optimizer, scheduler, loss_fn=torch.nn.BCELoss(reduction='sum'), loss_file=loss_image, autoencoder=True)