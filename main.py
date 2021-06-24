import itertools

import torch
from torch import optim

from partedvae.models import VAE
from partedvae.training import Trainer
from utils.dataloaders import get_dsprites_dataloader, get_mnist_dataloaders, get_celeba_dataloader
from utils.load_model import load
from viz.visualize import Visualizer
from utils.metrics import dis_by_fact_metric

load_model_path = '/path/to/saved/model/'
dataset = 'celeba'

LOAD_MODEL = False
LOAD_DATASET = True
TRAIN = True and LOAD_DATASET
SAVE = True and TRAIN
WARM_UP = True and TRAIN
RECON_TYPE = 'abs' if dataset == 'celeba' else 'bce'  # 'mse' is also possible

epochs = 80

batch_size = 64
lr_warm_up = 5e-4
lr_model = 5e-4

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def save(trainer, z_capacity, u_capacities, latent_spec, epochs, lr_warm_up, lr_model, dataset, recon_type):
    torch.save(trainer.model.state_dict(), 'model.pt')
    with open('specs.json', 'w') as f:
        f.write('''{
        "z_capacity": %s,
        "u_capacity": %s,
        "latent_spec": %s,
        "epochs": %d,
        "lr_warm_up": %f,
        "lr_model": %f,
        "dataset": "%s",
        "recon_type": "%s"
        }''' % (str(z_capacity), str(u_capacities), str(latent_spec).replace("'", '"'), epochs,
                lr_warm_up, lr_model, dataset, recon_type))


if __name__ == '__main__':
    if dataset == 'dsprites':
        disc_priors = [[0.33, 0.33, 0.34]]
        disc_count = len(disc_priors)
        img_size = (1, 64, 64)
        latent_spec = {
            'z': 5,
            'c': [3],
            'single_u': 1,
        }
        z_capacity = [0., 30., 300000, 50.]
        u_capacity = [0., 5., 300000, 50.]
        g_c, g_h = 100., 10.
        g_bc = 10.
        bc_threshold = 0.1
    elif dataset == 'mnist':
        disc_priors = [[0.1] * 10]
        disc_count = len(disc_priors)
        img_size = (1, 32, 32)
        latent_spec = {
            'z': 6,
            'c': [10],
            'single_u': 10
        }
        z_capacity = [0., 7.0, 100000, 15]
        u_capacity = [0., 7.0, 100000, 15]
        g_c, g_h = 15., 30.
        g_bc = 30.
        bc_threshold = 0.15
    else:
        disc_priors = [[0.42, 0.33, 0.18, 0.06], [0.9, 0.07, 0.03], [0.85, 0.15], [0.74, 0.15, 0.11],
                       [0.93, 0.07], [0.47, 0.53], [0.95, 0.05], [0.57, 0.43]]
        disc_count = len(disc_priors)
        img_size = (3, 218, 178)
        latent_spec = {
            'z': 10,
            'c': [4, 3, 2, 3, 2, 2, 2, 2],
            'single_u': 1,
        }
        z_capacity = [0., 30., 125000, 1000.]
        u_capacity = [0., 15., 125000, 1000.]
        g_c, g_h = 2000., 10.
        g_bc = 500.
        bc_threshold = 0.2

    if LOAD_DATASET:
        if dataset == 'dsprites':
            train_loader, warm_up_loader = get_dsprites_dataloader(batch_size=64, fraction=1,
                                                                   path_to_data='../datasets/dsprites/ndarray.npz',
                                                                   device=device, warm_up=WARM_UP)
            test_loader = train_loader
        elif dataset == 'mnist':
            train_loader, test_loader, warm_up_loader = get_mnist_dataloaders(batch_size=64,
                                                                              path_to_data='../datasets/',
                                                                              device=device, warm_up=WARM_UP)
        else:
            train_loader, test_loader, warm_up_loader = get_celeba_dataloader(batch_size=64,
                                                                              path_to_data='../datasets/',
                                                                              device=device, warm_up=WARM_UP)
    if not WARM_UP:
        warm_up_loader = None

    if LOAD_MODEL:
        # Note: When you load a model, capacities are restarted, which isn't intuitive if you are gonna re-train it
        model = load(load_model_path, img_size=img_size, disc_priors=disc_priors, device=device)
        model.sigmoid_coef = 8.
    else:
        model = VAE(img_size=img_size, latent_spec=latent_spec, c_priors=disc_priors, device=device)

    viz = Visualizer(model, root='result/')

    if TRAIN:
        optimizer_warm_up = optim.Adam(itertools.chain(*[
            model.img_to_features.parameters(),
            model.features_to_hidden.parameters(),
            model.h_to_c_logit_fc.parameters()
        ]), lr=lr_warm_up)
        optimizer_model = optim.Adam(model.parameters(), lr=lr_model)
        optimizers = [optimizer_warm_up, optimizer_model]

        trainer = Trainer(model, optimizers, dataset=dataset, device=device, recon_type=RECON_TYPE,
                          z_capacity=z_capacity, u_capacity=u_capacity, c_gamma=g_c, entropy_gamma=g_h,
                          bc_gamma=g_bc, bc_threshold=bc_threshold)
        trainer.train(train_loader, warm_up_loader=warm_up_loader, epochs=epochs, run_after_epoch=None,
                      run_after_epoch_args=[])

        if SAVE:
            save(trainer, z_capacity, u_capacity, latent_spec, epochs, lr_warm_up, lr_model, dataset, RECON_TYPE)

    with torch.no_grad():
        if LOAD_DATASET:
            loader = test_loader if test_loader else train_loader
            for batch, labels in loader:
                break

            viz.reconstructions(batch)
