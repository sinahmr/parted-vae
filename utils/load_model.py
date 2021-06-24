import json

import torch

from partedvae.models import VAE


def load(path, img_size, disc_priors, device):
    path_to_specs = path + 'specs.json'
    path_to_model = path + 'model.pt'

    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)
    latent_spec = specs["latent_spec"]

    model = VAE(img_size=img_size, latent_spec=latent_spec, c_priors=disc_priors, device=device)
    model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))

    return model
