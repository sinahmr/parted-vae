import math

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torchvision.utils import make_grid, save_image


def my_save_image(grid, filename, resize=False):
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if resize:
        im = im.resize((im.size[0] // 3, im.size[1] // 3), Image.ANTIALIAS)
    im.save(filename)


class Visualizer:
    def __init__(self, model, root='result/'):
        self.device = model.device
        self.model = model
        self.root = root

    def reconstructions(self, data, size=(8, 8), filename='recon.png'):
        # Plot reconstructions in test mode, i.e. without sampling from latent
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            recon_data, _ = self.model(data)
        self.model.train()

        # Upper half of plot will contain data, bottom half will contain reconstructions
        num_images = size[0] * size[1] // 2
        originals = data[:num_images].cpu()
        reconstructions = recon_data.view(-1, *self.model.img_size)[:num_images].cpu()
        # If there are fewer examples given than spaces available in grid, augment with blank images
        num_examples = originals.size()[0]
        if num_images > num_examples:
            blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
            originals = torch.cat([originals, blank_images])
            reconstructions = torch.cat([reconstructions, blank_images])
        comparison = torch.cat([originals, reconstructions])

        save_image(comparison.data, self.root + filename, nrow=size[0], pad_value=0.3)
        # return make_grid(comparison.data, nrow=size[0])

    def _traverse_standard_gaussian(self, idx, size, d, sample_prior=False):  # TODO size not used in cdf_traversals
        samples = torch.randn(size, d, device=self.device) if sample_prior else torch.zeros(size, d, device=self.device)

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the inverse CDF (ppf) of
            # a gaussian since the prior of the latent space is gaussian
            # cdf_traversal = np.linspace(0.05, 0.95, size)
            cdf_traversal = np.array([0.001, 0.01, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.99, 0.999])
            cont_traversal = torch.tensor(stats.norm.ppf(cdf_traversal), device=self.device)
            samples[:, idx] += cont_traversal

        return samples

    def _traverse_custom_gaussian(self, idx, size, mean, std):  # sample_prior not implemented   # TODO size not used in cdf_traversals
        samples = mean.unsqueeze(0).repeat(size, 1)

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the inverse CDF (ppf) of
            # a gaussian since the prior of the latent space is gaussian
            # cdf_traversal = np.linspace(0.05, 0.95, size)
            cdf_traversal = np.array([0.001, 0.01, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.99, 0.999])
            cont_traversal = torch.tensor(stats.norm.ppf(cdf_traversal), device=self.device)
            samples[:, idx] += std[idx] * cont_traversal

        return samples

    # C[j] = k | j in [0, C count), k in [0, disc_dim)
    def traverse_with_fix_c(self, j, k, dz_mean, dz_logvar, size=10, path='./', filename_prefix='', filename_suffix='.png', resize=True):
        dz_std = torch.exp(0.5 * dz_logvar)
        rows = list()
        for cont_idx in range(self.model.z_dim):
            line = list()
            line.append(self._traverse_standard_gaussian(cont_idx, size, self.model.z_dim))
            line.append(self._traverse_custom_gaussian(None, size, dz_mean, dz_std))
            rows.append(torch.cat(line, dim=1))
        for dz_idx in range(j * self.model.single_u_dim, (j + 1) * self.model.single_u_dim):
            line = list()
            line.append(self._traverse_standard_gaussian(None, size, self.model.z_dim))
            line.append(self._traverse_custom_gaussian(dz_idx, size, dz_mean, dz_std))
            rows.append(torch.cat(line, dim=1))

        generated = self._decode_latents(torch.cat(rows, dim=0))

        filename = filename_prefix + 'c_' + str(j) + '_' + str(k) + filename_suffix
        grid = make_grid(generated.data, nrow=size, pad_value=0.3)

        # Add a red line to distinguish z from dz
        place = self.model.z_dim * (self.model.img_size[1] + 2)
        grid[0, place:place + 2, 2:-2] = 1

        # Transfer to a new grid
        rows = self.model.z_dim + self.model.single_u_dim
        height = math.ceil(rows / 2) * (self.model.img_size[1] + 2) + 2
        width = grid.size(2) * 2 + 2
        new_grid = torch.zeros(3, height, width, device=self.device)
        new_grid[:, :, :grid.size(2)] = grid[:, :height, :]
        new_grid[1, :, grid.size(2):grid.size(2) + 2] = 1
        new_grid[:, 2:grid.size(1) - height + 2, grid.size(2) + 2:] = grid[:, height:, :]

        my_save_image(new_grid, self.root + path + filename, resize=resize)

    def celeba_all_traversals(self, path='traversals_normal/', bang=None, gender=None, beard=None, hat=None, resize=True):
        with torch.no_grad():
            null = torch.zeros(0, device=self.device)
            dz_base_prior_indices = np.cumsum([0] + self.model.c_dims[:-1])
            for disc_idx, disc_dim in enumerate(self.model.c_dims):
                for i in range(disc_dim):
                    priors_indices = dz_base_prior_indices.copy()
                    priors_indices[disc_idx] += i
                    if disc_idx in [beard, hat]:
                        priors_indices[gender] += 1
                    mean, logvar = self.model.u_prior_means[priors_indices].flatten(), self.model.u_prior_logvars[priors_indices].flatten()
                    self.traverse_with_fix_c(disc_idx, i, mean, logvar, null, null, path=path, resize=resize)
                    if disc_idx == bang:
                        priors_indices[gender] += 1
                        mean, logvar = self.model.u_prior_means[priors_indices].flatten(), self.model.u_prior_logvars[priors_indices].flatten()
                        self.traverse_with_fix_c(disc_idx, i, mean, logvar, null, null, path=path, filename_prefix='male_', resize=resize)

    def traverse_desired_u(self, desired_us, dim, name):
        z = torch.zeros(self.model.z_dim).unsqueeze(0)
        priors_indices = np.cumsum([0] + self.model.c_dims[:-1])
        mean = self.model.u_prior_means[priors_indices].detach().flatten().unsqueeze(0)
        u_list = list()
        for desired_u in desired_us:
            u = mean.clone()
            u[0, dim] = desired_u
            u_list.append(torch.cat([z, u], dim=1))
        generated = self._decode_latents(torch.cat(u_list, dim=0))

        generated = generated[:, :, 20:-20, :]

        grid = make_grid(generated.data, nrow=len(desired_us), pad_value=0)
        my_save_image(grid, self.root + '%s.png' % name, resize=False)

    def transform(self, images):
        intermediary = 8
        N, image_shape = images.size()[0], images.size()[1:]
        assert N % 2 == 0
        all_images = torch.zeros(N // 2, intermediary + 2, *image_shape)
        all_images[:, 0], all_images[:, -1] = images[:N//2], images[N//2:]

        latent_dist = self.model.encode(images)
        z = latent_dist['z'][0].view(2, N // 2, -1).unsqueeze(2)
        u = latent_dist['u'][0].view(2, N // 2, -1).unsqueeze(2)
        coefs = torch.linspace(0, 1, intermediary).view(1, -1, 1)
        z_interpolation = z[0] + (z[1] - z[0]) * coefs
        u_interpolation = u[0] + (u[1] - u[0]) * coefs
        latents = torch.cat([z_interpolation, u_interpolation], dim=2)
        all_images[:, 1:-1] = self._decode_latents(latents).view(N//2, intermediary, *image_shape)
        grid = make_grid(all_images.view(-1, *image_shape), nrow=intermediary+2, pad_value=0)
        my_save_image(grid, self.root + 'transform.png', resize=False)

    def swap(self, images):
        count = len(images)
        image_shape = images.size()[1:]
        latent_dist = self.model.encode(images)
        z = latent_dist['z'][0]
        u = latent_dist['u'][0]
        latents = list()
        for i in range(count):
            for j in range(count):
                latents.append(torch.cat([z[j], u[i]]).unsqueeze(0))
        all_images = torch.zeros(count+1, count+1, *image_shape)
        all_images[0, 1:] = images
        all_images[1:, 0] = images
        # all_images[torch.arange(count+1, (count+1)*(count+1), count+1)] = images
        all_images[1:, 1:] = self._decode_latents(torch.cat(latents, dim=0)).view(count, count, *image_shape)
        grid = make_grid(all_images.view(-1, *image_shape), nrow=count+1, pad_value=0)
        my_save_image(grid, self.root + 'swap.png', resize=False)

    def z_traversal(self):
        intermediary = 10
        linespace = torch.linspace(-3, 3, intermediary)
        priors_indices = np.cumsum([0] + self.model.c_dims[:-1])
        u = self.model.u_prior_means[priors_indices].detach().flatten().unsqueeze(0)
        latents = list()

        for i in [2, 3, 4, 6, 8, 9]:
            z = torch.zeros(intermediary, self.model.z_dim)
            z[:, i] = linespace
            latents.append(torch.cat([z, u.expand(intermediary, -1)], dim=1))
        generated = self._decode_latents(torch.cat(latents, dim=0))

        generated = generated[:, :, 20:-20, :]

        grid = make_grid(generated, nrow=intermediary, pad_value=0)
        my_save_image(grid, self.root + 'z.png', resize=False)

    def celeba_u_traversal(self):  # 2 sigma for u, 0.1 * z
        size = 10
        with torch.no_grad():
            base_prior_indices = np.cumsum([0] + self.model.c_dims[:-1])
            default_female = self.model.u_prior_means[base_prior_indices].flatten()
            base_prior_indices[-1] += 1
            default_male = self.model.u_prior_means[base_prior_indices].flatten()
            pairs = [[3, 9, 10, False], [3, 9, 11, True], [4, 12, 13, False], [5, 14, 15, False], [6, 16, 17, True], [7, 18, 19, False]]
            rows = list()
            for dim, left, right, male in pairs:
                z = torch.randn(1, self.model.z_dim).expand(size, -1) * 0.5
                u = (default_male.clone() if male else default_female.clone()).unsqueeze(0).repeat(size, 1)
                left_mu, left_std = self.model.u_prior_means[left, 0], torch.exp(0.5 * self.model.u_prior_logvars[left, 0])
                right_mu, right_std = self.model.u_prior_means[right, 0], torch.exp(0.5 * self.model.u_prior_logvars[right, 0])
                minn = min(left_mu - 2*left_std, right_mu - 2*right_std)
                maxx = max(left_mu + 2*left_std, right_mu + 2*right_std)
                u[:, dim] = torch.linspace(minn, maxx, size)
                rows.append(torch.cat([z, u], dim=1))
            generated = self._decode_latents(torch.cat(rows, dim=0))

            generated = generated[:, :, 20:-20, :]

            grid = make_grid(generated, nrow=size, pad_value=0)
            my_save_image(grid, self.root + 'u.png', resize=False)

    def no_bc_celeba_u_traversal(self):  # 2 sigma for u, 0.1 * z
        size = 10
        with torch.no_grad():
            base_prior_indices = np.cumsum([0] + self.model.c_dims[:-1])
            default_female = self.model.u_prior_means[base_prior_indices].flatten()
            base_prior_indices[-1] += 1
            default_male = self.model.u_prior_means[base_prior_indices].flatten()
            pairs = [[1, 4, 6, True], [2, 7, 8, True], [4, 12, 13, False], [6, 16, 17, True]]
            rows = list()
            for dim, left, right, male in pairs:
                z = torch.randn(1, self.model.z_dim).expand(size, -1) * 0.5
                u = (default_male.clone() if male else default_female.clone()).unsqueeze(0).repeat(size, 1)
                left_mu, left_std = self.model.u_prior_means[left, 0], torch.exp(0.5 * self.model.u_prior_logvars[left, 0])
                right_mu, right_std = self.model.u_prior_means[right, 0], torch.exp(0.5 * self.model.u_prior_logvars[right, 0])
                minn = min(left_mu - 2*left_std, right_mu - 2*right_std)
                maxx = max(left_mu + 2*left_std, right_mu + 2*right_std)
                u[:, dim] = torch.linspace(minn, maxx, size)
                rows.append(torch.cat([z, u], dim=1))
            generated = self._decode_latents(torch.cat(rows, dim=0))

            generated = generated[:, :, 20:-20, :]

            grid = make_grid(generated, nrow=size, pad_value=0)
            my_save_image(grid, self.root + 'u.png', resize=False)

    def all_traversals(self, path='traversals_normal/'):
        with torch.no_grad():
            null = torch.zeros(0, device=self.device)
            dz_base_prior_indices = np.cumsum([0] + self.model.c_dims[:-1])
            for disc_idx, disc_dim in enumerate(self.model.c_dims):
                for i in range(disc_dim):
                    priors_indices = dz_base_prior_indices.copy()
                    priors_indices[disc_idx] += i
                    mean, logvar = self.model.u_prior_means[priors_indices].flatten(), self.model.u_prior_logvars[priors_indices].flatten()
                    self.traverse_with_fix_c(disc_idx, i, mean, logvar, null, null, path=path, resize=False)

    def _decode_latents(self, latent_samples):
        with torch.no_grad():
            return self.model.decode(latent_samples).cpu()
