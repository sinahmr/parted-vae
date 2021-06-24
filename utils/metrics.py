import numpy as np
from sklearn import svm, linear_model
import torch


def all_metrics(dsprites_loader, model):
    latents, classes = get_all_repr(dsprites_loader, model)
    bvae = beta_vae_metric(latents, classes)
    fact = dis_by_fact_metric(latents, classes)
    sap = compute_sap_score(latents, classes)
    return sap, fact, bvae


def get_all_repr(dsprites_loader, model):
    model.eval()
    with torch.no_grad():
        latents = np.zeros((len(dsprites_loader.dataset), model.latent_dim))
        classes = np.zeros((len(dsprites_loader.dataset), 6))
        start = 0
        for i, (x, c) in enumerate(dsprites_loader):
            latent_dist = model.encode(x.to(model.device))
            r = range(start, start + x.size(0))
            latents[r, :] = torch.cat([latent_dist['z'][0], latent_dist['u'][0]], dim=1).cpu().numpy()
            classes[r] = c.cpu().numpy()
            start += x.size(0)
    return latents, classes


# beta-VAE
L = 100
dsprites_classes_num_states = [1, 3, 6, 40, 32, 32]

def get_diffs_and_labels(latents, classes):
    D = latents.shape[1]
    K = classes.shape[1]
    diffs, labels = list(), list()
    for k in range(1, K):  # Ignore 'Color'
        for val in range(dsprites_classes_num_states[k]):
            all_fk = np.where(classes[:, k] == val)[0]
            for i in range(len(all_fk) // (2 * L)):
                r = range(i * (2 * L), (i + 1) * (2 * L))
                current = all_fk[r]
                current_latents = latents[current].reshape(2, -1, D)
                diff = np.abs(current_latents[0, :, :] - current_latents[1, :, :])
                diffs.append(np.mean(diff, axis=0))
                labels.append(np.array([k]))
    diffs = np.stack(diffs, axis=0)
    labels = np.stack(labels, axis=0)
    return diffs, labels


def beta_vae_metric(latents, classes):
    N = latents.shape[0]
    train_z, train_y, test_z, test_y = subsample_train_and_test(latents, classes, int(N * 0.8), int(N * 0.2))
    train_diffs, train_labels = get_diffs_and_labels(train_z, train_y)
    test_diffs, test_labels = get_diffs_and_labels(test_z, test_y)

    model = linear_model.LogisticRegression()
    model.fit(train_diffs, train_labels)
    train_score = model.score(train_diffs, train_labels)
    test_score = model.score(test_diffs, test_labels)

    print(round(train_score, 4), round(test_score, 4), flush=True)
    return train_score, test_score


# FactorVAE
def dis_by_fact_metric(latents, classes):
    D = latents.shape[1]
    K = classes.shape[1]

    stds = np.std(latents, axis=0, keepdims=True)
    normalized_latents = latents / stds

    stats = np.zeros((K, D))
    for k in range(1, K):  # Ignore 'Color'
        for val in range(dsprites_classes_num_states[k]):
            all_fk = np.where(classes[:, k] == val)[0]
            for i in range(len(all_fk) // L):
                r = range(i * L, (i + 1) * L)
                current = all_fk[r]
                vars = np.var(normalized_latents[current], axis=0)
                d_star = np.argmin(vars)
                stats[k, d_star] += 1

    # Prune collapsed latent dimensions
    print(stds)
    effective_stats = np.copy(stats)
    for i in range(D):
        if stds[0, i] < 1e-1:
            effective_stats[:, i] = 0
            with open('latents.npy', 'wb') as f:
                np.save(f, latents)

    # A single latent dimension should only correspond to a single factor of variation, but a factor of variation can relate to multiple latent elements
    score = np.sum(np.max(effective_stats, axis=0)) / np.sum(effective_stats)
    np.set_printoptions(suppress=True)
    print(stats)
    print(score, flush=True)
    return score, stats  # or effective_stats?


# SAP
def subsample_train_and_test(latents, classes, num_train, num_test):
    indices = np.random.choice(np.arange(len(latents)), size=num_train+num_test, replace=False)
    # indices = torch.multinomial(torch.arange(len(latents)), num_train+num_test, replacement=False)
    train_indices, test_indices = indices[:num_train], indices[num_train:]
    return latents[train_indices], classes[train_indices], latents[test_indices], classes[test_indices]


def compute_sap_score(latents, classes):
    train_z, train_y, test_z, test_y = subsample_train_and_test(latents, classes, 20000, 5000)

    matrix = compute_score_matrix(train_z, train_y, test_z, test_y)
    # print(matrix)
    score = compute_avg_diff_top_two(matrix)
    print('SAP:', score, flush=True)
    return score


# From https://github.com/google-research/disentanglement_lib
def compute_score_matrix(mus, ys, mus_test, ys_test):
    D = mus.shape[1]
    K = ys.shape[1]
    score_matrix = np.zeros((D, K))
    for i in range(D):
        for j in range(1, K):
            mu_i, mu_i_test = mus[:, i], mus_test[:, i]
            y_j, y_j_test = ys[:, j], ys_test[:, j]
            classifier = svm.LinearSVC(C=0.01, class_weight='balanced', dual=False)
            classifier.fit(mu_i[:, np.newaxis], y_j)
            pred = classifier.predict(mu_i_test[:, np.newaxis])
            score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    diff = sorted_matrix[-1, :] - sorted_matrix[-2, :]
    return np.mean(diff[1:])  # Ignore 'Color'
