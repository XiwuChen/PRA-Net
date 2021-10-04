
import torch
import torch.nn.functional as F
from utils.pointnet2_utils import furthest_point_sample_cuda
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def knn(x, k):
    """
    :param x: (B,3,N)
    :param k: int
    :return: (B,N,k_hat)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim = 1, keepdim = True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k = k, dim = -1)[1]  # (batch_size, num_points, k_hat)
    return idx


def knn_withdis(x, k):
    """
    :param x: (B,3,N)
    :param k: int
    :return: (B,N,k_hat)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim = 1, keepdim = True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k = k, dim = -1)[1]  # (batch_size, num_points, k_hat)
    return idx, pairwise_distance


def knn_2(x, k):
    """
    :param x: (N,3)
    :param k: int
    :return: (N,k_hat)
    """
    inner = -2 * torch.matmul(x, x.transpose(1, 0))
    xx = torch.sum(x ** 2, dim = 0, keepdim = True)
    pairwise_distance = -xx - inner - xx.transpose(1, 0)

    idx = pairwise_distance.topk(k = k, dim = -1)[1]  # (num_points, k_hat)
    return idx



def farthest_point_sample(xyz, npoint):
    return furthest_point_sample_cuda(xyz, npoint)


def farthest_point_sample_python(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype = torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype = torch.long).to(device)
    batch_indices = torch.arange(B, dtype = torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype = torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


''' Calculate cross entropy loss, apply label smoothing if needed. '''


def cal_loss(pred, gold, smoothing = True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim = 1)

        loss = -(one_hot * log_prb).sum(dim = 1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction = 'mean')

    return loss


def plot_embedding(X, y, class_name, pplex, random_seed, title = None):
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components = 2, init = 'pca', perplexity = pplex, random_state = random_seed)
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    from collections import defaultdict
    digits = defaultdict(list)
    for i in range(X.shape[0]):
        digits[y[i]].append(np.expand_dims(X[i], axis = 0))
    for i in digits.keys():
        print(i)
        digits[i] = np.concatenate(digits[i])

    fig, ax = plt.subplots()
    for i in sorted(digits.keys()):
        X = digits[i]
        ax.scatter(X[:, 0], X[:, 1], s = 0.7, color = plt.cm.Set1(i / 40.), label = class_name[i])

    ax.legend()
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def fileprint(self, text):
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


