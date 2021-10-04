import os
import sys

import numpy as np

import scipy.misc
import string
import pickle
import plyfile
import h5py
import torch
from torch.utils.data import Dataset
from utils.mapping2 import OBJECTDATASET_TO_MODELNET
from utils import provider

ScanObjectNN_SHAPE_NAMES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                            "pillow",
                            "sink", "sofa", "toilet"]


def save_ply(points, filename, colors = None, normals = None):
    vertex = np.core.records.fromarrays(points.transpose(), names = 'x, y, z', formats = 'f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(), names = 'nx, ny, nz', formats = 'f4, f4, f4')
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names = 'red, green, blue',
                                                  formats = 'u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype = desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text = False)
    # if not os.path.exists(os.path.dirname(filename)):
    #    os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def shuffle_points(pcs):
    for pc in pcs:
        np.random.shuffle(pc)
    return pcs


def get_current_data(pcs, labels, num_points):
    sampled = []
    for pc in pcs:
        if (pc.shape[0] < num_points):
            # TODO repeat points
            print("Points too less.")
            return
        else:
            # faster than shuffle_points
            idx = np.arange(pc.shape[0])
            np.random.shuffle(idx)
            sampled.append(pc[idx[:num_points], :])

    sampled = np.array(sampled)
    labels = np.array(labels)

    # shuffle per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    labels = labels[idx]

    return sampled, labels


def normalize_data(pcs):
    for pc in pcs:
        # get furthest point distance then normalize
        d = max(np.sum(np.abs(pc) ** 2, axis = -1) ** (1. / 2))
        pc /= d

    # pc[:,0]/=max(abs(pc[:,0]))
    # pc[:,1]/=max(abs(pc[:,1]))
    # pc[:,2]/=max(abs(pc[:,2]))

    return pcs


def normalize_data_multiview(pcs, num_view = 5):
    pcs_norm = []
    for i in range(len(pcs)):
        pc = []
        for j in range(num_view):
            pc_view = pcs[i][j, :, :]
            d = max(np.sum(np.abs(pc_view) ** 2, axis = -1) ** (1. / 2))
            pc.append(pc_view / d)
        pc = np.array(pc)
        pcs_norm.append(pc)
    pcs_norm = np.array(pcs_norm)
    print("Normalized")
    print(pcs_norm.shape)
    return pcs_norm


# USE For SUNCG, to center to origin
def center_data(pcs):
    for pc in pcs:
        centroid = np.mean(pc, axis = 0)
        pc[:, 0] -= centroid[0]
        pc[:, 1] -= centroid[1]
        pc[:, 2] -= centroid[2]
    return pcs


##For h5 files
def get_current_data_h5(pcs, labels, num_points, shuffle = True):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]
    # sampled = pcs[:,:num_points,:]

    # shuffle point clouds per epoch
    if shuffle:
        idx = np.arange(len(labels))
        np.random.shuffle(idx)

        sampled = sampled[idx]
        labels = labels[idx]

    return sampled, labels


def get_current_data_withmask_h5(pcs, labels, masks, num_points, shuffle = True):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])

    if (shuffle):
        # print("Shuffled points: "+str(shuffle))
        np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]
    sampled_mask = masks[:, idx_pts[:num_points]]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))

    ##Shuffle order of the inputs
    if (shuffle):
        np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_mask = sampled_mask[idx]
    labels = labels[idx]

    return sampled, labels, sampled_mask


def get_current_data_parts_h5(pcs, labels, parts, num_points):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]

    sampled_parts = parts[:, idx_pts[:num_points]]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_parts = sampled_parts[idx]
    labels = labels[idx]

    return sampled, labels, sampled_parts


def get_current_data_discriminator_h5(pcs, labels, types, num_points):
    # shuffle points to sample
    idx_pts = np.arange(pcs.shape[1])
    np.random.shuffle(idx_pts)

    sampled = pcs[:, idx_pts[:num_points], :]

    # shuffle point clouds per epoch
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    sampled = sampled[idx]
    sampled_types = types[idx]
    labels = labels[idx]

    return sampled, labels, sampled_types


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label


def load_withmask_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mask = f['mask'][:]

    return data, label, mask


def load_discriminator_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    model_type = f['type'][:]

    return data, label, model_type


def load_parts_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    parts = f['parts'][:]

    return data, label, parts


def convert_to_binary_mask(masks):
    binary_masks = []
    for i in range(masks.shape[0]):
        binary_mask = np.ones(masks[i].shape)
        bg_idx = np.where(masks[i, :] == -1)
        binary_mask[bg_idx] = 0

        binary_masks.append(binary_mask)

    binary_masks = np.array(binary_masks)
    return binary_masks


class ScanObject(Dataset):

    def __init__(self, h5file_path = 'data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5',
                 center = True,
                 norm = True, with_bg = True, rotation = False, jit = False, num_point = 1024, use_mask = False,
                 shuffle = True):
        super(ScanObject, self).__init__()
        self.h5file_path = h5file_path
        self.use_mask = use_mask
        self.shuffle = shuffle
        if not self.use_mask:
            self.total_DATA, self.total_LABELS = load_h5(self.h5file_path)
            print(self.total_DATA.shape)
        else:
            self.total_DATA, self.total_LABELS, self.total_MASK = load_withmask_h5(self.h5file_path)
            self.total_MASK = convert_to_binary_mask(self.total_MASK)

        self.rotation = rotation
        self.jit = jit

        if center:
            self.total_DATA = center_data(self.total_DATA)
        if norm:
            self.total_DATA = normalize_data(self.total_DATA)
        self.num_point = num_point
        self.resample()
        self.num_len = self.LABEL.shape[0]

    def resample(self):
        # print('re-sample points')
        if self.use_mask:
            self.DATA, self.LABEL, self.MASK = get_current_data_withmask_h5(self.total_DATA, self.total_LABELS,
                                                                            self.total_MASK, self.num_point)
        else:
            self.DATA, self.LABEL = get_current_data_h5(self.total_DATA, self.total_LABELS, self.num_point,
                                                        self.shuffle)

    def __getitem__(self, index: int):
        """
        :param index:

        :return:
             point: (N,C)

            label: (N)

            mask: None or (N)
            The background mask, not used by default.
        """
        point, label = self.DATA[index], self.LABEL[index]

        # Data augmentaion
        if self.rotation:
            point = provider.rotate_point_cloud(point)
        if self.jit:
            point = provider.jitter_point_cloud(point)

        point = torch.tensor(point).float()
        label = torch.tensor(label).long()

        if self.use_mask:
            mask = self.MASK[index]
            mask = torch.tensor(mask).long()
            return point, label, mask
        else:
            return point, label

    def __len__(self):
        return self.num_len


class ScanObjectFilter(ScanObject):

    def __init__(self, h5file_path = 'h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5',
                 center = True,
                 norm = True, with_bg = True, rotation = False, jit = False, num_point = 1024, use_mask = False):

        self.file_path = h5file_path
        self.use_mask = use_mask

        # Total data and labels
        self.total_DATA, self.total_LABELS = load_h5(self.file_path)

        self.rotation = rotation
        self.jit = jit

        print(self.total_DATA.shape)
        print(self.total_LABELS.shape)

        filtered_data = []
        filtered_label = []
        for i in range(self.total_LABELS.shape[0]):
            if (self.total_LABELS[i] in OBJECTDATASET_TO_MODELNET.keys()):
                filtered_label.append(self.total_LABELS[i])
                filtered_data.append(self.total_DATA[i, :])
        # Filtered data
        filtered_data = np.array(filtered_data)
        filtered_label = np.array(filtered_label)
        self.total_DATA = filtered_data
        self.total_LABELS = filtered_label
        print(filtered_data.shape)
        print(filtered_label.shape)

        if center:
            self.total_DATA = center_data(self.total_DATA)
        if norm:
            self.total_DATA = normalize_data(self.total_DATA)
        self.num_point = num_point
        self.resample()
        self.num_len = self.LABEL.shape[0]


if __name__ == '__main__':

    scannet = ScanObject()
    for i, (point, label) in enumerate(scannet):
        print(i, point.shape)
        print(label.shape)
