import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

ModelNet40_SHAPENAMES = ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone",
                        "cup", "curtain", "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard",
                        "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant", "radio",
                        "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase",
                        "wardrobe", "xbox"]

def download(data_root):
    print(data_root)
    assert data_root.endswith('modelnet40_ply_hdf5_2048')
    basepath = os.path.dirname(data_root)
    if not os.path.exists(os.path.join(data_root)):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], basepath))
        print('mv %s %s' % (zipfile[:-4], basepath))


def load_data(partition, data_root):
    download(data_root)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_root, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis = 0)
    all_label = np.concatenate(all_label, axis = 0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low = 2. / 3., high = 4. / 3., size = [3])
    xyz2 = np.random.uniform(low = -0.2, high = 0.2, size = [3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma = 0.01, clip = 0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, data_root, num_points = 2048, partition = 'train', aug = True, shuffle = False):
        self.data, self.label = load_data(partition, data_root)
        self.num_points = num_points
        self.partition = partition
        self.aug = aug
        self.shuffle = shuffle

    def __getitem__(self, item):
        """
        :param item:

        :return:
            pointcloud: (N,C)

            label: (1)

        """
        pointcloud = self.data[item]
        if self.shuffle:
            np.random.shuffle(pointcloud)
        pointcloud = pointcloud[:self.num_points]
        label = self.label[item]
        if self.partition == 'train' and self.aug:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def resample(self):
        # don't need resampe operation on the ModelNet40 dataset
        pass




if __name__ == '__main__':
    train = ModelNet40(data_root='../data/modelnet40_ply_hdf5_2048', num_points=1024)
    # test = ModelNet40(1024, 'test')
    import torch
    DataLoader = torch.utils.data.DataLoader(train, batch_size=12, shuffle=True)
    for data, label in DataLoader:
        print(data.dtype)
        print(label.dtype)
        break
