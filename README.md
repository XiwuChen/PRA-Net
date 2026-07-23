# IEEE TIP 2021 | PRA-Net

[IEEE TIP](https://doi.org/10.1109/TIP.2021.3072214)
[arXiv](https://arxiv.org/abs/2112.04903)

PyTorch implementation of PRA-Net for point cloud classification on
ScanObjectNN and ModelNet40.

## Project structure

```text
.
├── cfg/
│   └── ScanObjectNN_train_PRANet.yaml   # ScanObjectNN training config
├── Datasets/
│   ├── ScanObjectNN_DATASET.py          # ScanObjectNN loader
│   ├── ModelNet40_DATASET.py            # ModelNet40 HDF5 loader
│   └── ModelNetDataLoader_withnorm.py   # ModelNet40 XYZ+normal loader
├── models/
│   ├── PRANet_classification.py         # Classification network
│   ├── intra_structure.py               # ISL module
│   ├── inter_region.py                  # IRL module
│   ├── dynamic_region_partition.py      # Region partition
│   └── dynamic_feature_aggregation.py   # Feature aggregation
├── utils/
│   ├── pointnet2_utils.py               # CUDA operator wrappers
│   └── util.py                          # Loss, sampling, and logging
├── _ext-src/                            # Custom CUDA operators
├── train_cls.py                         # Training entry point
└── README.md
```

## Environment and installation

The original code was developed with:

- Python 3.7
- PyTorch 1.2.0
- CUDA 10.0 or later
- NumPy 1.16.4
- An NVIDIA GPU

Linux is recommended because the training and dataset download code uses Unix
commands.

### 1. Create an environment

```bash
conda create -n pra-net python=3.7 -y
conda activate pra-net
```

Install a CUDA-compatible PyTorch 1.2.0 build by following
[PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/),
then install the remaining dependencies:

```bash
pip install numpy==1.16.4 h5py scipy scikit-learn tqdm "PyYAML<6" \
    matplotlib plyfile Pillow
```

`PyYAML<6` is recommended because `train_cls.py` uses the legacy
`yaml.load` API.

### 2. Build the CUDA extension

```bash
cd _ext-src
python setup.py install
cd ..
```

Verify the installation:

```bash
python -c "import fps; print('fps extension is available')"
```

The current implementation requires CUDA and does not support CPU-only
training.

## Training

Always run commands from the repository root and pass `--config` explicitly.
The default config path in `train_cls.py` does not exist in this repository.

### ScanObjectNN

Download the processed HDF5 files from the
[ScanObjectNN project](https://hkust-vgd.github.io/scanobjectnn/) and arrange
the PB_T50_RS main split as follows:

```text
data/
└── h5_files/
    └── main_split/
        ├── training_objectdataset_augmentedrot_scale75.h5
        └── test_objectdataset_augmentedrot_scale75.h5
```

The supplied configuration uses 1,024 points and 15 classes. Start training
with:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cls.py \
    --config cfg/ScanObjectNN_train_PRANet.yaml
```



### ModelNet40

The repository supports two ModelNet40 formats.

#### HDF5 format

Download
`[modelnet40_ply_hdf5_2048.zip](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)`
and extract it to:

```text
data/
└── modelnet40_ply_hdf5_2048/
    ├── ply_data_train0.h5
    ├── ply_data_train1.h5
    ├── ...
    ├── ply_data_test0.h5
    └── ...
```

Create `cfg/ModelNet40_train_PRANet.yaml`:

```yaml
common:
  exp_name: ModelNet40_PRANet
  init: True
  model: PRANet
  seed: 1
  dataset: ModelNet40

  smoothing: True
  epochs: 250

  k_hat: 20
  m_list: [0, 4, 4, 4]
  sample_ratio_list: [0, 4, 8, 16]
  start_layer: 1

  dropout: 0.5
  emb_dims: 1024

  workers: 4
  num_points: 1024
  num_classes: 40
  batch_size: 32

  lr: 0.1
  weight_decay: 0.0001
  momentum: 0.9

  checkpoint:
  data_root: "./data/modelnet40_ply_hdf5_2048"
```

Start training:

```bash
CUDA_VISIBLE_DEVICES=0 python train_cls.py \
    --config cfg/ModelNet40_train_PRANet.yaml
```



#### XYZ and normal format

The `ModelNet40Norm` loader expects:

```text
data/
└── modelnet40_normal_resampled/
    ├── modelnet40_shape_names.txt
    ├── modelnet40_train.txt
    ├── modelnet40_test.txt
    ├── airplane/
    │   ├── airplane_0001.txt
    │   └── ...
    └── ...
```

Each shape file must contain comma-separated XYZ coordinates and normals:

```text
x, y, z, nx, ny, nz
```

Use the ModelNet40 configuration above with these changes:

```yaml
  dataset: ModelNet40Norm
  data_root: "./data/modelnet40_normal_resampled"
  num_classes: 40
  uniform: False
  augment: True
  random_drop: True
```



### Outputs

Logs and checkpoints are written to:

```text
checkpoints/<exp_name>/
├── run.log
└── models/
    └── model_<OA>_epoch<epoch>.t7
```

To initialize the model from an existing checkpoint, set its path in the YAML:

```yaml
checkpoint: "checkpoints/<exp_name>/models/model_xxxxxx_epochxxx.t7"
```

Only model weights are restored; optimizer and scheduler states are not
restored.

## FAQ



### 1. `No module named 'fps'`

Activate the correct Python environment, rebuild the CUDA extension from
`_ext-src`, and verify it with:

```bash
python -c "import fps"
```



### 2. `load() missing 1 required positional argument: 'Loader'`

Install `PyYAML<6`, or replace `yaml.load(f)` with `yaml.safe_load(f)` in the
training script.

### 3. ScanObjectNN reports `FileNotFoundError`

Confirm that the PB_T50_RS files are located under
`data/h5_files/main_split/` and that `data_root` is set to
`./data/h5_files`.

### 4. ModelNet40 data cannot be found

For `dataset: ModelNet40`, `data_root` must directly contain files matching
`ply_data_train*.h5` and `ply_data_test*.h5`.

For `dataset: ModelNet40Norm`, `data_root` must contain
`modelnet40_shape_names.txt`, the train/test split files, and the category
directories.

### 5. CUDA out of memory

Reduce `batch_size` in the YAML configuration. If `num_points` is also
reduced, ensure that it remains larger than `k_hat` and is compatible with the
IRL sampling ratios.

## Citation

If this repository is useful to your research, please cite PRA-Net:

```bibtex
@article{cheng2021pranet,
  title   = {PRA-Net: Point Relation-Aware Network for 3D Point Cloud Analysis},
  author  = {Cheng, Silin and Chen, Xiwu and He, Xinwei and Liu, Zhe and Bai, Xiang},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30},
  pages   = {4436--4448},
  year    = {2021},
  doi     = {10.1109/TIP.2021.3072214}
}
```

## Acknowledgements

This implementation builds upon the following projects:

- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [ScanObjectNN](https://github.com/hkust-vgd/scanobjectnn)
- [Relation-Shape CNN (RSCNN)](https://github.com/Yochengliu/Relation-Shape-CNN)
