# PRA-Net
The source code of PRA-Net.

#### Requirements

- CUDA >= 10
- python = 3.7
- pytorch = 1.2.0
- numpy = 1.16.4

#### DataSet

- Classification
  - Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), and unzip it to `data/modelnet40_ply_hdf5_2048`
  - Download dataset from [ScanObjectNN Projcet](https://hkust-vgd.github.io/scanobjectnn/), and unzip it to 
    `data/h5_files`


#### Installation

- Before running the training script, you need to compile some customized torch operators.

  ~~~
  cd _ext-src
  python setup.py install
  ~~~

### Training

##### Classification

- To train a PR-Net model on the ScanObjectNN dataset:

  ~~~
  CUDA_VISIBLE_DEVICES=0 python train_cls.py --config cfg/ScanObjectNN_train_PRANet.yaml
  ~~~



Our code is mainly based on [DGCNN](https://github.com/WangYueFt/dgcnn), [ScanobjectNN](https://github.com/hkust-vgd/scanobjectnn)
and [RSCNN](https://github.com/Yochengliu/Relation-Shape-CNN), thanks for them! 
