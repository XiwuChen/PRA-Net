from __future__ import print_function

import argparse
import random
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

from Datasets.ModelNet40_DATASET import ModelNet40
from Datasets.ModelNetDataLoader_withnorm import ModelNetDataLoader
from Datasets.ScanObjectNN_DATASET import ScanObject
from models.PRANet_classification import PRANet_classification
from utils.util import IOStream, cal_loss

torch.backends.cudnn.enabled = False


#
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def _copy_source_snapshot(config_path, destination):
    """Copy source files used by an experiment into its checkpoint folder."""
    source_patterns = ("*.py", "models/*.py", "utils/*.py")
    for pattern in source_patterns:
        for source_path in Path(".").glob(pattern):
            shutil.copy2(str(source_path), str(destination / source_path.name))
    shutil.copy2(str(config_path), str(destination / config_path.name))


def _setup_experiment(args):
    """Seed random generators and prepare the experiment output folders."""
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    experiment_dir = Path("checkpoints") / args.exp_name
    (experiment_dir / "models").mkdir(parents=True, exist_ok=True)
    _copy_source_snapshot(Path(args.config), experiment_dir)
    return experiment_dir


def test(epoch, model, test_dataset, test_loader, device, io, test_num=1):
    """Evaluate a model over one or more random point resamplings."""
    oa_list = []
    mAcc_list = []
    model.eval()
    for _ in range(test_num):
        test_pred = []
        test_true = []
        test_dataset.resample()
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            with torch.no_grad():
                logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                  test_acc,
                                                                  avg_per_class_acc)
        io.cprint(outstr)
        oa_list.append(test_acc)
        mAcc_list.append(avg_per_class_acc)

    return oa_list, mAcc_list


def train(args, io):
    # Load models
    device = torch.device("cuda")
    if args.model == 'PRANet':
        model_name = PRANet_classification
    else:
        raise Exception("Not implemented")

    if args.dataset == 'ModelNet40Norm':
        model_name = partial(model_name, input_channel=6)
    model = model_name(args, output_channels=args.num_classes).to(device)

    io.cprint(time.asctime())

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Load dataset
    supported_datasets = ['ScanObjectNN', 'ModelNet40', 'ModelNet40Norm']
    if args.dataset not in supported_datasets:
        raise ValueError("Unsupported dataset: %s" % args.dataset)

    if args.dataset == 'ScanObjectNN':
        data_root = Path(args.data_root) / "main_split"
        train_dataset = ScanObject(
            h5file_path=str(data_root / "training_objectdataset_augmentedrot_scale75.h5"),
            center=True, norm=True, with_bg=True, rotation=True, jit=True,
            num_point=args.num_points)

        test_dataset = ScanObject(
            h5file_path=str(data_root / "test_objectdataset_augmentedrot_scale75.h5"),
            center=True, norm=True, with_bg=True, rotation=False, jit=False,
            num_point=args.num_points)

    elif args.dataset == 'ModelNet40':
        train_dataset = ModelNet40(data_root=args.data_root, partition='train', num_points=args.num_points,
                                   aug=True)

        test_dataset = ModelNet40(data_root=args.data_root, partition='test', num_points=args.num_points,
                                  aug=False)
    elif args.dataset == 'ModelNet40Norm':

        train_dataset = ModelNetDataLoader(args.data_root, npoint=args.num_points, split='train', uniform=args.uniform,
                                           normal_channel=True, cache_size=15000, augment=args.augment,
                                           random_drop=args.random_drop)
        test_dataset = ModelNetDataLoader(args.data_root, npoint=args.num_points, split='test', uniform=args.uniform,
                                          normal_channel=True, cache_size=15000, augment=False,
                                          random_drop=args.random_drop)
    train_loader = DataLoader(train_dataset, num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.workers,
                             batch_size=(args.batch_size // 2), shuffle=False, drop_last=False)

    io.cprint(str(model))
    io.cprint('*********************************************\nmodel_name: %s' % args.model)

    print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=0.001)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        train_dataset.resample()

        for points, label in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
            # (B, N, C)
            points, label = points.to(device), label.to(device).squeeze()

            batch_size = points.size()[0]
            points = points.permute(0, 2, 1).contiguous()

            opt.zero_grad()
            # (B, C, N) -> (B, C2)
            logits = model(points)
            loss = cal_loss(logits, label, args.smoothing)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        scheduler.step()
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################

        if epoch < 10 or epoch > 200:
            oa_list, mAcc_list = test(
                epoch, model, test_dataset, test_loader, device, io, test_num=1
            )
            io.cprint('Test %d, OA std: %.6f, mean: %.6f' % (epoch, np.std(oa_list), np.average(oa_list)))
            io.cprint('Test %d, mAcc std: %.6f , mean: %.6f' % (epoch, np.std(mAcc_list), np.average(mAcc_list)))
            torch.save(model.state_dict(),
                       'checkpoints/%s/models/model_%0.6f_epoch%3d.t7' % (args.exp_name, np.average(oa_list), epoch))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PRA-Net Classification Training')
    parser.add_argument(
        '--config',
        default='cfg/ScanObjectNN_train_PRANet.yaml',
        type=str,
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:' % (k), v)
    print("\n**************************\n")

    experiment_dir = _setup_experiment(args)
    print('seed', args.seed)

    io = IOStream(str(experiment_dir / "run.log"))
    try:
        io.cprint(str(args))
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                torch.cuda.device_count()) + ' devices')
        train(args, io)
    finally:
        io.close()
