import os
import numpy as np
from utils import pc_util
import scipy
from Datasets import ScanObjectNN_DATASET as data_utils
from Datasets import ScanObjectNN_SHAPE_NAMES as SHAPE_NAMES


def save_error_pointcloud(points, DUMP_DIR, lable, pred, start_idx):
    """
    :param points:(B,N,3)
    :param lable:(B,1)
    :param pred:(B,1)
    :return:
    """
    bs_cur = points.shape[0]

    for i in range(bs_cur):
        if lable[i] == pred[i]:
            continue
        l = lable[i]
        p = pred[i]
        idx = start_idx + i
        img_filename = '%d_label_%s_pred_%s.jpg' % (idx, SHAPE_NAMES[l],
                                                    SHAPE_NAMES[p])
        img_filename = os.path.join(DUMP_DIR, img_filename)
        output_img = pc_util.point_cloud_three_views(np.squeeze(points[i]))
        scipy.misc.imsave(img_filename, output_img)

        # save ply
        ply_filename = '%d_label_%s_pred_%s.ply' % (idx, SHAPE_NAMES[l],
                                                    SHAPE_NAMES[p])
        ply_filename = os.path.join(DUMP_DIR, ply_filename)
        data_utils.save_ply(np.squeeze(points[i]), ply_filename)
