import cv2
import utils
import numpy as np
from typing import Optional
from dataset import RandomFlip
from dataset import Dataset
from torch.utils.data import DataLoader
import pickle
import os
from PIL import Image
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as mds
from models_tf_2_3loss_ms import SegDecNet_tf
import mindspore.dataset.transforms as transforms
import argparse
from config import Config
from c2net.context import prepare
# 初始化导入数据集和预训练模型到容器内
c2net_context = prepare()
# 获取数据集路径
kolektorsdd2_path = c2net_context.dataset_path + "/" + "KolektorSDD2"
# 获取预训练模型路径
_3_loss_model_ch5s_path = c2net_context.pretrain_model_path + "/" + "3_loss_model_ch5s"
# 输出结果必须保存在该目录
you_should_save_here = c2net_context.output_path

LVL_INFO = 5
LOG = 1


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def init(net):
    param_dict = mindspore.load_checkpoint(_3_loss_model_ch5s_path + '/best_state_dict.ckpt')
    mindspore.load_param_into_net(net, param_dict)
    return net


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=str, required=False, default='Ascend', choices=['Ascend', 'CPU'], help="ID of GPU used for training/evaluation.")
    parser.add_argument('--RUN_NAME', type=str, required=False, default="train_KSDD2",
                        help="Name of the run, used as directory name for storing results.")
    parser.add_argument('--DATASET', type=str, required=False, default="KSDD2", help="Which dataset to use.")
    parser.add_argument('--DATASET_PATH', type=str, required=False, default=kolektorsdd2_path,
                        help="Path to the dataset.")

    parser.add_argument('--EPOCHS', type=int, required=False, default=100, help="Number of training epochs.")

    parser.add_argument('--LEARNING_RATE', type=float, required=False, default=0.01, help="Learning rate.")
    parser.add_argument('--DELTA_CLS_LOSS', type=float, required=False, default=1,
                        help="Weight delta for classification loss.")

    parser.add_argument('--BATCH_SIZE', type=int, required=False, default=1, help="Batch size for training.")

    parser.add_argument('--WEIGHTED_SEG_LOSS', type=str2bool, required=False, default=True,
                        help="Whether to use weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_P', type=float, required=False, default=2,
                        help="Degree of polynomial for weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_MAX', type=float, required=False, default=3,
                        help="Scaling factor for weighted segmentation loss.")
    parser.add_argument('--DYN_BALANCED_LOSS', type=str2bool, required=False, default=True,
                        help="Whether to use dynamically balanced loss.")
    parser.add_argument('--GRADIENT_ADJUSTMENT', type=str2bool, required=False, default=True,
                        help="Whether to use gradient adjustment.")
    parser.add_argument('--FREQUENCY_SAMPLING', type=str2bool, required=False, default=True,
                        help="Whether to use frequency-of-use based sampling.")

    parser.add_argument('--DILATE', type=int, required=False, default=45, help="Size of dilation kernel for labels")

    parser.add_argument('--FOLD', type=int, default=0, help="Which fold (KSDD) or class (DAGM) to train.")
    parser.add_argument('--TRAIN_NUM', type=int, required=False, default=-1,
                        help="Number of positive training samples for KSDD or STEEL.")
    parser.add_argument('--NUM_SEGMENTED', type=int, required=False, default=246,
                        help="Number of segmented positive  samples.")
    parser.add_argument('--RESULTS_PATH', type=str, default="./results/", help="Directory to which results are saved.")

    parser.add_argument('--VALIDATE', type=str2bool, default=True, help="Whether to validate during training.")
    parser.add_argument('--VALIDATE_ON_TEST', type=str2bool, default=True, help="Whether to validate on test set.")
    parser.add_argument('--VALIDATION_N_EPOCHS', type=int, default=None,
                        help="Number of epochs between consecutive validation runs.")
    parser.add_argument('--USE_BEST_MODEL', type=str2bool, default=True,
                        help="Whether to use the best model according to validation metrics for evaluation.")

    parser.add_argument('--ON_DEMAND_READ', type=str2bool, default=None,
                        help="Whether to use on-demand read of data from disk instead of storing it in memory.")
    parser.add_argument('--REPRODUCIBLE_RUN', type=str2bool, default=None,
                        help="Whether to fix seeds and disable CUDA benchmark mode.")

    parser.add_argument('--MEMORY_FIT', type=int, default=None, help="How many images can be fitted in GPU memory.")
    parser.add_argument('--SAVE_IMAGES', type=str2bool, default=None, help="Save test images or not.")

    args = parser.parse_args()

    return args


def read_split(num_segmented: int, kind: str):
    fn = f"KSDD2/split_{num_segmented}.pyb"
    # with open(f"splits/{fn}", "rb") as f:
    with open(f"{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        else:
            raise Exception('Unknown')


class KSDD2Dataset(Dataset):
    def __init__(self, kind: str, cfg: Config, transform):
        super(KSDD2Dataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.transform = transform
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)

        for part, is_segmented in data_points:
            is_segmented = True
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
            seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_GT.png")

            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            seg_mask_inside, positive = self.read_label_resize(seg_mask_path, resize_dim=self.image_size, dilate=0)
            seg_mask, positive = self.read_label_resize(seg_mask_path, resize_dim=self.image_size,
                                                        dilate=self.cfg.DILATE)

            if positive:
                # image = self.to_tensor(image)
                # seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.distance_transform_inside(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX,
                                                               self.cfg.WEIGHTED_SEG_LOSS_P, seg_mask_inside)
                seg_loss_mask = self.downsize(seg_loss_mask)
                seg_mask = self.downsize(seg_mask)
                # if self.kind in ['TRAIN']:
                #     sample = [image,seg_mask,seg_loss_mask]
                #     image,seg_mask,seg_loss_mask = self.transform(sample)
                pos_samples.append((self.to_tensor(image), self.to_tensor(seg_mask), self.to_tensor(seg_loss_mask),
                                    is_segmented, image_path, seg_mask_path, part))
            else:
                # image = self.to_tensor(image)
                seg_loss_mask = self.downsize(np.ones_like(seg_mask))
                seg_mask = self.downsize(seg_mask)
                # if self.kind in ['TRAIN']:
                #     sample = [image,seg_mask,seg_loss_mask]
                #     image,seg_mask,seg_loss_mask = self.transform(sample)
                neg_samples.append((
                                   self.to_tensor(image), self.to_tensor(seg_mask), self.to_tensor(seg_loss_mask), True,
                                   image_path, seg_mask_path, part))
                # neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        # self.len = len(pos_samples) + len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()


def get_dataset(kind: str, cfg: Config) -> Optional[DataLoader]:
    if kind == "VAL" and not cfg.VALIDATE:
        return None
    if kind == "VAL" and cfg.VALIDATE_ON_TEST:
        kind = "TEST"
    if cfg.DATASET == "KSDD2":
        train_transform = transforms.Compose([RandomFlip(p=0.5)])
        # train_transform.transforms.append(RandomFlip(p=0.5))
        ds = KSDD2Dataset(kind, cfg, transform=train_transform)
    else:
        raise Exception(f"Unknown dataset {cfg.DATASET}")

    shuffle = kind == "TRAIN"
    batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1
    num_workers = 0
    drop_last = kind == "TRAIN"
    pin_memory = False
    # datas = mds.GeneratorDataset(ds, ['image', 'seg_mask', 'seg_loss_mask', 'is_segmented', 'sample_name'], shuffle=shuffle)
    datas = mds.GeneratorDataset(ds, ['image', 'seg_mask', 'seg_loss_mask', 'is_segmented', 'sample_name'], shuffle=shuffle)

    # return DataLoader(dataset=datas, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)
    return datas


def eval_model(model, eval_loader, save_folder, save_images, is_validation, plot_seg, cfg):
    model.set_train(False)
    # model.eval()
    # print(eval_loader)
    dsize = cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT

    res = []
    predictions, ground_truths = [], []

    for data_point in eval_loader:
        image, seg_mask, seg_loss_mask, _, sample_name = data_point
        # print(image.shape)
        # print(seg_mask.shape)
        # print(seg_loss_mask.shape)
        # is_pos = (seg_mask.max() > 0).reshape((1, 1)).item()
        # is_pos = (seg_mask.max() > 0)
        image = image[None, :]
        seg_mask = seg_mask[None, :]
        seg_loss_mask = seg_loss_mask[None, :]
        # print((seg_mask.max() > 0).reshape((1, 1)))
        is_pos = (seg_mask.max() > 0).reshape((1, 1)).item((0,0))

        prediction, pred_seg_1, pred_seg = model(image)
        # print(prediction)
        pred_seg = nn.Sigmoid()(pred_seg)
        prediction = nn.Sigmoid()(prediction)
        # print(prediction)
        prediction = prediction.item((0,0))
        image = ops.stop_gradient(image).numpy()
        # image = image.detach().numpy()
        # pred_seg = pred_seg.detach().numpy()
        pred_seg = ops.stop_gradient(pred_seg).numpy()
        # seg_mask = seg_mask.detach().numpy()
        seg_mask = ops.stop_gradient(seg_mask).numpy()
        predictions.append(prediction)
        ground_truths.append(is_pos)
        # print(prediction, None, None, is_pos, sample_name) 0.5142088 None None False 20637
        res.append((prediction, None, None, is_pos, sample_name))
        if not is_validation:
            if save_images:
                image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(
                    pred_seg[0, :, :], dsize)
                seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                prediction = float(prediction)
                if cfg.WEIGHTED_SEG_LOSS:
                    seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                    utils.plot_sample(sample_name, image, pred_seg, seg_loss_mask, save_folder, decision=prediction,
                                      plot_seg=plot_seg)
                else:
                    utils.plot_sample(sample_name, image, pred_seg, seg_mask, save_folder, decision=prediction,
                                      plot_seg=plot_seg)

    if is_validation:
        metrics = utils.get_metrics(np.array(ground_truths), np.array(predictions))
        FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
        _log(
            f"VALIDATION || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
            f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

        return metrics["AP"], metrics["accuracy"]
    else:
        run_name = f"{cfg.RUN_NAME}_FOLD_{cfg.FOLD}" if cfg.DATASET in ["KSDD", "DAGM"] else cfg.RUN_NAME
        results_path = os.path.join(cfg.RESULTS_PATH, cfg.DATASET)
        run_path = os.path.join(results_path, cfg.RUN_NAME)
        utils.evaluate_metrics(res, run_path, run_name)


def eval(cfg, model, save_images, plot_seg, outputs_path):
    test_loader = get_dataset("TEST", cfg)
    eval_model(model, test_loader, save_folder=outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg, cfg=cfg)


def _log(message, lvl=LVL_INFO):
    n_msg = f"{message}"
    if lvl >= LOG:
        print(n_msg)


if __name__ == '__main__':
    args = parse_args()
    configuration = Config()
    configuration.merge_from_args(args)
    configuration.init_extra()
    if configuration.GPU == 'Ascend':
        mindspore.set_context(device_target='Ascend', device_id=0)
        print('ok')
    net = SegDecNet_tf('cpu', configuration.INPUT_WIDTH, configuration.INPUT_HEIGHT, configuration.INPUT_CHANNELS)
    net = init(net)
    eval(configuration, net,  configuration.SAVE_IMAGES, False, 'out')

