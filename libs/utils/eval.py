# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

import logging
from collections import OrderedDict
import PIL.Image as img
from PIL import ImageDraw
import numpy as np

from fvcore.common.file_io import PathManager


# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
        
def save_annotation(label,
                    filename,
                    add_colormap=None,
                    normalize_to_unit_values=False,
                    scale_values=False,
                    colormap=None,
                    palette=None,
                    image=None):
    # Add colormap for visualizing the prediction.
    if add_colormap:
        colored_label = label_to_color_image(label, colormap)
    else:
        colored_label = label
    if normalize_to_unit_values:
        min_value = np.amin(colored_label)
        max_value = np.amax(colored_label)
        range_value = max_value - min_value
        if range_value != 0:
            colored_label = (colored_label - min_value) / range_value
    
    if scale_values:
        colored_label = 255. * colored_label

    if image is not None:
        colored_label = 0.5 * colored_label + 0.5 * image
    pil_image = img.fromarray(colored_label.astype(dtype=np.uint8))
    if palette is not None:
        pil_image.putpalette(palette)
    with open(filename, mode='wb') as f:
        pil_image.save(f, 'PNG')

def label_to_color_image(label, colormap=None):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
        colormap: A colormap for visualizing segmentation results.
    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the dataset color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

    if colormap is None:
        raise ValueError('Expect a valid colormap.')

    return colormap[label]

class SemanticEvaluator:
    """
    Evaluate semantic segmentation
    """
    def __init__(self, num_classes, ignore_label=255, output=True, train_id_to_eval_id=None):
        """
        Args:
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            train_id_to_eval_id (list): maps training id to evaluation id.
        """
        self._output = output
        #if self._output_dir:
        #    PathManager.mkdirs(self._output_dir)
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1  # store ignore label in the last class
        self._train_id_to_eval_id = train_id_to_eval_id

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
            train_id_to_eval_id (list): maps training id to evaluation id.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction

    def update(self, pred, gt, image_filename=None, palette=None):
        pred = pred.astype(np.int)
        gt = gt.astype(np.int)
        gt[gt == self._ignore_label] = self._num_classes

        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)
        
        if self._output:
            if self._train_id_to_eval_id is not None:
                pred = self._convert_train_id_to_eval_id(pred, self._train_id_to_eval_id)
            if image_filename is None:
                raise ValueError('Need to provide filename to save.')
            save_annotation(
                pred, image_filename, palette=palette)
        
    def reset(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
    
    
    
    
    
class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pACC": acc,
                "mACC": acc_cls,
                "fwIoU": fwavacc,
                "mIoU": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
