import json
import os
import glob
from collections import namedtuple
import zipfile
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
"""
_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train': 2975,
                     'trainval': 3475,
                     'val': 500,
                     'test': 1525},
    num_classes=19,
    ignore_label=255,
)
"""

_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

class Cityscapes(VisionDataset):
    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "semantic",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []

        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        
        self.images = self._get_files('image', self.split)
        self.targets = self._get_files('label', self.split)
        
        
        
    def _get_files(self, data, dataset_split):
        """from panoptic-deeplab
        Gets files for the specified data type and dataset split.
        Args:
            data: String, desired data ('image' or 'label').
            dataset_split: String, dataset split ('train', 'val', 'test')
        Returns:
            A list of sorted file names or None when getting label for test set.
        """
        pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
        search_files = os.path.join(
            self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
        filenames = glob.glob(search_files)
        if data == 'label' and dataset_split == 'test':
            f_names = []
            for k in range(len(filenames)):
                path = filenames[k]
                bar_p = []
                under_p = []
                for i, x in enumerate(path):
                    if(x == '/'):
                        bar_p.append(i)
                file = path[bar_p[-1]+1:]
                for i, x in enumerate(file):
                    if(x == '_'):
                        under_p.append(i)
                name = file[:under_p[-2]]
                f_names.append(name+'.png')
            return sorted(f_names)
        return sorted(filenames)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')
        if self.split == 'test':
            image = F.to_tensor(image)
            image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            return image, self.targets[index]
        
        target = Image.open(self.targets[index])
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.images)
    
    def train_id_to_eval_id(self):
        return _CITYSCAPES_TRAIN_ID_TO_EVAL_ID
    
    def _convert_train_id_to_eval_id(self, prediction):
        """Converts the predicted label for evaluation.
        There are cases where the training labels are not equal to the evaluation
        labels. This function is used to perform the conversion so that we could
        evaluate the results on the evaluation server.
        Args:
            prediction: Semantic segmentation prediction.
        Returns:
            Semantic segmentation prediction whose labels have been changed.
        """
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(self.train_id_to_eval_id()):
            converted_prediction[prediction == train_id] = eval_id
        return converted_prediction
    
    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)