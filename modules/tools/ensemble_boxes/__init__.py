# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
'''
Downloaded from Github repo ZFTurbo/Weighted-Boxes-Fusion
Attribute:  
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
'''

from .ensemble_boxes_wbf import weighted_boxes_fusion
from .ensemble_boxes_nmw import non_maximum_weighted
from .ensemble_boxes_nms import nms_method
from .ensemble_boxes_nms import nms
from .ensemble_boxes_nms import soft_nms
from .ensemble_boxes_wbf_3d import weighted_boxes_fusion_3d
from .ensemble_boxes_wbf_1d import weighted_boxes_fusion_1d
from .ensemble_boxes_wbf_experimental import weighted_boxes_fusion_experimental