import os
import config.paths as paths
from .functions.yolodet import YOLODetection
from datetime import datetime


def infer_yolo(model_weight: str = paths.pvmt_weights,
               output_fn: str = 'pvmt_infer.json',
               vis_pred: bool = False,
               output_vis_dir: str = None):
    yolodet = YOLODetection(
        model_path = model_weight,
        img_path = paths.pavement_indir,
        test_dir = paths.pavement_outdir,
        output_fn = output_fn,
        # cat_weight={},
        # sel_image=False,
        vis_pred=vis_pred,
        output_vis_dir = output_vis_dir,
    )
    yolodet.forward()

def det_pvmt():
    print("Start pavement detection...")

    # infer pavement distresses
    infer_yolo(model_weight = paths.pvmt_weights,
               output_fn = 'pvmt_infer.json',
               vis_pred = True,
               output_vis_dir = 'pvmt_vis')

    print("Start detecting extra defects on pavements...")
    # start_extra = datetime.now()
    # infer extra defects (alligator cracks, worn lane marks...)
    infer_yolo(model_weight = paths.pvmt_extra_weights,
               output_fn = 'pvmt_extra_infer.json',
               vis_pred = True,
               output_vis_dir = 'extra_vis')


if __name__ == '__main__':
    det_pvmt()
                 
                    

