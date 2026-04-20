import torch, glob, os
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from . import plot as myplot

class predMask:
    def __init__(self, 
                 model_cfg, 
                 checkpoint):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.predictor = SAM2ImagePredictor(build_sam2(self.model_cfg, self.checkpoint))

    # Predicting masks
    def create_imagelist(self,
                       input_dir,
                       input_header = None,
                       ):
        if input_header is None:
            patterns = ['*.jpg', '*.JPG', '*.png']
        else:
            patterns = [
                f'{input_header}*.jpg',
                f'{input_header}*.JPG',
                f'{input_header}*.png'
            ]
        image_list = []
        for pattern in patterns:
            image_list.extend(glob.glob(os.path.join(input_dir, pattern)))
        image_list = sorted(image_list)
        return image_list

    
    def sam_pred_mask(self, 
                       image,                   # cv2 image object
                       sep_centroids = None,    # list of (x,y) tuples
                       df_bbox = None,             # list of bounding boxes
                       bbox_os = 0           # bbox offset
                       ):
        masks_accum, scores_accum, logits_accum = [], [], []

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            h, w = image.shape[:2]
            # masks, _, _ = predictor.predict(<input_prompts>)

            # generate cues from sep_centroids and df_bbox
            if sep_centroids is not None:
                cues = [[centroid, [centroid[0], min(centroid[1] + 200, w)]] for centroid in sep_centroids]
            elif df_bbox is not None:
                # Find bbox of the segmented objects
                # Assumed bbox format: [x, y, w, h]
                h, w = image.shape[:2]
                cues = [[max(box[0]-bbox_os,0), 
                            max(box[1]-bbox_os,0), 
                            min(max(box[0]-bbox_os,0)+box[2]+2*bbox_os,w),
                            min(max(box[1]-bbox_os,0)+box[3]+2*bbox_os,h)] for box in df_bbox]
            else:
                cues = [[0, 0, w, h]]  # whole image as cue
            
            for cue in cues:
                # create a negative point for each centroid
                # pnt_in = [centroid, [centroid[0], min(centroid[1] + 200, w)]]
                # pnt_in = [centroid]
                if sep_centroids is not None:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=np.array(cue),
                        point_labels=np.array([1, 0]),
                        multimask_output=False,
                    )
                else:
                    masks, scores, logits = self.predictor.predict(
                        box=cue,
                        multimask_output=False,
                    )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
                masks_accum.append(masks)
                scores_accum.append(scores)
                logits_accum.append(logits)

            # del predictor
        return masks_accum, scores_accum, logits_accum
    
    def one_polygon(self,
                 masks_accum,
                 df_bbox,):
        plygn_out, area_out, bbox_out = [], [], []
        for x, mask in enumerate(masks_accum):
            plygn, area, bbox = myplot.mask_to_polygon(mask[0])        # restricted SAM to output 1 mask per centroid
            if plygn is None or area == 0 or bbox is None:
                plygn = [[]]
                area = df_bbox[x][2] * df_bbox[x][3]  # Use the area of the bbox as a fallback
                actual_bbox = df_bbox[x]
            else:
                actual_bbox = myplot.find_remasked_bbox(mask[0])    # I just want 1 bbox for polygon (not every sub-polygon)
            plygn_out.append(plygn)
            area_out.append(area)
            bbox_out.append(actual_bbox)
        
        return plygn_out, area_out, bbox_out