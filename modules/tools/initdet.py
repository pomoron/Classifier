import json, torch
import numpy as np
from retrying import retry
import pandas as pd
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForCausalLM
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
from google import genai
import config.paths as paths
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTForDepthEstimation    # Depth estimation

# ---------------------------
# Common utilities for different detectors
# ---------------------------

# Florence2 - image (+ text) -> text/bbox
class initFlorence2:
    def __init__(self):
        self.model, self.processor = self.init_florence2()
    
    # Initiate Florence-2
    def init_florence2(self):
        model_id = 'microsoft/Florence-2-large'
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # added attn_implementation="eager" to bypass the SDPA check in case the model code hasn't updated to use this
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype='auto', attn_implementation="eager").eval().cuda()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor
    
    # Run Florence-2
    def run_example(self, task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.model.generate(
            # input_ids=inputs["input_ids"].cuda(),
            # pixel_values=inputs["pixel_values"].cuda(),
            **inputs,
            max_new_tokens=1024,
            use_cache=False,  # Add this line to disable caching - some bugs appeared when using cache
            # early_stopping=False,
            # do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

    def convert_to_od_format(self, data):  
        """  
        Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
        Parameters:  
        - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys. 
        Returns:  
        - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
        """  
        # Extract bounding boxes and labels  
        bboxes = data.get('bboxes', [])  
        labels = data.get('labels', [])  
        
        # Construct the output format  
        od_results = {  
            'bboxes': bboxes,  
            'labels': labels  
        }  
        
        return od_results

    # For detecting masks with Florence-2
    def convert_to_loc_format(coords, height, width):
        sf = 1000
        divide_by = [width, height, width, height]
        new_coords = [int(x/divide_by[idx]*sf) for idx, x in enumerate(coords)]
        x1s, y1s, x2s, y2s = new_coords
        loc_format = f'<loc_{x1s}><loc_{y1s}><loc_{x2s}><loc_{y2s}>'
        return loc_format

# Gemini - image -> text
class initGemini:
    # Initialize the model via Vertex AI (free credits from Google Cloud)
    # Separated this as a class as the initialisation method may be used in other classes
    def __init__(self, 
                 project=paths.gemini_project, 
                 location=paths.gemini_location,
                 model_id="gemini-2.0-flash-001"):
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model_id = model_id
    
    # Define a retry strategy for running Gemini
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
    def send_message_with_retry(self, chat, message):
        response = chat.send_message(message)
        if 'error' in response:
            raise Exception(response['error'])
        return response

class initGeminiFromAPI(initGemini):
    # Initialise Gemini via Gemini API instead of Vertex AI
    def __init__(self, 
                 api_key,
                 model_id="gemini-2.0-flash-001"):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

# Abandoned SAM2 - check sam3mask.py for the latest SAM implementation using SAM3
# # Segment Anything - image + click/bbox -> mask
# class initSAM:
#     def __init__(self, 
#                 paths=paths,):
#         self.predictor = self.init_sam(paths.sam_checkpoint, paths.sam_cfg)   
    
#     # initiate SAM
#     def init_sam(self, checkpoint, model_cfg):
#         checkpoint = checkpoint
#         model_cfg = model_cfg
#         predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
#         return predictor

# Depth Estimation - image -> depth array
class DepthEstimationProcessor:
    def __init__(self, 
                 depth_model_name=paths.depth_model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = AutoImageProcessor.from_pretrained(depth_model_name)
        if depth_model_name == "facebook/dpt-dinov2-giant-nyu":
            self.model = DPTForDepthEstimation.from_pretrained(depth_model_name).to(self.device).eval()
        else:
            self.model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(self.device).eval()

    def get_depth(self, image_path):

        # prepare image for the model
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # interpolate to original size and visualize the prediction
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        np_predicted_depth = predicted_depth.detach().cpu().numpy()
        return np_predicted_depth
    
    def get_depth_dinov2(self,
                         image_path):
        '''
        Get depth estimation using DINOv2 model
        The function first resizes the image to a fixed width of 640 while maintaining aspect ratio.
        It then runs the depth estimation model and interpolates the output back to the original image size.

        Inputs:
        - image_path: str - path to the input image

        Outputs:
        - output: np.array - depth estimation array (size: HxW) corresponding to the input image
        '''
        image = Image.open(image_path)
        im_w, im_h = image.size
        if os.path.splitext(image_path)[-1].lower() == '.png':
            image = image.convert("RGB")    # Convert PNG to RGB if necessary
        w_ratio = 640 / im_w
        image = image.resize((640, int(im_h * w_ratio)))    # Resize to a fixed size for the model for efficient depth estimation

        # Step 1 - Predict depth
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(   # interpolate to original size
            predicted_depth.unsqueeze(1),
            size=(im_h, im_w),
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        return output