import json, torch
import numpy as np
from retrying import retry
import pandas as pd
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForCausalLM 
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from google import genai
import config.paths as paths
from transformers import AutoImageProcessor, AutoModelForDepthEstimation    # Depth estimation

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
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
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
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
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
        labels = data.get('bboxes_labels', [])  
        
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
                 location=paths.gemini_location):
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model_id = "gemini-2.0-flash-001"
    
    # Define a retry strategy for running Gemini
    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
    def send_message_with_retry(self, chat, message):
        response = chat.send_message(message)
        if 'error' in response:
            raise Exception(response['error'])
        return response

# Segment Anything - image + click/bbox -> mask
class initSAM:
    def __init__(self, 
                paths=paths,):
        self.predictor = self.init_sam(paths.sam_checkpoint, paths.sam_cfg)   
    
    # initiate SAM
    def init_sam(self, checkpoint, model_cfg):
        checkpoint = checkpoint
        model_cfg = model_cfg
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        return predictor

# Depth Estimation - image -> depth array
class DepthEstimationProcessor:
    def __init__(self, 
                 depth_model_name=paths.depth_model_name):
        self.image_processor = AutoImageProcessor.from_pretrained(depth_model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)

    def get_depth(self, image_path):

        # prepare image for the model
        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # interpolate to original size and visualize the prediction
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        return predicted_depth