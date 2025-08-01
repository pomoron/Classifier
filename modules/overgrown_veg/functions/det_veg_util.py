import json, torch
import numpy as np
# from retrying import retry
import pandas as pd
from PIL import Image
import os
# from transformers import AutoProcessor, AutoModelForCausalLM 
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from google import genai
from modules.tools.initdet import initGemini, initFlorence2, initSAM

# ---------------------------
# Functions for detecting vegetation and pavement
# ---------------------------

class VegPvmtDetector:

    def init(self, 
             input_dir, 
             out_fn,  
             paths,):
        self.home = os.path.expanduser('~')
        self.input_dir = input_dir
        self.out_fn = out_fn
        # Instantiate initFlorence2 and preserve its functions as an attribute.
        self.florence = initFlorence2()
        self.model, self.processor = self.florence.model, self.florence.processor
        # Initialize Gemini
        self.my_gemini = initGemini()
        self.client = self.my_gemini.client
        self.gemini_model_id = self.my_gemini.model_id
        # Initialize SAM
        self.sam = initSAM()
        self.predictor = self.sam.predictor
        
    # Run Florence-2
    def run_example(self, task_prompt, image, text_input=None):
        return self.florence.run_example(task_prompt, image, text_input)

    def convert_to_od_format(self, data):
        return self.florence.convert_to_od_format(data)  

    # For detecting masks with Florence-2
    def convert_to_loc_format(coords, height, width):
        return self.florence.convert_to_loc_format(coords, height, width)

    # Run Gemini with a retry strategy
    def send_message_with_retry(self, chat, message):
        return self.my_gemini.send_message_with_retry(chat, message)


# convert all np.integer, np.floating and np.ndarray into json recognisable int, float and lists
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Make category df
def make_category_df(categories):
    cat_df = pd.DataFrame(categories, columns=['name'])
    cat_df['category_id'] = cat_df.index + 1
    cat_df = cat_df[['category_id', 'name']]
    return cat_df

# Make image df
def make_image_df(image_list):
    width_all, height_all, fn_base = [], [], []
    # assumed the image_list already contains the full path and is already sorted
    for image_fn in image_list:
        with Image.open(image_fn) as img:
            width, height = img.size
            width_all.append(width)
            height_all.append(height)
            fn_base.append(os.path.basename(image_fn))
    img_df = pd.DataFrame(fn_base, columns=['file_name'])
    img_df['width'] = width_all
    img_df['height'] = height_all
    img_df['id'] = img_df.index + 1
    img_df = img_df[['id', 'width','height', 'file_name']]
    return img_df