from PIL import Image
import pandas as pd
import os, glob, json, torch
import numpy as np
from modules.overgrown_veg.functions.det_veg_util import VegPvmtDetector, make_category_df, make_image_df
from config import paths
from modules.tools.initdet import initFlorence2, initGemini

class DetAndCrop:
    def __init__(self,
                 categories: list,
                 input_dir = paths.input_dir,
                 input_header = 'front',
                 output_dir = paths.output_dir,
                 output_img_header = '',
                 output_fn = '',
                 ):
        self.florence2 = initFlorence2()
        self.my_gemini = initGemini()
        self.categories = categories
        self.input_dir = input_dir
        self.input_header = input_header
        self.output_dir = output_dir
        self.output_img_header = output_img_header
        self.output_fn = output_fn

    def det(self):
        os.makedirs(self.output_dir, exist_ok=True)     # create output directory if it doesn't exist           
        image_list = self.create_imagelist()

        for idx, image_fn in enumerate(image_list):
            image = Image.open(image_fn)
            print(f"Processing {os.path.basename(image_fn)}")
            
            for cat_idx, cat in enumerate(self.categories):
                
                # Step 1 - detect the object
                det = self.detect_object(image, cat)
                
                # Step 2 - check if the crop actually contains vegetation/pavement by Gemini - can be replaced later with a free VLM
                for i, box in enumerate(det['bboxes']):
                    x1, y1, x2, y2 = box 
                    image_crop = image.crop((x1, y1, x2, y2))
                    
                    # Step 3 - if yes, save the crop in the output folder
                    if self.verify_crop(cat, image_crop):
                        base_fn, ext = os.path.splitext(os.path.basename(image_fn))       # use this method because it can be .jpg, .JPG or .png
                        output_fn = f"{self.output_dir}/{base_fn}{self.output_img_header}{ext}"
                        image_crop.save(output_fn)
    
    def det_with_descriptions(self):
        os.makedirs(self.output_dir, exist_ok=True)     # create output directory if it doesn't exist           
        image_list = self.create_imagelist()
        # crop_list = []

        for idx, image_fn in enumerate(image_list):
            image = Image.open(image_fn)
            print(f"Processing {os.path.basename(image_fn)}")
            
            for cat_idx, cat in enumerate(self.categories):
                
                # Step 1 - detect the object
                det = self.detect_object(image, cat)
                
                # Step 2 - check if the crop actually contains the category by Gemini - can be replaced later with a free VLM
                for i, box in enumerate(det['bboxes']):
                    x1, y1, x2, y2 = box 
                    image_crop = image.crop((x1, y1, x2, y2))
                    
                    # Step 3 - if yes, save the crop in the output folder
                    if self.describe_crop(cat, image_crop):         # ask Gemini to describe the image. Check if cat is in the response
                        base_fn, ext = os.path.splitext(os.path.basename(image_fn))       # use this method because it can be .jpg, .JPG or .png
                        output_fn = f"{self.output_dir}/{base_fn}{self.output_img_header}{ext}"
                        image_crop.save(output_fn)
        #                 # Step 4 - save a json file for putting back the crop to the image
        #                 crop_list = self.create_list_for_json(crop_list, image_fn, output_fn, box)
        
        # # save the results
        # crop_df = pd.DataFrame(crop_list)
        # dict_to_json = crop_df.to_dict('records')
        # with open(self.output_fn, "w") as outfile:
        #     json.dump(dict_to_json, outfile, cls=myplot.NpEncoder)

    def create_imagelist(self):
        patterns = [
            f'{self.input_header}*.jpg',
            f'{self.input_header}*.JPG',
            f'{self.input_header}*.png'
        ]
        image_list = []
        for pattern in patterns:
            image_list.extend(glob.glob(os.path.join(self.input_dir, pattern)))
        image_list = sorted(image_list)
        return image_list

    def detect_object(self, image, cat):
        task_prompt = '<OPEN_VOCABULARY_DETECTION>'
        results = self.florence2.run_example(task_prompt, image, text_input=cat)
        return self.florence2.convert_to_od_format(results.get(task_prompt, {}))

    def verify_crop(self, cat, image_crop):
        veriq_1 = f"Does the image contain {cat} ? Only answer yes or no."
        try:
            chat = self.my_gemini.client.chats.create(model=self.my_gemini.model_id)
            answer1 = self.my_gemini.send_message_with_retry(chat, [veriq_1, image_crop])
            response = answer1.text.lower()
            if "yes" in response:
                return True
            elif "no" in response:
                return False
            else:
                print(f"Unexpected response: {response}")
                return False
        except Exception as e:
            print(f"Failed to get response: {e}")
            return False
    
    def describe_crop(self, cat, image_crop):
        veriq_1 = f"Please describe in details what the image contains."
        try:
            chat = self.my_gemini.client.chats.create(model=self.my_gemini.model_id)
            answer1 = self.my_gemini.send_message_with_retry(chat, [veriq_1, image_crop])
            response = answer1.text.lower()
            if cat in response:
                return True
            else:
                return False
        except Exception as e:
            print(f"Failed to get response: {e}")
            return False
    
    def create_list_for_json(self, 
                             crop_list,
                             image_fn,
                             output_fn,
                             bbox):
        x1, y1, x2, y2 = bbox
        crop_list.append({'original_image': os.path.basename(image_fn), 
        'cropped_image': output_fn,
        'bbox': [x1, y1, x2-x1, y2-y1]})
        return crop_list