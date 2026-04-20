from PIL import Image
import pandas as pd
import os, glob, json, torch
import numpy as np
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
                 gemini_client = None,
                 task_prompt = '<OPEN_VOCABULARY_DETECTION>',
                 skip_verification = False,
                 ):
        self.florence2 = initFlorence2()
        self.my_gemini = gemini_client if gemini_client is not None else initGemini()
        self.categories = categories
        self.input_dir = input_dir
        self.input_header = input_header
        self.output_dir = output_dir
        self.output_img_header = output_img_header
        self.output_fn = output_fn
        self.task_prompt = task_prompt
        self.skip_veri = skip_verification
        self.det_bbox = []

    def det(self, bool_output_image=True):
        os.makedirs(self.output_dir, exist_ok=True)     # create output directory if it doesn't exist           
        image_list = self.create_imagelist()

        for idx, image_fn in enumerate(image_list):
            image = Image.open(image_fn)
            print(f"Processing {os.path.basename(image_fn)}")
            
            for cat_idx, cat in enumerate(self.categories):
                
                # Step 1 - detect the object
                det = self.detect_object(image, cat)
                
                # Step 2 - check if the crop actually contains vegetation/pavement by Gemini - can be replaced later with a free VLM
                if det['bboxes']:
                    # Find the largest bbox instead of processing all
                    # This however limits to one asset per type per image
                    largest_box = max(det['bboxes'], key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                    x1, y1, x2, y2 = largest_box
                    image_crop = image.crop((x1, y1, x2, y2))
                    
                    if self.skip_veri:
                        should_save = True
                    elif bool_output_image == False:
                        should_save = False
                    else:
                        should_save = self.verify_crop(cat, image_crop)
                    # if self.verify_crop(cat, image_crop):
                    if should_save:
                        base_fn, ext = os.path.splitext(os.path.basename(image_fn))
                        output_fn = f"{self.output_dir}/{base_fn}{self.output_img_header}{ext}"
                        image_crop.save(output_fn)
                    
                    image_json = {'name': os.path.basename(image_fn), 'bbox': largest_box, 'category': cat}
                    self.det_bbox.append(image_json)
                    # for i, box in enumerate(det['bboxes']):
                    #     x1, y1, x2, y2 = box 
                    #     image_crop = image.crop((x1, y1, x2, y2))
                        
                    #     # Step 3 - if yes, save the crop in the output folder
                    #     if self.verify_crop(cat, image_crop):
                    #         base_fn, ext = os.path.splitext(os.path.basename(image_fn))       # use this method because it can be .jpg, .JPG or .png
                    #         if base_fn.endswith('crop_0'):
                    #             base_fn = base_fn.replace('crop_0', f'crop_{i}')   # replace crop_0 with crop_i
                    #         output_fn = f"{self.output_dir}/{base_fn}{self.output_img_header}{ext}"
                    #         image_crop.save(output_fn)
                else:
                    print(f"No {cat} detected in {os.path.basename(image_fn)}")
        
        return self.det_bbox
            
    
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
        task_prompt = self.task_prompt
        results = self.florence2.run_example(task_prompt, image, text_input=cat)
        return self.florence2.convert_to_od_format(results.get(task_prompt, {}))
        # try:
        #     results = self.florence2.run_example(task_prompt, image, text_input=cat)
        #     return self.florence2.convert_to_od_format(results.get(task_prompt, {}))
        # except Exception as e:
        #     print(f"Failed to get detection results: {e}")
        #     return {'bboxes': [], 'labels': []}

    def verify_crop(self, cat, image_crop):
        # Added in case more than 1 word is supplied in the category
        if self.task_prompt != '<OPEN_VOCABULARY_DETECTION>':
            check_cat = cat.split(' ')[0]   # take the first word only
        else:
            check_cat = cat
        
        # Ask Gemini if the crop contains the category
        veriq_1 = f"Does the image contain {check_cat} ? Only answer yes or no."
        try:
            chat = self.my_gemini.client.chats.create(model=self.my_gemini.model_id)
            answer1 = self.my_gemini.send_message_with_retry(chat, [veriq_1, image_crop])
            response = answer1.text.lower()
            # print(f"Verification response: {response}")
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