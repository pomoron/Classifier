from PIL import Image
import pandas as pd
import os, glob, json, torch
import numpy as np
from modules.overgrown_veg.functions.det_veg_util import VegPvmtDetector, make_category_df, make_image_df
from config import paths
import modules.tools.plot as myplot

def main(
    input_dir=paths.input_dir,
    detveg_fn=paths.detveg_fn,
    want_plot=paths.detveg_want_plot,
    ):
    df_cols = ['id', 'image_id', 'category_id', 'segmentation', 'area', 'bbox', 'score']
    categories = ['vegetation', 'pavement']

    if want_plot:
        output_dir = paths.detveg_visdir
        os.makedirs(output_dir, exist_ok=True)

    ovd = VegPvmtDetector()
    ovd.init( 
            input_dir = input_dir, 
            out_fn = detveg_fn,  
            paths = paths, 
            )
    image_list = sorted(glob.glob(os.path.join(input_dir, 'front*.jpg')))
    df = pd.DataFrame(columns=df_cols)
    category_df = make_category_df(categories)
    image_df = make_image_df(image_list)

    for idx, image_fn in enumerate(image_list):
        image = Image.open(image_fn)
        print(f"Processing {os.path.basename(image_fn)}")
        
        for cat_idx, cat in enumerate(categories):
            # Step 1 - detect the object
            task_prompt = '<OPEN_VOCABULARY_DETECTION>'
            results = ovd.run_example(task_prompt, image, text_input=cat)
            det = ovd.convert_to_od_format(results.get(task_prompt, {}))
            for i, box in enumerate(det['bboxes']):
                x1, y1, x2, y2 = box 
                image_crop = image.crop((x1, y1, x2, y2))
                defect = dict()

                # Step 2 - check if the crop actually contains vegetation/pavement by Gemini - can be replaced later with a free VLM
                veriq_1 = f"Does the image contain {cat} ? Only answer yes or no."
                try:
                    chat = ovd.client.chats.create(model=ovd.gemini_model_id)
                    answer1 = ovd.send_message_with_retry(chat, [veriq_1, image_crop])
                except Exception as e:
                    print(f"Failed to get response: {e}")
                    continue
                
                # Step 3 - if yes, create a segmentation mask in the bbox
                if 'yes' in answer1.text.lower():

                    # Create segmentation mask using SAM
                    box_np = np.array(box)
                    ovd.predictor.set_image(image)
                    masks, scores, _ = ovd.predictor.predict(
                        point_coords=np.array([[1000, 1500]]),      # Given I know the top of the car
                        point_labels=np.array([0]),
                        box=box_np,
                        multimask_output=False,
                    )
                    polygon, area, _ = myplot.mask_to_polygon(masks[0])
                    defect['area'] = area
                    defect['score'] = scores[0]

                    defect['id'] = len(df) + 1
                    defect['image_id'] = idx + 1
                    defect['category_id'] = cat_idx + 1       
                    defect['segmentation'] = polygon
                    defect['bbox'] = [x1, y1, x2-x1, y2-y1]
                    defect_df = pd.DataFrame([defect], columns=df_cols)
                    df = pd.concat([df, defect_df], ignore_index=True)
                else:
                    print(f"Detected bbox does not contain {cat}")
        
        # Step 4 - plot the results    
        if want_plot:
            df_plot = df[df['image_id'] == idx + 1]
            plot_name = os.path.join(output_dir, os.path.basename(image_fn))
            myplot.new_plot(image_fn, df_plot, category_df, plot_name)
        
    del ovd.predictor, ovd.model, ovd.processor
    torch.cuda.empty_cache()
    # save the results
    category_df, df = myplot.cleanForJson(category_df, df)
    # Write categories, images and annotations as arrays containing individual entries as a dictionary
    # see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html for to_dict styles
    dict_to_json = {
        "categories": category_df.to_dict('records'),
        "images": image_df.to_dict('records'),
        "annotations": df.to_dict('records')
        }
    with open(detveg_fn, "w") as outfile:
        json.dump(dict_to_json, outfile, cls=myplot.NpEncoder)

if __name__ == "__main__":
    main()