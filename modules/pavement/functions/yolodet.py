from ultralytics import YOLO
import pandas as pd
import os, json
import config.paths as path
import modules.tools.plot as myplot
import utils.merge as mymerge

class YOLODetection:
    def __init__(self, 
                 model_path,
                 img_path,
                 test_dir,
                 output_fn,
                 cat_weight: dict={},
                 sel_image=False,
                 vis_pred=False,
                 output_vis_dir:str =None,
                 ):
        self.model = YOLO(model_path)
        self.img_path = img_path
        self.test_dir = test_dir
        self.output_fn = os.path.join(self.test_dir, output_fn)
        self.sel_image = sel_image
        self.cat_weight = cat_weight
        self.vis_pred = vis_pred
        self.output_vis_dir = os.path.join(self.test_dir, output_vis_dir)
    
    def forward(self,):
        results = self.predict(self.img_path)
        category, images, df, df_coltitle = self.create_df()

        # Process results list
        # iterate to unpack the results object. Objects are divided by image
        for i, result in enumerate(results):

            try:
                result_sum = result.summary()  # returns a list of instance data for each image
                # but result.summary() always returns the last mask of the image, which is wrong...
            except:
                print(f"No bbox predicted or other errors in opening {result.path}")
                continue

            # set categories in the first run
            if category.empty:
                category_list = [{"category_id": k+1, "name": v} for k, v in result.names.items()]
                category = pd.DataFrame(category_list, columns=["category_id", "name"])

            # extract masks from result
            if result_sum:
                mask_list = self.extract_masks(result)

            # add image to images df
            img_name = os.path.basename(result.path)
            height, width, _ = result.orig_img.shape
            images.loc[len(images)] = {"id": i+1, "file_name": img_name, "width": width, "height": height}

            # add annotations to df
            for j, obj in enumerate(result_sum):
                bbox = myplot.process_bbox(obj['box'])
                segment = myplot.process_segment(mask_list[j])
                area = myplot.calculate_area(segment)
                df.loc[len(df)] = {"id": len(df)+1, "image_id": i+1, "category_id": obj['class']+1, "bbox": bbox, "area": area, "segmentation": segment, "score": obj['confidence']}

            # combine masks - YOLO predicts a lot of masks but sometimes they can be combined
            df_check = df[df['image_id']== i+1]
            if len(result_sum) > 1:
                df, df_check = self.combine_masks(i, df, df_check, df_coltitle, height, width)
            
            # visualize the results
            if self.vis_pred:
                os.makedirs(self.output_vis_dir, exist_ok=True)
                pred_img_fn = os.path.join(self.output_vis_dir, img_name)
                myplot.new_plot(os.path.join(self.img_path, img_name), df_check, category, pred_img_fn)
        
        # output image selection
        if self.sel_image and len(df)>0:
            if not self.cat_weight:     # if no weights are given, set all to 1
                cat_weight = {x: 1 for x in df['category_id'].unique()}
            _, images_score = self.image_sel(df, images, cat_weight)
            images_score = images_score.sort_values(by='total_score', ascending=False)
            images_score.to_csv(os.path.join(self.test_dir,'images_score.csv'), index=False)
        
        # clean dataframes for json dump
        category, df = myplot.cleanForJson(category, df)
        dict_to_json = {
            "categories": category.to_dict('records'),
            "images": images.to_dict('records'),
            "annotations": df.to_dict('records')
            }
        with open(self.output_fn, "w") as outfile:
            json.dump(dict_to_json, outfile, cls=myplot.NpEncoder)

        del category, images, df

    # make inference
    def predict(self, img_path, conf=0.25):
        results = self.model(img_path, stream=True, conf=conf)
        return results
    
    # create initial dataframes
    def create_df(self):
        images = pd.DataFrame(columns=["id", "file_name", "width", "height"])
        df_columnsTitles = ["id", "image_id", "category_id", "bbox", "area", "segmentation", "score"]
        df = pd.DataFrame(columns=df_columnsTitles)
        category = pd.DataFrame(columns=["category_id", "name"])  # Initialize category as an empty DataFrame
        return category, images, df, df_columnsTitles
    
    def extract_masks(self, result):
        result_mask = result.masks
        mask_list = []
        for x in result_mask.xy:
            seg_dict = {"x": x[:,0], "y": x[:,1]}
            mask_list.append(seg_dict)
        return mask_list

    def combine_masks(self, i, df, df_check, df_columnsTitles, height, width):
        # print(f"nos. predictions: {len(result_sum)}")
        df_check = df_check[df_check['segmentation'].apply(lambda x: x != [[]])]        # remove instances that have empty segmentations
        df_check = mymerge.df_combine_masks(df_check, height, width)
        df = df.drop(df[df['image_id'] == i+1].index)
        df = pd.concat([df,df_check], ignore_index=True)
        df = df.reindex(columns=df_columnsTitles)
        df['id'] = df.index + 1
        return df, df_check
    
    # Score util
    def image_sel(self, df, images, weights):
        # check if weights cover all categories
        for i, x in enumerate(df['category_id'].unique()):
            if x not in weights.keys():
                weights[x] = 0
        
        # calculate individual scores
        df_score = df.copy()
        cat_nos = df['category_id'].value_counts().to_dict()
        cat_score = {k: sum(cat_nos.values())/v for k, v in cat_nos.items()}
        df_score['cls_score'] = 1 - df_score['score']
        df_score['abun_score'] = 1
        df_score['cat_score'] = df_score.apply(lambda p: cat_score[p['category_id']] * weights[p['category_id']], axis=1)

        images_df = images.copy()
        # calculate final score on image level
        for i, x in enumerate(df_score['image_id'].unique()):
            df_concern = df_score[df_score['image_id']==x]
            cls_avg_score = 1 * df_concern['cls_score'].mean()
            cls_avg_conf = 1 - cls_avg_score
            nos_pred = len(df_concern)
            abun_score = 0.1 * df_concern['abun_score'].sum()
            cat_pred = df_concern['category_id'].to_list()
            cat_score = 1 * df_concern['cat_score'].mean()
            total_score =  cls_avg_score + abun_score + cat_score
            new_columns = ['class avg. conf', 'nos predictions', 'categories', 'cls_score', 'abun_score', 'cat_score', 'total_score']
            add_list = [cls_avg_conf, int(nos_pred), str(cat_pred), cls_avg_score, abun_score, cat_score, total_score]
            for ind, col in enumerate(new_columns):
                images_df.loc[images_df['id'] == x, col] = add_list[ind]
 
        return df_score, images_df
