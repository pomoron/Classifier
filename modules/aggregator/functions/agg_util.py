import re
import pandas as pd
import numpy as np
from modules.misc_vlm.functions.call import vlmCleanQ2
import math, json
from collections import Counter
from itertools import chain

class Aggregator(vlmCleanQ2):
    def __init__(self,
                #   *args, **kwargs
                  ):
        pass    # The input parameters to vlmCleanQ2 are only for producing the cleaned Q2 csv. I just need the functions here. 
        # super().__init__(*args, **kwargs)
        # self.extra_param = extra_param
        
    def extract_justification(self,
                              text):
        # First, try to find a header like "Justification:" and capture text after it.
        header_match = re.search(r'Justification:\s*(.*)', text, flags=re.DOTALL | re.IGNORECASE)
        if header_match:
            justification = header_match.group(1).strip()
            return justification

        # Next, check if the text starts with Yes or No followed by justification text.
        # yes_no_match = re.compile(r'^(Yes|No)(?:\s+(.*)|\s*\n\s*(?:\n\s*)?(.*))', text, flags=re.DOTALL | re.IGNORECASE)
        pattern = re.compile(
            r'^(Yes|No)(?:\s+(.*)|\s*\n\s*(?:\n\s*)?(.*))',
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
        )
        yes_no_match = pattern.match(text)
        if yes_no_match:
            # If inline justification is found, use it; otherwise use the newline version.
            justification = yes_no_match.group(2) if yes_no_match.group(2) else yes_no_match.group(3)
            return justification.strip() if justification else ''

        # If no header, try to extract bullet points.
        bullet_iter = list(re.finditer(r'^\s*\*\s*.*', text, flags=re.MULTILINE))
        if bullet_iter:
            # Get text after the last bullet point.
            last_bullet_end = bullet_iter[-1].end()
            justification = text[last_bullet_end:].strip()
            if justification:
                return justification

        # Fallback: return the entire text trimmed.
        return text.strip()
    
    def assemble_df(self,
                    overall_df,
                    read_df,
                    folder):
        # Read comments in csv
        q1 = read_df['q1'].tolist()
        q2 = read_df['q2'].tolist()
        
        # Add metadata
        read_df.loc[:, 'asset_type'] = folder

        for ind, x in enumerate(q2):    
            # Extract the defects
            if folder=="obstructions":
                defects = self.extract_one_defect(x, 'Toppled')
                if defects == False:    # some problems when extracting defects
                    continue
            else:
                defects = self.extract_defects(x)
            # Skip if all defects are 'No'
            if all(value == 'No' for value in defects.values()):
                continue
            read_df.loc[ind, 'defects'] = [defects]
            
            # Extract the justification
            justification = self.extract_justification(x)
            # For cases in obstruction where there are no justifications after yes/no
            if not justification or folder=="obstructions":   
                justification = justification + q1[ind]
            read_df.loc[ind, 'descriptions'] = justification

        read_df.drop(columns=['q1', 'q2'], inplace=True)    # Drop the original columns
        overall_df = pd.concat([overall_df, read_df], ignore_index=True)        # Merge the dataframes
        return overall_df
    
    def is_nan(self, x):
        try:
            return math.isnan(x)
        except TypeError:
            return False

    def majority_vote_defects(self, 
                              defects_list):
        """
        Given a list of dictionaries (each with defect keys and values 'Yes'/'No' or NaN),
        returns a dictionary where each defect key maps to a dictionary containing:
        - 'decision': the majority vote ('Yes' or 'No')
        - 'indices': a list of indices from defects_list that contributed that vote.
        
        If votes are missing or tied, defaults to 'No' with contributing indices for 'No'.
        
        Parameters:
            defects_list (list): List of dictionaries with defect verdicts, which can include NaN.
            
        Returns:
            dict: Dictionary with a majority vote decision and contributing indices for each defect.
        """
        if not defects_list:
            return {}

        vote_counter = {}
        contributions = {}
        list_contributions = []

        # Process each defect dict with its index.
        for idx, defect in enumerate(defects_list):
            if defect is None or self.is_nan(defect):
                continue
            for key, value in defect[0].items():    # defect is a list of dict
                # Skip if the value is NaN or None.
                if value is None or self.is_nan(value):
                    continue
                if key not in vote_counter:
                    vote_counter[key] = Counter()
                    contributions[key] = {"Yes": [], "No": []}
                vote_counter[key][value] += 1
                contributions[key][value].append(idx)

        # Determine majority vote and flag indices for each defect key.
        majority_defects = {}
        for key, counter in vote_counter.items():
            yes_votes = counter.get('Yes', 0)
            no_votes = counter.get('No', 0)
            if yes_votes > no_votes:
                final_vote = 'Yes'
            else:
                final_vote = 'No'
            majority_defects[key] = final_vote
            list_contributions.append(contributions[key].get(final_vote, []))
            # {
            #     "decision": final_vote,
            #     "indices": contributions[key].get(final_vote, [])
            # }
        
        return majority_defects, list_contributions
    
    # decide on multiple detections on the same image
    def majority_df(self,
                    overall_df):
        # Find duplicates and non-duplicates
        screened = overall_df[overall_df.duplicated(subset=['image', 'asset_type'], keep=False)]
        not_dup = overall_df[~overall_df.isin(screened)]
        new_df = pd.DataFrame(columns=overall_df.columns)

        # For each asset type with duplicates, in each image file find the majority vote and the contributing indices
        dup_assets = screened['asset_type'].unique()
        for asset in dup_assets:
            this_asset = screened[screened['asset_type']==asset]
            unique_files = this_asset['image'].unique()
            for file in unique_files:
                this_file = this_asset[this_asset['image']==file]
                defects_list = this_file['defects'].tolist()

                # find the majority vote and assemble the descriptions
                majority_vote, list_contributions = self.majority_vote_defects(defects_list)
                contrib_ind = list(set(chain.from_iterable(list_contributions)))
                descriptions = this_file['descriptions'].tolist()
                new_des = [descriptions[i] for i in contrib_ind]

                # change the defects and descriptions of the first row of this_file
                this_file.at[this_file.index[0], 'defects'] = [majority_vote]
                this_file.at[this_file.index[0], 'descriptions'] = new_des
                
                # stick it to a new df
                row_to_append = this_file.iloc[[0]].dropna(axis=1, how='all')
                new_df = pd.concat([new_df, row_to_append], ignore_index=True)    # only keep the edited first row of this_file
        
        new_overall_df = pd.concat([not_dup, new_df], ignore_index=True).sort_values(by=['asset_type', 'image'])
        new_overall_df = new_overall_df.dropna(axis=0, how='all').reset_index(drop=True)

        return new_overall_df
    
    def extract_veg_defects(self,
                           veg_df):
        veg_keep = veg_df[(veg_df['overgrown_start'] == True) | (veg_df['overgrown_end'] == True)]
        veg_keep.loc[:, 'asset_type'] = 'vegetation'
        veg_keep.loc[:, 'defects'] = veg_keep.apply(lambda x: {'Overgrown': 'Yes'}, axis=1)
        veg_keep = veg_keep.drop(columns=['image_id', 'start_point', 'end_point', 'overgrown_start', 'overgrown_end'])
        veg_keep = veg_keep.rename(columns={'file_name': 'image'})
        # veg_keep['descriptions'] = veg_keep['comment']    # to be added when the extent of overgrowing is measured
        return veg_keep
    
    def add_geolocations(self,
                         overall_df,
                         georef_df):
        overall_df['pano_key'] = overall_df['image'].str.extract(r'(pano_\d+_\d+)')
        overall_df = pd.merge(overall_df, georef_df[['file_name', 'projectedX[m]', 'projectedY[m]', 'projectedZ[m]']],
                            left_on='pano_key', right_on='file_name', how='left')
        overall_df = overall_df.drop(columns=['pano_key', 'file_name'])
        overall_df = overall_df.rename(columns={'image': 'file_name'})
        return overall_df

class AggPvmt:
    def __init__(self):
        pass    

    def readPolygon(self,
                    filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            data_feature = data['features']
            at_accum = []
            for feature in data_feature:
                at = feature['attributes']
                try:    # in case the json file doesn't have the desired geometry (e.g. exports from vertices)
                    at['geometry'] = feature['geometry']['rings']       # Assume all features are polygons
                except:
                    pass
                at_accum.append(at)
            df = pd.DataFrame(at_accum)
        return df

    # functions
    def rot_angle(self,
                  x):
        angle = math.atan((x[-2][1]-x[0][1])/(x[-2][0]-x[0][0]))*180/np.pi
        return angle

    def scale(self, lengths, pnt, desired_size):
        cal_width = math.sqrt((pnt[0][3][0]-pnt[0][0][0])**2 + (pnt[0][3][1]-pnt[0][0][1])**2)
        cal_height = math.sqrt((pnt[0][1][0]-pnt[0][0][0])**2 + (pnt[0][1][1]-pnt[0][0][1])**2)
        
        if cal_width/cal_height < 0.9 or cal_width/cal_height >1.1:       # if the calculated width is more than 10% different from the height
            x_scale = desired_size[0] / cal_width
            y_scale = desired_size[1] / cal_height
        else:                                # more likely to be a good square
            # check against the provided side length
            side_length_check = lengths / 4
            if cal_width/side_length_check < 0.9:         # if the calculated side length is less than 90% of side length/4
                print(f"Side length error of {cal_width/side_length_check:.2f}")
            x_scale = desired_size[0] / cal_width
            y_scale = x_scale
        
        return x_scale, y_scale    # in pixels per metre

    def rotate_vertices(self, vertices, angle):
        angle = math.radians(angle)
        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        vertices = np.array(vertices)
        vertices = np.dot(vertices, rot_matrix)
        rounded_vertices = np.round(vertices, 3)
        return rounded_vertices

    def scale_geom(self, georef_df):
        # scale the centres, area, and segmentation
        for j in range(len(georef_df['geometry'])):
            # calculate the scale of images
            lengths, pnt, desired_size = georef_df.loc[j, ['Shape_Length', 'geometry', 'desired_size']]
            x_scale, y_scale = self.scale(lengths, pnt, desired_size)
            georef_df.loc[j, ['x_scale', 'y_scale']] = x_scale, y_scale
            # find image origin
            origin = (pnt[0][1][0], pnt[0][1][1])
            # scale the centre
            centre = np.array(georef_df.loc[j, 'centre'])/np.array(x_scale, y_scale)
            scaled_centre = self.rotate_vertices([centre], georef_df.loc[j, 'rotation']) + origin
            # scale the area
            scaled_area = round(georef_df.loc[j, 'area']/(x_scale*y_scale),4)
            # scale the segmentation
            segmentation = georef_df.loc[j, 'segmentation'][0]
            scaled_segmentation = [self.rotate_vertices(np.array(segmentation[i:i+2])/np.array(x_scale, y_scale), georef_df.loc[j, 'rotation']) + origin
                                    for i in range(0, len(segmentation), 2)]    # when iterating, the points are first scaled, then rotated, then translated by the origin
            # put them back in georef_df
            georef_df.loc[j, 'projectedX[m]'] = scaled_centre[0][0]
            georef_df.loc[j, 'projectedY[m]'] = scaled_centre[0][1]
            georef_df.loc[j, 'descriptions'] = [{'area': scaled_area, 'segmentation': scaled_segmentation}]

        # clean the dataframe
        georef_df = georef_df[['file_name', 'asset_type', 'defects', 'descriptions', 'projectedX[m]', 'projectedY[m]']]
        return georef_df