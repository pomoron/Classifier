from retrying import retry
import re, csv, os
import pandas as pd
from google import genai
import config.paths as paths

# Define a retry strategy
@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=10)
def send_message_with_retry(chat, message):
    response = chat.send_message(message)
    if 'error' in response:
        raise Exception(response['error'])
    return response

# Extract the data using regular expressions
class vlmCleanQ2:
    def __init__(self,
                 num_trial,
                 result_dir,
                 result_header,
                 ans_header,
                 print_header: list,
                 ):
        self.num_trial=num_trial
        self.result_dir=result_dir
        self.result_header=result_header
        self.ans_header=ans_header
        self.print_header=print_header

    def extract_defects(self, text):
        defects = {}
        bullet_points = re.findall(r'\* (.*?): (Yes|No)', text)
        for point in bullet_points:
            key, value = point
            key = self.extract_category(key)
            defects[key.strip()] = value.strip()
        return defects

    # In case you have many variations of a category 
    def extract_category(self, entry):
        if 'graffiti' in entry.lower():
            return 'Graffiti'
        elif 'vandalism' in entry.lower():
            return 'Vandalism'
        elif 'fading' in entry.lower():
            return 'Fading'
        elif 'deformation' in entry.lower():
            return 'Deformation'
        elif 'dirt' in entry.lower():
            return 'Dirt'
        elif 'rusting' in entry.lower():
            return 'Rusting'
        # new categories combined into preset ones
        elif 'peeling' in entry.lower():
            return 'Fading'
        elif 'stickers' in entry.lower():
            return 'Vandalism'
        else:
            return 'Other' 

    def extract_one_defect(self, 
                           text, 
                           the_defect='Toppled'):
        defects = {}
        bullet_points = re.findall(r'(Yes|No)', text, re.MULTILINE)
        try:
            defects[the_defect]=bullet_points[0]
            return defects
        except:
            return False

    def det_clean_q2(self):
        for n in range(self.num_trial):
            ans_df = pd.read_csv(os.path.join(self.result_dir,f'{self.result_header}_{n}.csv'))
            image_list = ans_df['image'].tolist()
            q2_list = ans_df['q2'].tolist()
            print_df = pd.DataFrame(columns=self.print_header)
            if len(self.print_header) <= 2:
                one_defect = True
            else:
                one_defect = False
            # print_df = pd.DataFrame(columns=['Image','Graffiti', 'Vandalism', 'Fading', 'Deformation', 'Dirt', 'Rusting', 'Collapse'])
            # print_df = pd.DataFrame(columns=['Image', 'Vandalism', 'Fading', 'Peeling', 'Rusting', 'Other'])	# For barrier
            # print_df = pd.DataFrame(columns=['Image', 'Toppled'])	# For lamp post

            # Extracted data
            for ind, text in enumerate(q2_list):
                if one_defect:
                    defects = self.extract_one_defect(text)
                    if defects == False:
                        continue
                else:
                    defects = self.extract_defects(text)

                defects['Image'] = image_list[ind]
                new_df = pd.DataFrame(defects, index=[0])
                print_df = pd.concat([print_df, new_df], ignore_index=True)

            print_df.to_csv(os.path.join(self.result_dir, f'{self.ans_header}_{n}.csv'), index=False)