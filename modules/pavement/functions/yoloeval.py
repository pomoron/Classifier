from ultralytics import YOLO
import pandas as pd
import os
import yaml

# ---------------------------
# 250313: To be fixed in due course
# ---------------------------

class YOLOeval:
    def __init__(self, model_dir, test_dir, names, nc):
        self.model_dir = model_dir
        self.test_dir = test_dir
        self.names = names
        self.nc = nc
        self.yaml_fn = None

    def create_yaml(self, 
                    yaml_filename='testing.yaml'):
        """Creates a YAML file for evaluation.

        Args:
            yaml_filename (str): Filename for the generated YAML file.
        """
        # Remove '/datasets' from test_dir if present.
        cleaned_test_dir = self.test_dir.replace('/datasets', '')
        with open(yaml_filename, "w") as outfile:
            outfile.write(f'train: {cleaned_test_dir}/run/images\n')
            outfile.write(f'val: {cleaned_test_dir}/test/images\n')
            outfile.write(f'nc: {self.nc}\n')
            outfile.write(f'names: {self.names}\n')
        print(f'{yaml_filename} prepared!')
        self.yaml_fn = yaml_filename

    def eval_export_excel(self, excel_file_path):
        """Evaluates the model on the YAML file and exports evaluation results to Excel.

        Args:
            excel_file_path (str): Path to the Excel file where results will be saved.
        """
        if not self.yaml_fn:
            raise ValueError('YAML file not created. Please run create_yaml first.')
        # Load the model from the weights directory
        model_path = os.path.join(self.model_dir, 'weights', 'best.pt')
        model = YOLO(model_path)
        # Evaluate the model with the given yaml file
        # ------------------------------
        # TRY NOT TO USE YAML FILE TO DIRECT THE MODEL TO THE TEST DIRECTORY
        # ------------------------------
        metrics = model.val(data=self.yaml_fn, iou=0.5)
        
        # Export each evaluation curve to a separate sheet in Excel
        for i in range(len(metrics.curves_results)):
            x_key = metrics.curves_results[i][2]
            # Create column names from metrics.names values
            y_key = [x for key, x in metrics.names.items()]
            x_value = metrics.curves_results[i][0]
            y_value = metrics.curves_results[i][1].T
            metrics_df = pd.DataFrame(y_value, columns=y_key)
            metrics_df.insert(0, x_key, x_value)
            excel_tab_name = metrics.curves[i]

            # Append or write new Excel file
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a' if os.path.exists(excel_file_path) else 'w') as writer:
                metrics_df.to_excel(writer, sheet_name=excel_tab_name, index=False)
        print(f"Evaluation results exported to {excel_file_path}")

# inputs
model_dir = './runs/segment/a14-o-spalling'
test_dir = './datasets/240906_sampling'
img_path = './datasets/a12-p/test/images'

# Read the training custom.yaml to get the number of classes and names
with open(r'./custom.yaml', "r") as file:
    data = yaml.safe_load(file)
nc = data['nc']
names = data['names']

for it in range(1,2):
    model = YOLO(f'{model_dir}_{str(it).zfill(2)}/weights/best.pt')      # trained pavement model

    # Write a good test_yaml for testing
    cleaned_test_dir = test_dir.replace('/datasets', '')
    with open(r'./testing.yaml', "w") as outfile:
        outfile.write(f'train: {cleaned_test_dir}/run_{str(it).zfill(2)}/images \n')
        outfile.write(f'val: {cleaned_test_dir}/test/images \n')
        # outfile.write('test: a14-p/test/images \n \n')
        outfile.write(f'nc: {nc} \n')
        outfile.write(f'names: {names} \n')
    print('testing.yaml prepared!')

    # # Run batched inference on a list of images
    # model.predict(img_path, save=True, imgsz=640, visualize=True)      # return a list of Results objects
    metrics = model.val(data='testing.yaml', conf=0.05, iou=0.5)

from ultralytics import YOLO
import pandas as pd
import numpy as np
import os

# inputs
model_dir = './runs/segment/tcr_semi20pc-0'
model = YOLO(f'{model_dir}/weights/best.pt')
yaml_fn = 'sample_custom.yaml'
excel_file_path = f'{model_dir}/metrics_best.xlsx'       # Define the path to the Excel file

metrics = model.val(data=yaml_fn, iou=0.5)

# in each m.curves_results[i] - 0: x_value, 1: y_values, 2: x_key, 3: y_key
for i in range(len(metrics.curves_results)):
    x_key = metrics.curves_results[i][2]
    y_key = [x for key, x in metrics.names.items()]
    x_value = metrics.curves_results[i][0]
    y_value = metrics.curves_results[i][1].T
    metrics_df = pd.DataFrame(y_value, columns=y_key)
    metrics_df.insert(0, x_key, x_value)
    excel_tab_name = metrics.curves[i]

    # Use ExcelWriter to create a new Excel file or append to an existing one
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a' if os.path.exists(excel_file_path) else 'w') as writer:
        metrics_df.to_excel(writer, sheet_name=excel_tab_name, index=False)