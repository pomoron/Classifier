import os
from .project import gemini_project, gemini_location    # Edit your GCP project details in project.py

home = os.path.expanduser("~")

# File directories
input_dir = os.path.join(home, "Classifier/input")
output_dir = os.path.join(home, "Classifier/output")
intermediate_dir = os.path.join(output_dir, "intermediate")
os.makedirs(intermediate_dir, exist_ok=True)
# Placed for the time being - to be added to the input directory
pavement_indir = os.path.join(home, 'yolov8/datasets/syna12-o/test/images')
sign_indir = os.path.join(input_dir, 'sign')
detveg_indir = os.path.join(input_dir, 'overgrown_veg')
barrier_indir = os.path.join(home, "PTrans/panoconvert/mass_barrier/inpainted")
obstruction_indir = os.path.join(input_dir, 'obs_cropped')
# georeferences
georef_dir = os.path.join(input_dir, 'georef')
georef_pano = os.path.join(georef_dir, 'reference.csv')
georef_pvmt = os.path.join(georef_dir, 'image_polygon.json')

# Output file locations
pavement_outdir = os.path.join(output_dir, "pavements")
sign_outdir = os.path.join(output_dir, "signs")
barrier_outdir = os.path.join(output_dir, "barriers")
obstruction_outdir = os.path.join(output_dir, "obstructions_cropped")
detveg_dir = os.path.join(output_dir, "overgrown_veg")
detveg_final_fn = os.path.join(detveg_dir, "overgrown_list2.csv")

# Intermediate file locations
# Pavement detection
# Sign detection
sign_num_trial = 5
cropsign_dir = os.path.join(intermediate_dir, "signs")
cropsign_json = os.path.join(cropsign_dir, "crop_list.json")
# Rusty barrier detection
bar_num_trial = 5
cropbarrier_dir = os.path.join(intermediate_dir, "barriers")
# Obstruction detection
obs_num_trial = 5
# Overgrown vegetation detection
detveg_fn = os.path.join(intermediate_dir, "seg_list.json")
detveg_want_plot = False
detveg_visdir = os.path.join(output_dir, "overgrown_veg/vegvis")
detveg_line_fn = os.path.join(intermediate_dir, "accum_line.csv")
detveg_overgrown_vis = False
detveg_overgrown_visdir = os.path.join(output_dir, "overgrown_veg/overgrown_vis")

# Detection Utilities
sam_checkpoint = os.path.join(home,"Syndata/sam2/checkpoints/sam2.1_hiera_large.pt")
sam_cfg = os.path.join(home,"Syndata/sam2","/configs/sam2.1/sam2.1_hiera_l.yaml")
gemini_project = gemini_project
gemini_location = gemini_location
depth_model_name = "depth-anything/Depth-Anything-V2-large-hf"
# Add your pavement weights here
pvmt_weights = 'modules/pavement/weights/synreal230-v11-3.pt'
pvmt_extra_weights = 'modules/pavement/weights/syna12-o-0.pt'