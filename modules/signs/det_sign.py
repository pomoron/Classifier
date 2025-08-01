import config.paths as paths
from modules.tools.detcrop import DetAndCrop
from modules.misc_vlm.functions.ask2q import ask2q
from modules.misc_vlm.functions.call import vlmCleanQ2
from datetime import datetime
import torch

def crop_sign():
    categories = ['sign']
    dc = DetAndCrop(categories = categories,
                 input_dir = paths.sign_indir,
                 input_header = 'front',
                 output_dir = paths.cropsign_dir,
                 output_img_header = '_crop',)
                #  output_fn = paths.cropsign_json,)
    dc.det_with_descriptions()
    # dc.det()
    del dc
    torch.cuda.empty_cache()

def det_sign(num_trial=1):
    image_dir = paths.cropsign_dir
    output_dir = paths.sign_outdir
    question_1 = 'What defect does this sign contain?'
    question_2 = 'Does the sign have graffiti (including drawings and paint), vandalism (such as stickers torn or untorn, bullet holes, scratches), fading (and peeling), deformation (broken, distorted, cracked), dirt (dirty, grease), rusting or collapse?'
    system_message = 'Answer one by one in bullet points, with <defect, without explanations in parenthesis>: Yes/No <with no extra descriptions behind>. Additional justifications shall be given in a separate line.'
    num_trial = num_trial

    ask_sign = ask2q(image_dir, output_dir, question_1, question_2, system_message, num_trial)
    ask_sign.run()

def det_sign_clean(num_trial=1):
    q2_header = ['Image', 'Graffiti', 'Vandalism', 'Fading', 'Deformation', 'Dirt', 'Rusting', 'Collapse']
    sign_q2 = vlmCleanQ2(
        num_trial=num_trial,
        result_dir=paths.sign_outdir,
        result_header='gemini_result',
        ans_header='gemini_q2',
        print_header=q2_header
    )
    sign_q2.det_clean_q2()

def run_sign_demo(num_trial=paths.sign_num_trial):
    start_time = datetime.now()
    print("Detecting and cropping signs")
    crop_sign()
    end_crop_time = datetime.now()
    duration_crop = end_crop_time - start_time
    print(f"Duration for cropping: {duration_crop}")
    print(f"Detecting sign defects for {num_trial} trials")
    det_sign(num_trial)
    det_sign_clean(num_trial)
    end_time = datetime.now()
    duration_det = end_time - end_crop_time
    print(f"Duration for detection: {duration_det}")

def run_sign(num_trial=paths.sign_num_trial):

    # print("Detecting and cropping signs")
    # crop_sign()

    print(f"Detecting sign defects for {num_trial} trials")
    det_sign(num_trial)
    det_sign_clean(num_trial)

if __name__ == '__main__':
    run_sign(num_trial=paths.sign_num_trial)