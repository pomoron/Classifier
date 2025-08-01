from .functions.ask2q import ask2q
from .functions.call import vlmCleanQ2
import config.paths as paths
from modules.tools.detcrop import DetAndCrop
from datetime import datetime
import torch

# for rusty barriers
def crop_barrier():
    categories = ['barrier']
    dc = DetAndCrop(categories = categories,
                 input_dir = paths.barrier_indir,
                 input_header = 'left',
                 output_dir = paths.cropbarrier_dir,
                 output_img_header = '')
    dc.det()
    del dc
    torch.cuda.empty_cache()

def det_barrier(num_trial=1):
    image_dir = paths.cropbarrier_dir
    output_dir = paths.barrier_outdir
    question_1 = 'What defect does this barrier contain?'
    question_2 = 'Does the barrier have vandalism (such as stickers torn or untorn), fading (and peeling) or rusting?'
    system_message = 'Answer one by one in bullet points, with <defect, without explanations in parenthesis>: Yes/No <with no extra descriptions behind>. Additional justifications shall be given in a separate line.'
    num_trial = num_trial

    ask_bar = ask2q(image_dir, output_dir, question_1, question_2, system_message, num_trial)
    ask_bar.run()

def det_barrier_clean(num_trial=1):
    q2_header = ['Image', 'Vandalism', 'Fading', 'Rusting', 'Other']
    bar_q2 = vlmCleanQ2(
        num_trial=num_trial,
        result_dir=paths.barrier_outdir,
        result_header='gemini_result',
        ans_header='gemini_q2',
        print_header=q2_header
    )
    bar_q2.det_clean_q2()

def run_bar(num_trial=paths.bar_num_trial):

    print("Detecting and cropping barriers")
    crop_barrier()

    print(f"Detecting rust on barriers for {num_trial} trials")
    det_barrier(num_trial)
    det_barrier_clean(num_trial)


if __name__ == '__main__':
    run_bar(num_trial=paths.bar_num_trial)


