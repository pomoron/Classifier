from .functions.ask2q import ask2q
from .functions.call import vlmCleanQ2
import config.paths as paths

# for obstructions
def det_obs(num_trial=1):
    image_dir = paths.obstruction_indir
    output_dir = paths.obstruction_outdir
    question_1 = 'What defect can you see from the road?'
    # question_2 = 'Are there any obstructions visible on the road surface in this image?'
    question_2 = 'Carefully examine the road surface and immediate surroundings in the middle distance directly ahead. Identify and describe any objects or features that are not part of the normal road infrastructure (e.g., markings, pavement), not traffic and not the smoke or water splash that disturbs the images. Consider size, shape, and orientation in your description.'

    system_message = 'First answer yes/no, then state your justifications and observations with header "justification" in a new line'
    # system_message = 'Answer Yes/No <with no extra descriptions behind>. Additional justifications shall be given in a separate line.'
    num_trial = num_trial

    ask_obs = ask2q(image_dir, output_dir, question_1, question_2, system_message, num_trial)
    ask_obs.run()

def det_obs_clean(num_trial):
    q2_header = ['Image', 'Toppled']
    obs_q2 = vlmCleanQ2(
        num_trial=num_trial,
        result_dir=paths.obstruction_outdir,
        result_header='gemini_result',
        ans_header='gemini_q2',
        print_header=q2_header
    )
    obs_q2.det_clean_q2()

def run_obs(num_trial=paths.obs_num_trial):
    print(f"Running obstruction detection for {num_trial} trials")
    det_obs(num_trial)
    det_obs_clean(num_trial)

if __name__ == '__main__':
    run_obs(num_trial=paths.obs_num_trial)
