import os, glob
import pandas as pd
from PIL import Image
# from .call import send_message_with_retry, initGemini
from modules.tools.initdet import initGemini

class ask2q:
    def __init__(self,
                 image_dir,
                 output_dir,
                 question_1,
                 question_2,
                 system_message,
                 num_trial=1):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.question_1 = question_1
        self.question_2 = question_2
        self.system_message = system_message
        self.image_list = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        self.num_trial = num_trial
        
        # Create an instance of initGemini
        self.my_gemini = initGemini()
        self.client = self.my_gemini.client
        self.model_id = self.my_gemini.model_id

    def run(self):
        for n in range(self.num_trial):
            store_ans = []

            for image_file in self.image_list:
                ans_dict = {}
                print(f'Image: {os.path.basename(image_file)}')
                ans_dict['image'] = os.path.basename(image_file)
                image = Image.open(image_file)
                chat = self.client.chats.create(model=self.model_id)
                
                # add retry to allow backoff to cope with rate limit
                try:
                    answer1 = self.my_gemini.send_message_with_retry(chat, [self.question_1, image])
                    answer2 = self.my_gemini.send_message_with_retry(chat, self.question_2 + self.system_message)
                except Exception as e:
                    print(f"Failed to get response: {e}")
                    continue

                ans_dict['q1'] = answer1.text
                ans_dict['q2'] = answer2.text
                store_ans.append(ans_dict)


            ans_out = pd.DataFrame(store_ans)
            ans_file = os.path.join(self.output_dir, f'gemini_result_{n}.csv')
            ans_out.to_csv(ans_file, index=False)


