from typing import Dict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class NLPManager:
    def __init__(self):
        # initialize the model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForQuestionAnswering.from_pretrained("models/nlp_best").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("models/nlp_tokenizer")

        self.words_to_numbers = {
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'niner': '9',
        }

        self.questions = [["heading", "What is the heading to face"], ["target", "What is the target to counter"], ["tool", "What is the tool to be used"]]

        pass

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        output = {}
        for question in self.questions:
            key = question[0]
            qn = question[1]
            inputs = self.tokenizer(qn, context, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start_index = torch.argmax(outputs.start_logits)
            answer_end_index = torch.argmax(outputs.end_logits)

            predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            ans = self.tokenizer.decode(predict_answer_tokens)
            if key == "heading":
                ans = ans.split(' ')
                temp = []
                for word in ans:
                    if word in self.words_to_numbers.keys():
                        temp.append(self.words_to_numbers[word])
                    else:
                        temp.append(word)
                translated_ans = ''.join(temp)
                output[key] = translated_ans
            elif key == "tool" and " - " in ans:
                output[key] = ans.replace(' - ', '-')
            else:
                output[key] = ans
        return output
