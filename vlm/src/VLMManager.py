import re
import cv2
import numpy as np
from typing import List
from ultralytics import YOLOWorld
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class VLMManager:
    def __init__(self) -> None:
        # best.pt is zedong's best
        # best_shuce.pt is shuce's best but not used
        self.model = YOLOWorld("best.pt", verbose=False).to('cuda:0')
        
        self.stopwords = set(nltk.corpus.stopwords.words('english')) - {'and'}
        self.colors = {"red", "green", "blue", "yellow", "purple", "orange", "pink", 
                       "brown", "black", "white", "grey", "violet", "indigo", "cyan", 
                       "magenta", "aqua"}

    def XYXY_to_LTWH(self, box: List[int]) -> List[int]:
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]
    
    def clean_caption(self, caption: str) -> str:
        caption = re.sub(r'[^\w\s,]', ' ', caption).strip()
        tokens = word_tokenize(caption)
        tagged_tokens = pos_tag(tokens)
        
        filtered_tokens = []
        for i, (word, pos) in enumerate(tagged_tokens):
            lw = word.lower()
            if pos.startswith(('JJ', 'NN', 'CC', ',')) or lw in self.colors:
                if i > 0:
                    if tagged_tokens[i-1][0] == "silver" and lw == "fighter":
                        filtered_tokens.append("silver")
                filtered_tokens.append(lw)
                
        result_tokens = ' '.join(word for word in filtered_tokens if word not in self.stopwords)
        return re.sub(r'\s+,', ',', result_tokens).strip()
    
    def identify(self, image: bytes, caption: str) -> List[int]:
        image_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        self.model.set_classes([self.clean_caption(caption)])
        results = self.model(image, conf=0, iou=1, imgsz=1536, device="cuda:0", verbose=False, augment=True)
        total_boxes = [list(box) + [conf] for result in results for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        bbox = list(map(int, max(total_boxes, key=lambda x: x[-1])[:4])) if total_boxes else [950, 300, 100, 50]
        return self.XYXY_to_LTWH(bbox)
