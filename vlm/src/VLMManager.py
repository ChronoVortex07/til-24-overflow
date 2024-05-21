import re
import cv2
import nltk
import numpy as np
from typing import List
from ultralytics import YOLOWorld

nltk.download('stopwords')

class VLMManager:
    def __init__(self, verbose=0) -> None:
        self.verbose = verbose
        model = YOLOWorld("vlm_album_large.pt", verbose=False)
        model.to('cuda')
        self.model = model
        self.exact_match_count = 0
        self.inferior_match_count = 0
        self.no_match_count = 0
        self.no_initial_match_count = 0
        
    def XYXY_to_LTWH(self, box: List[int]) -> List[int]:
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]
    
    def clean_caption(self, caption: str) -> str:
        # This function cleans a caption by removing stopwords and special characters.
        caption = re.sub(r'[^\w\s]', '', caption)
        return ' '.join([word for word in caption.split() if word.lower() not in nltk.corpus.stopwords.words('english')])
    
    def identify(self, image: bytes, caption: str) -> List[int]:
        image_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        self.model.set_classes([self.clean_caption(caption)])

        results = self.model(image, conf=0.001, verbose=False)

        total_boxes = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                total_boxes.append(list(box) + [conf])
        if total_boxes:
            bbox = list(map(int, max(total_boxes, key=lambda x: x[-1])[:4]))
            bbox = self.XYXY_to_LTWH(bbox)
        else:
            bbox = [0, 0, 0, 0]
        return bbox
    
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    vlm_manager = VLMManager(verbose=2)
    with open(r"C:\Users\zedon\Documents\GitHub\til-24-overflow\data\images\image_0.jpg", "rb") as f:
        image = f.read()
    bbox = vlm_manager.identify(image, "red, white, and blue light aircraft on runway")
    x1, y1, w, h = bbox
    image_array = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()
    print(bbox)