import re
import cv2
import nltk
import itertools
import numpy as np

from typing import List
from ultralytics import YOLOWorld
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

class VLMManager:
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.model = YOLOWorld("vlm_album_large.pt", verbose=(self.verbose == 2))
        self.all_labels = self.model.names
        
        self.vectorizer = TfidfVectorizer()
        
        self.exact_match_count = 0
        self.inferior_match_count = 0
        self.no_match_count = 0
        self.no_initial_match_count = 0

    def identify(self, image: bytes, caption: str) -> List[int]:
        image_array = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        caption = caption.lower().replace(' and', '').replace(',', '')
        
        
        # perform object detection with a vision-language model
        bbox = [0, 0, 0, 0] # x1, y1, w, h
        # self.model.set_classes(self.create_inferior_labels(caption))
        results = self.model(image, verbose=(self.verbose == 2))
        
        # if no objects are found, denoise the image and try again
        # this is only done as a last resort, as it is computationally expensive
        if len(results[0].boxes.xyxy.cpu()) == 0:
            self.no_initial_match_count += 1
            # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            self.model.set_classes(list(self.all_labels.values()))
            results = self.model(image, verbose=(self.verbose == 2))
            
        if len(results[0].boxes.xyxy.cpu()) > 0:
            labels = [self.all_labels[cls] for cls in results[0].boxes.cls.cpu().numpy()]
            similarity = self.cosine_similarity(caption, labels)
            print(similarity)
            best_index = similarity.index(max(similarity))
            bbox = self.XYXY_to_XYWH(results[0].boxes.xyxy.cpu()[best_index].tolist())
        else:
            self.no_match_count += 1
        return bbox
    
    def create_inferior_labels(self, caption):
        # This function generates multiple inferior versions of a label by having different permutations of reduced adjectives for the label.
        # For example, a red cargo airplane can be labeled as a red cargo airplane, a red airplane, a cargo airplane, and an airplane.
        word_list = self.clean_caption(caption).split()
        
        # get all possible combinations of words
        inferior_labels = []
        for i in range(len(word_list), 0, -1):
            for subset in itertools.combinations(word_list, i):
                inferior_labels.append(' '.join(subset))
    
        
        return inferior_labels
    
    def clean_caption(self, caption: str) -> str:
        # This function cleans a caption by removing stopwords and special characters.
        caption = re.sub(r'[^\w\s]', '', caption)
        return ' '.join([word for word in caption.split() if word.lower() not in nltk.corpus.stopwords.words('english')])
    
    def cosine_similarity(self, caption: str, labels: List[str]) -> List[float]:
        # This function calculates the cosine similarity between a caption and a list of labels.
        # The caption is tokenized and vectorized using the TfidfVectorizer, and the cosine similarity is calculated.
        caption = self.clean_caption(caption)
        labels = [self.clean_caption(label) for label in labels]
        
        labels.append(caption)
        vectors = self.vectorizer.fit_transform(labels)
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        return similarity.tolist()[0]
    
    def XYXY_to_XYWH(self, box: List[int]) -> List[int]:
        x1, y1, x2, y2 = box
        return [x1, y1, x2 - x1, y2 - y1]
    
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