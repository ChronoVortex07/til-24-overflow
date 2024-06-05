import nltk
import clip

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
clip_model = clip.load("ViT-B/32")[0]