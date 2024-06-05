from transformers import AutoModelForQuestionAnswering, AutoTokenizer

distilbert_model = AutoModelForQuestionAnswering.from_pretrained("models/nlp_best")
distilbert_tokenizer = AutoTokenizer.from_pretrained("models/nlp_tokenizer")