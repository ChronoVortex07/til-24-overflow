from transformers import WhisperForConditionalGeneration, WhisperProcessor

whisper_model = WhisperForConditionalGeneration.from_pretrained("models/asr_best")
whisper_processor = WhisperProcessor.from_pretrained("models/asr_processor")