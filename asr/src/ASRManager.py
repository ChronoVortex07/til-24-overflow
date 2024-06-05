import io
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained("models/asr_best").to(self.device)
        self.processor = WhisperProcessor.from_pretrained("models/asr_processor", language="English", task="transcribe")
        pass

    def transcribe(self, raw_bytes: bytes) -> str:
        audio_bytes, sampling_rate = sf.read(io.BytesIO(raw_bytes))
        # perform ASR transcription
        inputs = self.processor(audio_bytes, sampling_rate=sampling_rate, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(inputs["input_features"])
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
