import io
import soundfile as sf
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

class ASRManager:
    def __init__(self):
        # initialize the model here
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        pass

    def transcribe(self, raw_bytes: bytes) -> str:
        audio_bytes , sampling_rate = sf.read(io.BytesIO(raw_bytes))
        # perform ASR transcription
        inputs = self.processor(audio_bytes, sampling_rate=sampling_rate, return_tensors="pt")
        generated_ids = self.model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription[0]
