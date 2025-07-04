import base64
import io
import modal
import numpy as np
import requests
import torch.nn as nn
import torchaudio.transforms as T
import torch
from pydantic import BaseModel
import soundfile as sf
import librosa

from model import AudioCNN


app = modal.App("audio-cnn-inference")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["libsndfile1"])
    .add_local_python_source("model")
)

model_volume = modal.Volume.from_name("esc-model")


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025,
            ),
            T.AmplitudeToDB(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


class InferenceRequest(BaseModel):
    audio_data: str


@app.cls(
    image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15
)
class AudioClassifier:
    @modal.enter()
    def load_model(self):
        print("Loading models on enter")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load("/models/best_model.pth", map_location=self.device)
        self.classes = checkpoint["classes"]

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):

        # What we are doing for this example
        # Frontend uploads file directly to this
        # inference endpoint
        # Frontend -> send file directly -> inference endpoint

        audio_bytes = base64.b64decode(request.audio_data)

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 44100:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=44100
            )

        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        with torch.no_grad():
            output = self.model(spectrogram)
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(
                output, dim=1
            )  # dim=0 -> batch, dim=1 -> class
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)

            predictions = [
                {"class": self.classes[idx.item()], "confidence": prob.item()}
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        response = {"predictions": predictions}

        return response


@app.local_entrypoint()
def main():
    audio_data, sample_rate = sf.read("sample_audio/chirping-birds.wav")
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, 22050, format="WAV")
    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_base64}
    server = AudioClassifier()
    url = server.inference.get_web_url()
    if url is not None:
        response = requests.post(url, json=payload)
    else:
        raise ValueError("Inference endpoint URL is None.")
    response.raise_for_status()

    result = response.json()
    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f" -{pred['class']} {pred['confidence']:0.2%}")
