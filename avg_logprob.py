# !pip install git+https://github.com/openai/whisper.git 

import torch
import numpy as np
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = whisper.load_model("medium", device=device)
audio = whisper.load_audio("data_sber/sber_5408.wav")

result = model.transcribe(audio)

for segment in result["segments"]:
    print(segment['text'])
    print(f"avg_logprob: {segment['avg_logprob']}")
    print(f"Вероятность правильного распознавания: {np.exp(segment['avg_logprob'])}")
    print(" ")