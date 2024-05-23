# !pip install git+https://github.com/openai/whisper.git 

import torch
import numpy as np
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = whisper.load_model("medium", device=device)
audio = whisper.load_audio("data_sber/sber_5408.wav")

audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
options = whisper.DecodingOptions(temperature = 0.5, language = "ru")

texts = []
for i in range(10):
    result = whisper.decode(model, mel, options)
    print(result.text)
    texts.append(result.text)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)
average_similarity = np.mean(similarity_matrix)

print(f"Средняя схожесть предложений: {average_similarity}")