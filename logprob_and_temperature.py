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
avg = []
for i in range(5):
    result = whisper.decode(model, mel, options)
    texts.append(result.text)
    avg.append(np.exp(result.avg_logprob))

for it in range(len(texts)):
    print(f"Предсказание: {texts[it]}")
    print(f"Экспонента от логарифма правдоподобия {avg[it]}")

max_index = avg.index(max(avg))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
similarity_matrix = cosine_similarity(tfidf_matrix)
average_similarity = np.mean(similarity_matrix)

print(" ")
print(f"Наивысшее значение от экспоненты avg_logprob {avg[max_index]}")
print(" ")
print(f"Схожесть предсказаний: {average_similarity}")
print(" ")
print(f"Предсказание: {texts[max_index]}")
print(f"Уверенность модели: {(average_similarity + avg[max_index]) / 2}")