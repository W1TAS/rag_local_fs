from sentence_transformers import SentenceTransformer
import os

# Имя модели
model_name = "ai-forever/sbert_large_nlu_ru"

# Локальная папка для модели (создастся автоматически)
local_path = "./models/sbert_large_nlu_ru"

# Скачиваем модель
print(f"Downloading {model_name} to {local_path}...")
model = SentenceTransformer(model_name)
model.save(local_path)
print(f"Model saved to {local_path}. Size: {os.path.getsize(local_path + '/modules.json') / (1024*1024):.2f} MB (example file)")

# Тестируем загрузку локально
print("Testing local load...")
local_model = SentenceTransformer(local_path)
print("Success! Model loaded offline.")