from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # <-- 1. Tambahkan import ini
from typing import List
import os
import random
import torch
import re
import string
import pandas as pd
import wget
from groq import Groq

app = FastAPI()

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# ==========================================
# 1. SETUP PATH DIREKTORI (BACKEND & FRONTEND)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(current_dir) 
FRONTEND_DIR = os.path.join(BASE_DIR, "Frontend")

# ==========================================
# 2. SETUP KAMUS & FUNGSI PREPROCESSING
# ==========================================
url_kamus = "https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/kamus_singkatan.csv"
file_kamus = os.path.join(current_dir, "kamus_singkatan.csv")

if not os.path.exists(file_kamus):
    print("Sedang mengunduh kamus singkatan...")
    wget.download(url_kamus, out=file_kamus)

kamus_df = pd.read_csv(file_kamus, header=None, names=['singkatan', 'baku'])
slang_dict = pd.Series(kamus_df.baku.values, index=kamus_df.singkatan.values).to_dict()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    words = text.split()
    normalized = [slang_dict[word] if word in slang_dict else word for word in words]
    return ' '.join(normalized)


# ==========================================
# 3. SETUP MODEL NLP (INDOBERT) & PYDANTIC
# ==========================================
FINE_TUNED_MODEL_PATH = os.path.join(current_dir, "IndoBERT") 
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
model.eval() 

class ChatMessage(BaseModel):
    sender: str
    text: str

class PesanPengguna(BaseModel):
    teks: str
    history: List[ChatMessage] = []


# ==========================================
# 4. KONFIGURASI GROQ API (LLAMA 3)
# ==========================================
# os.environ["GROQ_API_KEY"] = "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"
# client = Groq()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_bot_response(emotion, user_input_text, chat_history):
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah seorang konselor chatbot psikologi yang sangat berempati, hangat, dan suportif. "
                "Gunakan bahasa Indonesia yang santai, ramah, layaknya teman dekat (gunakan panggilan 'kamu' atau 'kak'). "
                "Jawab dengan singkat, padat, dan menenangkan (maksimal 3-4 kalimat). "
                "Di akhir jawaban, berikan SATU pertanyaan lanjutan yang nyambung dengan curhatan user."
            )
        }
    ]
    
    if chat_history:
        for msg in chat_history:
            sender = msg.sender
            text = msg.text
            if not text:
                continue
            role = "assistant" if sender in ["bot", "assistant"] else "user"
            messages.append({"role": role, "content": text})
            
    prompt_terbaru = f"[Konteks emosi saat ini: {emotion}] {user_input_text}"
    messages.append({"role": "user", "content": prompt_terbaru})
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant", 
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ ERROR GROQ CRASH: {e}")
        return f"⚠ SYSTEM ERROR PADA BACKEND GROQ: {str(e)}"


# ==========================================
# 5. ENDPOINTS / ROUTE
# ==========================================

# Endpoint Prediksi untuk API Chatbot kamu
@app.post("/predict")
async def predict_stress(pesan: PesanPengguna):
    teks_clean = preprocess_text(pesan.teks)
    teks_final = normalize_slang(teks_clean)

    inputs = tokenizer(teks_final, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    prediksi_id = torch.argmax(logits, dim=1).item()
    daftar_label = ['anger', 'fear', 'joy', 'love', 'sadness']
    label_prediksi = daftar_label[prediksi_id]
    
    bot_response_text = get_bot_response(label_prediksi, pesan.teks, pesan.history) 
    
    return {
        "input_asli": pesan.teks,
        "input_setelah_preprocessing": teks_final,
        "bot_response": bot_response_text, 
        "prediksi_label": label_prediksi,
        "confidence_score": torch.softmax(logits, dim=1).max().item()
    }

# 2. PENTING: Mount folder Frontend ditaruh paling bawah setelah route POST.
# Parameter html=True otomatis akan mencari file index.html ketika root "/" diakses,
# sekaligus membuka akses ke style.css, javascript, dll.
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"⚠ Peringatan: Folder Frontend tidak ditemukan di {FRONTEND_DIR}")