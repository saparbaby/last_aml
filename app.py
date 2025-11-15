import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 0

import streamlit as st
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

class MLPv2(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.res = nn.Linear(dim_in, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        h = h + self.res(x)
        return self.out(h)


class CrossAttentionGRU(nn.Module):
    def __init__(self, emb_dim=384, hidden=128, heads=4):
        super().__init__()
        self.gru = nn.GRU(
            emb_dim,
            hidden,
            batch_first=True,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden * 2,
            num_heads=heads,
            batch_first=True
        )
        fused_dim = hidden * 2 * 4 + 3
        self.mlp = MLPv2(fused_dim)

    def encode(self, x):
        out, _ = self.gru(x)
        return self.norm(out)

    def forward(self, r, j, skillfit, flag):
        r = r.view(1, 1, -1)
        j = j.view(1, 1, -1)
        r_enc = self.encode(r)
        j_enc = self.encode(j)
        attn, _ = self.cross_attn(r_enc, j_enc, j_enc)
        r_f = (r_enc + attn).mean(dim=1)
        j_f = (j_enc + attn).mean(dim=1)
        cosine = nn.functional.cosine_similarity(r_f, j_f, dim=1).unsqueeze(1)
        feats = torch.cat([
            r_f,
            j_f,
            torch.abs(r_f - j_f),
            r_f * j_f,
            cosine,
            skillfit.unsqueeze(1),
            flag.unsqueeze(1)
        ], dim=1)
        return self.mlp(feats)


@st.cache_resource
def load_models():
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    encoder = AutoModel.from_pretrained(name)
    model = CrossAttentionGRU()
    state = torch.load("model_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return tokenizer, encoder, model


tokenizer, encoder, model = load_models()


def encode_texts(texts):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = encoder(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        attention_mask = attention_mask.to(token_embeddings.device)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        sentence_embs = summed / counts
        sentence_embs = nn.functional.normalize(sentence_embs, p=2, dim=1)
    return sentence_embs.numpy()


SKILLS = {
    "python","java","javascript","c++","sql","nosql","git","linux",
    "docker","kubernetes","ml","nlp","pandas","numpy","scikit-learn",
    "pytorch","tensorflow","react","node","flask","django","fastapi",
    "aws","azure","gcp","data analysis","machine learning","deep learning"
}

def extract_skills(text):
    text = text.lower()
    return [s for s in SKILLS if s in text]


LABELS = {0: "❌ No Fit", 1: "⚙️ Partial Fit", 2: "✅ Good Fit"}


def predict(res_text, job_text):
    r_emb = encode_texts([res_text])[0]
    j_emb = encode_texts([job_text])[0]
    r_emb_t = torch.tensor(r_emb, dtype=torch.float32)
    j_emb_t = torch.tensor(j_emb, dtype=torch.float32)
    skills_r = set(extract_skills(res_text))
    skills_j = set(extract_skills(job_text))
    matched_set = skills_r & skills_j
    missing_set = skills_j - skills_r
    matched = len(matched_set)
    skillfit = torch.tensor([float(matched)], dtype=torch.float32)
    flag = torch.tensor([1.0 if matched > 0 else 0.0], dtype=torch.float32)

    with torch.no_grad():
        logits = model(r_emb_t, j_emb_t, skillfit, flag)
        probs = torch.softmax(logits, dim=1)[0].numpy()

    p_no, p_partial, p_good = probs
    if p_good >= 0.65 and p_good - max(p_no, p_partial) >= 0.10:
        pred = 2
    elif p_no >= 0.65 and p_no - max(p_partial, p_good) >= 0.10:
        pred = 0
    else:
        pred = 1
    return pred, probs, matched_set, missing_set


st.set_page_config(page_title="Resume ↔ Job Match Scorer", layout="wide")

st.title("🔍 Resume ↔ Job Match Scorer (MiniLM + CrossAttention GRU)")

col1, col2 = st.columns(2)
with col1:
    res_text = st.text_area("📝 Resume Text", height=300)
with col2:
    job_text = st.text_area("💼 Job Description", height=300)

if st.button("🔎 Evaluate Match"):
    if not res_text.strip() or not job_text.strip():
        st.error("❗ Please enter both resume and job description.")
    else:
        pred, probs, matched, missing = predict(res_text, job_text)
        st.subheader("📌 Match Result:")
        st.write(f"### {LABELS[pred]}")
        st.subheader("📊 Probabilities:")
        st.write(f"No Fit: {probs[0]:.3f}")
        st.write(f"Partial Fit: {probs[1]:.3f}")
        st.write(f"Good Fit: {probs[2]:.3f}")
        st.subheader("🧩 Skills Analysis:")
        st.write("**Matched Skills:**", ", ".join(sorted(matched)) if matched else "—")
        st.write("**Missing Required Skills:**", ", ".join(sorted(missing)) if missing else "—")
