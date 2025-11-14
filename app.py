import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np

# ============================================================
# 1. Model definition (MUST MATCH TRAINING EXACTLY)
# ============================================================

class MLPv2(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim_in,256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.res = nn.Linear(dim_in,128)
        self.out = nn.Linear(128,3)

    def forward(self,x):
        h=self.fc1(x)
        h=self.fc2(h)
        h=h+self.res(x)   # residual
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

        # These dimensions MUST MATCH THE TRAINED MODEL
        fused_dim = hidden * 2 * 4 + 3   # = 1027
        self.mlp = MLPv2(fused_dim)

    def encode(self,x):
        out,_ = self.gru(x)
        return self.norm(out)

    def forward(self,r,j,skillfit,flag):
        # Force correct shape (1,1,emb_dim)
        r = r.view(1,1,-1)
        j = j.view(1,1,-1)

        r_enc = self.encode(r)
        j_enc = self.encode(j)

        attn,_ = self.cross_attn(r_enc,j_enc,j_enc)

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


# ============================================================
# 2. Load model + SBERT
# ============================================================

DEVICE = "cpu"

model = CrossAttentionGRU()
model.load_state_dict(torch.load("model_best.pt", map_location="cpu"))
model.eval()

sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ============================================================
# 3. Skills extraction
# ============================================================

SKILLS = {
    "python","java","javascript","c++","sql","nosql","git","linux",
    "docker","kubernetes","ml","nlp","pandas","numpy","scikit-learn",
    "pytorch","tensorflow","react","node","flask","django","fastapi",
    "aws","azure","gcp","data analysis","machine learning","deep learning"
}

def extract_skills(text):
    text = text.lower()
    return [s for s in SKILLS if s in text]


# ============================================================
# 4. Predict function
# ============================================================

LABELS = {0:"❌ No Fit", 1:"⚙️ Partial Fit", 2:"✅ Good Fit"}

def predict(res_text, job_text):
    r_emb = sbert.encode([res_text], normalize_embeddings=True)[0]
    j_emb = sbert.encode([job_text], normalize_embeddings=True)[0]

    r_emb_t = torch.tensor(r_emb, dtype=torch.float32).view(1,1,-1)
    j_emb_t = torch.tensor(j_emb, dtype=torch.float32).view(1,1,-1)

    skills_r = set(extract_skills(res_text))
    skills_j = set(extract_skills(job_text))

    matched = len(skills_r & skills_j)
    missing = len(skills_j - skills_r)

    skillfit = torch.tensor([matched], dtype=torch.float32)
    flag = torch.tensor([1.0 if matched > 0 else 0.0], dtype=torch.float32)

    with torch.no_grad():
        logits = model(r_emb_t, j_emb_t, skillfit, flag)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return pred, probs.numpy(), skills_r & skills_j, skills_j - skills_r


# ============================================================
# 5. Streamlit UI
# ============================================================

st.set_page_config(page_title="Resume ↔ Job Match Scorer", layout="wide")

st.title("🔍 Resume ↔ Job Match Scorer (SBERT + CrossAttention GRU)")

st.write("Enter resume and job description below:")

col1, col2 = st.columns(2)

with col1:
    res_text = st.text_area("📝 Resume Text", height=300)

with col2:
    job_text = st.text_area("💼 Job Description", height=300)

if st.button("🔎 Evaluate Match"):
    if len(res_text.strip()) == 0 or len(job_text.strip()) == 0:
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

        st.write("**Matched Skills:**", ", ".join(matched) if matched else "—")
        st.write("**Missing Required Skills:**", ", ".join(missing) if missing else "—")
