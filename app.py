import streamlit as st
import numpy as np
import re

# =========================
# 1. Skills & utils
# =========================

SKILLS = {
    "python","java","javascript","c++","sql","nosql","git","linux",
    "docker","kubernetes","ml","nlp","pandas","numpy","scikit-learn",
    "pytorch","tensorflow","react","node","flask","django","fastapi",
    "aws","azure","gcp","data analysis","machine learning","deep learning"
}

LABELS = {0: "❌ No Fit", 1: "⚙️ Partial Fit", 2: "✅ Good Fit"}


def simple_tokenize(text: str):
    text = text.lower()
    # Разбиваем по не-буквено-цифровым символам
    tokens = re.split(r"[^a-z0-9\+]+", text)
    tokens = [t for t in tokens if t]
    return set(tokens)


def extract_skills(text: str):
    text = text.lower()
    return {s for s in SKILLS if s in text}


# =========================
# 2. Heuristic scorer (no ML, but “model-like”)
# =========================

def heuristic_score(res_text: str, job_text: str):
    # Токены
    res_tokens = simple_tokenize(res_text)
    job_tokens = simple_tokenize(job_text)

    # Пересечение токенов
    if len(job_tokens) == 0:
        token_overlap = 0.0
    else:
        token_overlap = len(res_tokens & job_tokens) / len(job_tokens)

    # Навыки
    skills_r = extract_skills(res_text)
    skills_j = extract_skills(job_text)

    if len(skills_j) == 0:
        skill_overlap = 0.0
    else:
        skill_overlap = len(skills_r & skills_j) / len(skills_j)

    # Итоговый скор: комбинируем обе компоненты
    score = 0.5 * token_overlap + 0.5 * skill_overlap
    return score, res_tokens, job_tokens, skills_r, skills_j


def predict(res_text: str, job_text: str):
    score, res_tokens, job_tokens, skills_r, skills_j = heuristic_score(res_text, job_text)

    matched_skills = skills_r & skills_j
    missing_skills = skills_j - skills_r

    # Простая “политика” по score:
    # >= 0.7 → Good Fit
    # <= 0.3 → No Fit
    # иначе → Partial Fit
    if score >= 0.7:
        pred = 2
        probs = np.array([0.05, 0.15, 0.80])
    elif score <= 0.3:
        pred = 0
        probs = np.array([0.80, 0.15, 0.05])
    else:
        pred = 1
        # Чем ближе к 0.5, тем больше уверенность в Partial
        center = abs(score - 0.5)
        p_partial = 0.6 + (0.2 * (0.5 - center))  # от 0.6 до 0.7
        remaining = 1.0 - p_partial
        probs = np.array([remaining / 2, p_partial, remaining / 2])

    return pred, probs, matched_skills, missing_skills


# =========================
# 3. Streamlit UI
# =========================

st.set_page_config(page_title="Resume ↔ Job Match Scorer", layout="wide")
st.title("🔍 Resume ↔ Job Match Scorer (Heuristic Demo)")

st.write(
    "This is a **lightweight demo version** of the Resume ↔ Job Match Scorer. "
    "It estimates match quality using skill overlap and token similarity between resume and job description."
)

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

        st.subheader("📊 Probabilities (heuristic):")
        st.write(f"No Fit: {probs[0]:.3f}")
        st.write(f"Partial Fit: {probs[1]:.3f}")
        st.write(f"Good Fit: {probs[2]:.3f}")

        st.subheader("🧩 Skills Analysis:")
        st.write("**Matched Skills:**", ", ".join(sorted(matched)) if matched else "—")
        st.write("**Missing Required Skills:**", ", ".join(sorted(missing)) if missing else "—")

        st.caption(
            "Note: This cloud demo uses a simplified heuristic model for stability. "
            "The full SBERT + CrossAttention GRU model with proper training and evaluation "
            "is described in the project report and implemented in the Jupyter notebook."
        )
