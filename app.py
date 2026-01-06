import re
import os
import nltk
import spacy
import numpy as np
import requests

from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from knowledge_base import knowledge_base


nltk.download("punkt")
nltk.download("stopwords")

# ---------- LOAD ----------
nlp = spacy.load("en_core_web_sm")
STOP_WORDS = set(stopwords.words("english"))

# contraction remnants to kill permanently
BAD_TOKENS = {
    "im","ive","ill","id","youre","dont","cant","wont",
    "isnt","arent","wasnt","werent","havent","hasnt",
    "hadnt","doesnt","didnt","shouldnt","wouldnt","couldnt"
}

# ---------- NORMALIZATION ----------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[’']", "", text)   # remove apostrophes
    return text.strip()

# ---------- NOISE REMOVAL ----------
def remove_noise(text):
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # emojis
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ---------- TOKEN CLEANING (THE REAL FIX) ----------
def clean_tokens(tokens):
    cleaned = []
    for t in tokens:
        t = t.lower()
        if (
            t not in STOP_WORDS and
            t not in BAD_TOKENS and
            len(t) > 2 and
            t.isalpha()
        ):
            cleaned.append(t)
    return cleaned

# ---------- LEMMATIZATION ----------
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [t.lemma_ for t in doc]

# ---------- FINAL PIPELINE ----------
def preprocess(text):
    text = normalize(text)
    text = remove_noise(text)
    tokens = word_tokenize(text)
    tokens = clean_tokens(tokens)
    tokens = lemmatize(tokens)
    return tokens


INTENT_RULES = {
    "greeting": {
        "keywords": ["hi", "hello", "namaste", "hey"],
        "response": "Hello! How can I help you with farming today?"
    },

    "weather_advice": {
        "keywords": ["weather", "rain", "rainfall", "temperature", "forecast"],
        "response": "Please tell your location to get the weather forecast."
    },

    "market_price": {
        "keywords": ["price", "rate", "mandi", "market"],
        "response": "Tell me the crop name and mandi location."
    },

    "pest_issue": {
        "keywords": ["pest", "insect", "disease", "spots", "worms", "leaf curl"],
        "response": "Please tell the crop name and symptoms."
    },

    "fertilizer_advice": {
        "keywords": ["fertilizer", "urea", "dap", "npk", "manure"],
        "response": "Tell me the crop name and growth stage."
    },

    "govt_scheme": {
        "keywords": ["scheme", "subsidy", "loan", "insurance", "pmfby"],
        "response": "Which government scheme information do you need?"
    }
}

def preprocess_kb():
    for item in knowledge_base:
        item["processed_tokens"] = set(preprocess(item["question"]))

preprocess_kb()


# =========================
# STAGE 1: INTENT DETECTION
# =========================
def detect_intent(user_tokens, threshold=3):
    best_intent = None
    best_score = 0

    for intent, rule in INTENT_RULES.items():
        rule_tokens = set()
        for kw in rule["keywords"]:
            rule_tokens.update(preprocess(kw))
        score = len(user_tokens & rule_tokens)
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score >= threshold:
        return best_intent, best_score
    return None, 0

# =========================
# STAGE 2: KB SEARCH BY INTENT
# =========================
def search_kb_by_intent(user_tokens, intent, min_score=1):
    best_match = None
    best_score = 0

    for item in knowledge_base:
        if item["intent"] != intent:
            continue
        score = len(user_tokens & item["processed_tokens"])
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= min_score:
        return {
            "intent": best_match["intent"],
            "crop": best_match["crop"],
            "answer": best_match["answer"],
            "confidence": best_score
        }
    return None

# =========================
# STAGE 3: GENERIC INTENT RESPONSE
# =========================
def intent_generic_response(intent, confidence):
    return {
        "intent": intent,
        "crop": "none",
        "answer": INTENT_RULES[intent]["response"],
        "confidence": confidence
    }

# =========================
# STAGE 4: GLOBAL KB SEARCH
# =========================
def search_kb_globally(user_tokens, min_score=2):
    best_match = None
    best_score = 0

    for item in knowledge_base:
        score = len(user_tokens & item["processed_tokens"])
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= min_score:
        return {
            "intent": best_match["intent"],
            "crop": best_match["crop"],
            "answer": best_match["answer"],
            "confidence": best_score
        }
    return None

# =========================
# CONTROLLER
# =========================
def rule_based_chatbot(user_input):
    user_tokens = set(preprocess(user_input))

    intent, intent_score = detect_intent(user_tokens)

    if intent:
        kb_hit = search_kb_by_intent(user_tokens, intent)
        if kb_hit:
            return kb_hit
        return intent_generic_response(intent, intent_score)

    global_hit = search_kb_globally(user_tokens)
    if global_hit:
        return global_hit

    return {
        "intent": "fallback",
        "crop": "none",
        "answer": "Sorry, I could not understand your question.",
        "confidence": 0
    }

# =========================
# RAG PREPARATION (Layer-2)
# =========================

rag_model = SentenceTransformer("all-MiniLM-L6-v2")

rag_questions = []
rag_answers = []

for item in knowledge_base:
    rag_questions.append(" ".join(item["processed_tokens"]))
    rag_answers.append(item["answer"])

rag_embeddings = rag_model.encode(rag_questions)

# =========================
# RAG SEARCH
# =========================
def rag_chatbot(user_input, threshold=0.55):
    query = " ".join(preprocess(user_input))
    query_vec = rag_model.encode([query])

    scores = cosine_similarity(query_vec, rag_embeddings)[0]
    best_idx = np.argmax(scores)

    if scores[best_idx] >= threshold:
        return {
            "intent": "rag_retrieval",
            "crop": "none",
            "answer": rag_answers[best_idx],
            "confidence": float(scores[best_idx])
        }

    return None




GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
GOOGLE_CX = os.environ.get("GOOGLE_CX")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")



def google_search(query, top_k=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
        "num": top_k
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    return [
        {
            "title": i.get("title",""),
            "snippet": i.get("snippet",""),
            "link": i.get("link","")
        }
        for i in data.get("items", [])
    ]



client = genai.Client(api_key=GEMINI_API_KEY)

def summarize_with_gemini(query, search_results):
    if not search_results:
        return "NOT FOUND"

    context = "\n\n".join(
        f"Title: {r['title']}\nContent: {r['snippet']}"
        for r in search_results
    )

    prompt = f"""
You are a factual summarizer.

RULES:

You are an agricultural-domain question answering system.

RULES (ABSOLUTE – NO EXCEPTIONS):
1. You must answer ONLY agriculture-related questions.
2. Agriculture includes ONLY: crops, soil, fertilizers, irrigation, pests, diseases, weather impact on farming, livestock, farming techniques, agricultural economics, farm machinery, seeds.
3. If the user question is NOT strictly related to agriculture, reply EXACTLY:
   question is irrelevant
4. Do NOT answer general knowledge, health, population, politics, technology, personal, or social questions.
5. Do NOT answer even if the search engine provides an answer for a non-agriculture question.
6. Do NOT rephrase, explain, or add extra text.
7. If the question IS agriculture-related but the provided information does not contain an answer, reply EXACTLY:
   NOT FOUND
8. Use ONLY the information explicitly present in the provided context.
9. Do NOT use prior knowledge.
10. Do NOT infer or guess.

DECISION PROCESS (MANDATORY):
- First, classify the question as AGRICULTURE or NON-AGRICULTURE.
- If NON-AGRICULTURE → respond exactly: question is irrelevant
- If AGRICULTURE:
    - If answer exists in provided data → give concise answer
    - Else → respond exactly: NOT FOUND

OUTPUT FORMAT:
- Output ONLY the final answer text.
- No explanations.
- No prefixes.
- No suffixes.


"

QUESTION:
{query}

INFORMATION:
{context}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"NOT FOUND"

def search_and_answer(query):
    results = google_search(query)
    answer = summarize_with_gemini(query, results)
    return answer

# =========================
# CHAT LOOP 
# =========================

app = Flask(__name__)

@app.route("/")
def home():
    return "Farmer Chatbot Service Active"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    # STEP 1 – Try RULE BASED
    rule_response = rule_based_chatbot(user_input)

    if rule_response["intent"] != "fallback":
        return jsonify({
            "source": "rule_based",
            "reply": rule_response["answer"],
            "confidence": rule_response["confidence"]
        })

    # STEP 2 – Try RAG
    rag_hit = rag_chatbot(user_input)

    if rag_hit:
        return jsonify({
            "source": "rag",
            "reply": rag_hit["answer"],
            "confidence": rag_hit["confidence"]
        })

    # STEP 3 – Try Google Search + Gemini
    search_reply = search_and_answer(user_input)

    if search_reply != "NOT FOUND":
        return jsonify({
            "source": "search_gemini",
            "reply": search_reply,
            "confidence": 0.5
        })

    # FINAL FALLBACK
    return jsonify({
        "source": "fallback",
        "reply": "Sorry, I could not find an answer.",
        "note": "Please provide more crop related details",
        "confidence": 0
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


