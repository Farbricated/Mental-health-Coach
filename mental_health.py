"""
Mindful Pro v3.0 — Streamlit Web App
Powered by Groq (llama-3.1-8b-instant)

Run:
    pip install streamlit plotly requests python-dotenv
    streamlit run app.py
"""

import streamlit as st
import requests
import json
import os
import sqlite3
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Mindful Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL   = "llama-3.1-8b-instant"   # fast, small, great quality
    TEMPERATURE  = 0.75
    MAX_TOKENS   = 500
    DB_PATH      = "mindful_pro.db"

    AFFIRMATIONS = [
        "You are stronger than you think.",
        "Progress, not perfection.",
        "Your feelings are valid — and they will pass.",
        "Small steps still move you forward.",
        "You've survived every hard day so far.",
        "You are not your thoughts.",
        "Healing is not linear — be patient with yourself.",
        "You deserve the same compassion you give others.",
        "Rest is productive. Recovering is worthwhile.",
        "One moment at a time is always enough.",
    ]

    CRISIS_LINES = [
        ("🇮🇳 iCall (India)",      "+91-9152987821"),
        ("🇮🇳 AASRA (India)",       "+91-9820466726"),
        ("🇺🇸 988 Lifeline (USA)",   "988"),
        ("🇺🇸 Crisis Text (USA)",    "Text HOME → 741741"),
        ("🇬🇧 Samaritans (UK)",      "116 123"),
    ]

    EMOTIONS = ["😰 Anxious","😢 Sad","😠 Angry","😓 Overwhelmed",
                "😴 Exhausted","🤔 Confused","😐 Neutral","🙂 Hopeful","😊 Happy"]


# ═══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS  — calm, therapeutic dark theme
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --bg:          #0e1117;
        --surface:     #161b25;
        --surface2:    #1e2535;
        --border:      #2a3347;
        --accent:      #5b8ff0;
        --accent-soft: #3d6ad6;
        --green:       #52c788;
        --amber:       #f0b429;
        --red:         #e05b5b;
        --text:        #e8ecf4;
        --muted:       #8892a4;
        --radius:      14px;
    }

    /* ── Global reset ── */
    html, body, .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', Georgia, serif !important;
        color: var(--text) !important;
        letter-spacing: -0.02em;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Chat messages ── */
    .user-msg {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: var(--radius) var(--radius) 4px var(--radius);
        padding: 14px 18px;
        margin: 8px 0 8px 48px;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    .bot-msg {
        background: linear-gradient(135deg, #1a2540 0%, #1c2a45 100%);
        border: 1px solid #2e4070;
        border-radius: var(--radius) var(--radius) var(--radius) 4px;
        padding: 16px 20px;
        margin: 8px 48px 8px 0;
        font-size: 0.95rem;
        line-height: 1.75;
    }
    .bot-label {
        font-size: 0.72rem;
        color: var(--accent);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .user-label {
        font-size: 0.72rem;
        color: var(--muted);
        text-align: right;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 500;
    }

    /* ── Cards ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin: 12px 0;
    }
    .card-accent {
        border-left: 3px solid var(--accent);
    }
    .card-green  { border-left: 3px solid var(--green); }
    .card-amber  { border-left: 3px solid var(--amber); }
    .card-red    { border-left: 3px solid var(--red); }

    /* ── Emotion badge ── */
    .emotion-badge {
        display: inline-block;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        color: var(--accent);
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* ── Mood bar ── */
    .mood-bar-wrap { background: var(--border); border-radius: 4px; height: 8px; }
    .mood-bar-fill { height: 8px; border-radius: 4px;
                     background: linear-gradient(90deg, var(--accent-soft), var(--green)); }

    /* ── Metric tiles ── */
    .metric-tile {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 18px 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: var(--accent);
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 6px;
    }

    /* ── Affirmation banner ── */
    .affirmation {
        background: linear-gradient(135deg, #1a2a4a, #1e3555);
        border: 1px solid #2e4d80;
        border-radius: var(--radius);
        padding: 18px 24px;
        font-family: 'DM Serif Display', serif;
        font-style: italic;
        font-size: 1.15rem;
        color: #a8c4f0;
        text-align: center;
        margin: 12px 0 20px 0;
    }

    /* ── Crisis box ── */
    .crisis-box {
        background: #2a1818;
        border: 1px solid var(--red);
        border-radius: var(--radius);
        padding: 20px 24px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(91,143,240,0.2) !important;
    }

    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] { padding: 6px 0; }
    .stSlider [data-testid="stSlider"] > div > div > div {
        background: var(--accent) !important;
    }

    /* ── Select boxes ── */
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface2) !important;
        color: var(--text) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 16px 0 !important; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── Plotly chart backgrounds ── */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Goal item ── */
    .goal-item {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .goal-done { opacity: 0.5; text-decoration: line-through; }

    /* ── Streak fire ── */
    .streak-number {
        font-family: 'DM Serif Display', serif;
        font-size: 3.5rem;
        color: var(--amber);
        line-height: 1;
    }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_db():
    return Database()

class Database:
    def __init__(self):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, role TEXT, content TEXT, session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS mood_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, mood_score INTEGER, emotion TEXT,
                    note TEXT, energy INTEGER, sleep_hours REAL
                );
                CREATE TABLE IF NOT EXISTS gratitude (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT, entry TEXT
                );
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created TEXT, title TEXT, category TEXT,
                    completed INTEGER DEFAULT 0, completed_date TEXT
                );
                CREATE TABLE IF NOT EXISTS streaks (
                    id INTEGER PRIMARY KEY, last_checkin TEXT,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    total_days INTEGER DEFAULT 0
                );
            """)

    # ── Mood ──────────────────────────────────────────────────────────────────
    def log_mood(self, score, emotion, note, energy, sleep):
        today = date.today().isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("INSERT INTO mood_journal(date,mood_score,emotion,note,energy,sleep_hours)"
                      " VALUES(?,?,?,?,?,?)", (today,score,emotion,note,energy,sleep))
        self._update_streak()

    def get_mood_history(self, days=14):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT date,mood_score,emotion,note,energy,sleep_hours "
                             "FROM mood_journal WHERE date>=? ORDER BY date", (since,)).fetchall()
        return [{"date":r[0],"score":r[1],"emotion":r[2],
                 "note":r[3],"energy":r[4],"sleep":r[5]} for r in rows]

    # ── Gratitude ─────────────────────────────────────────────────────────────
    def log_gratitude(self, entries: List[str]):
        today = date.today().isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            for e in entries:
                if e.strip():
                    c.execute("INSERT INTO gratitude(date,entry) VALUES(?,?)", (today,e.strip()))

    def get_gratitude(self, days=30):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT date,entry FROM gratitude "
                             "WHERE date>=? ORDER BY date DESC", (since,)).fetchall()
        return [{"date":r[0],"entry":r[1]} for r in rows]

    # ── Goals ─────────────────────────────────────────────────────────────────
    def add_goal(self, title, category):
        with sqlite3.connect(Config.DB_PATH) as c:
            cur = c.execute("INSERT INTO goals(created,title,category) VALUES(?,?,?)",
                            (datetime.now().isoformat(), title, category))
            return cur.lastrowid

    def complete_goal(self, goal_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("UPDATE goals SET completed=1, completed_date=? WHERE id=?",
                      (date.today().isoformat(), goal_id))

    def delete_goal(self, goal_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("DELETE FROM goals WHERE id=?", (goal_id,))

    def get_goals(self, completed=None):
        with sqlite3.connect(Config.DB_PATH) as c:
            if completed is None:
                rows = c.execute("SELECT id,created,title,category,completed,completed_date "
                                 "FROM goals ORDER BY completed,created DESC").fetchall()
            else:
                rows = c.execute("SELECT id,created,title,category,completed,completed_date "
                                 "FROM goals WHERE completed=? ORDER BY created DESC",
                                 (1 if completed else 0,)).fetchall()
        return [{"id":r[0],"created":r[1],"title":r[2],"category":r[3],
                 "completed":r[4],"completed_date":r[5]} for r in rows]

    # ── Streaks ───────────────────────────────────────────────────────────────
    def _update_streak(self):
        today     = date.today().isoformat()
        yesterday = (date.today()-timedelta(days=1)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            row = c.execute("SELECT last_checkin,current_streak,longest_streak,total_days "
                            "FROM streaks WHERE id=1").fetchone()
            if not row:
                c.execute("INSERT INTO streaks VALUES(1,?,1,1,1)", (today,)); return
            last,cur,longest,total = row
            if last == today: return
            cur = cur+1 if last==yesterday else 1
            c.execute("UPDATE streaks SET last_checkin=?,current_streak=?,longest_streak=?,total_days=?"
                      " WHERE id=1", (today,cur,max(longest,cur),total+1))

    def get_streak(self):
        with sqlite3.connect(Config.DB_PATH) as c:
            row = c.execute("SELECT last_checkin,current_streak,longest_streak,total_days "
                            "FROM streaks WHERE id=1").fetchone()
        return {"current":row[1],"longest":row[2],"total":row[3],"last":row[0]} if row \
               else {"current":0,"longest":0,"total":0,"last":None}

    # ── Conversation log ──────────────────────────────────────────────────────
    def log_turn(self, role, content, session_id):
        with sqlite3.connect(Config.DB_PATH) as c:
            c.execute("INSERT INTO conversations(timestamp,role,content,session_id) VALUES(?,?,?,?)",
                      (datetime.now().isoformat(), role, content, session_id))

    def get_mood_stats(self, days=7):
        since = (date.today()-timedelta(days=days)).isoformat()
        with sqlite3.connect(Config.DB_PATH) as c:
            rows = c.execute("SELECT mood_score,emotion FROM mood_journal "
                             "WHERE date>=?", (since,)).fetchall()
        if not rows:
            return {"avg_score":0,"top_emotion":"—","count":0}
        scores  = [r[0] for r in rows if r[0]]
        emotions= [r[1] for r in rows if r[1]]
        from collections import Counter
        top = Counter(emotions).most_common(1)[0][0] if emotions else "—"
        return {"avg_score":round(sum(scores)/len(scores),1) if scores else 0,
                "top_emotion":top, "count":len(rows)}


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAIN  (lightweight emotion analysis — no dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

class Brain:
    EMOTION_KW = {
        "anxious":     ["anxious","nervous","worried","panic","scared","stressed","fear","dread","anxiety"],
        "sad":         ["sad","depressed","down","hopeless","empty","lonely","miserable","grief","crying"],
        "angry":       ["angry","frustrated","furious","annoyed","mad","irritated","rage","bitter"],
        "overwhelmed": ["overwhelmed","drowning","too much","can't handle","crushing","swamped"],
        "exhausted":   ["exhausted","burnt out","tired","drained","no energy","burned out","depleted"],
        "confused":    ["confused","lost","don't know","unsure","uncertain","unclear"],
        "hopeful":     ["hopeful","better","improving","good","happy","excited","optimistic"],
    }
    CRISIS_KW = ["suicide","kill myself","end my life","want to die","end it all",
                 "take my life","not worth living","better off dead","self harm","hurt myself"]

    def analyze(self, text: str) -> Dict:
        t = text.lower()
        scores = {}
        for em,kws in self.EMOTION_KW.items():
            c = sum(1 for kw in kws if kw in t)
            if c: scores[em] = min(100, c*28)
        primary = max(scores, key=scores.get) if scores else "neutral"
        is_crisis = any(kw in t for kw in self.CRISIS_KW)
        return {"primary_emotion": primary,
                "emotion_intensity": scores.get(primary, 0),
                "is_crisis": is_crisis}


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ  (llama-3.1-8b-instant)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Mindful Pro, a warm and emotionally intelligent mental health companion.

Your character:
- Like a wise, caring friend who happens to have therapeutic training
- Warm and human — use contractions, varied sentence length, never sound robotic
- Empathetic first, advice second — always validate before suggesting

Rules:
- Validate feelings BEFORE any advice or techniques
- Give specific, actionable guidance — never vague platitudes  
- Explain briefly WHY a technique works when you suggest one
- Weave in context from earlier in the conversation when relevant
- End with a gentle question or open space — don't close things off
- Keep responses to 2-3 paragraphs (~150-200 words)
- NEVER diagnose, prescribe, or claim to replace professional therapy
- If you detect crisis language, immediately provide crisis hotlines"""

def groq_chat(messages: List[Dict], api_key: str) -> str:
    try:
        resp = requests.post(
            Config.GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": Config.GROQ_MODEL,
                  "messages": messages,
                  "temperature": Config.TEMPERATURE,
                  "max_tokens": Config.MAX_TOKENS},
            timeout=25,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        err = ""
        try: err = resp.json().get("error",{}).get("message","")
        except: pass
        return f"⚠️ Groq error ({resp.status_code}): {err or resp.text[:120]}"
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except Exception as e:
        return f"⚠️ Error: {e}"


def build_messages(history: List[Dict], user_msg: str, analysis: Dict) -> List[Dict]:
    hint = (f"\n\n[Emotion context: {analysis['primary_emotion']} "
            f"({analysis['emotion_intensity']}% intensity). "
            f"Validate first, then support.]")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT + hint}]
    msgs += history
    msgs.append({"role": "user", "content": user_msg})
    return msgs


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "chat_history":  [],      # [{role, content}]
        "session_id":    datetime.now().strftime("%Y%m%d%H%M%S"),
        "page":          "Chat",
        "api_key":       Config.GROQ_API_KEY,
        "affirmation":   random.choice(Config.AFFIRMATIONS),
        "brain":         Brain(),
        "checkin_done_today": False,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(db: Database):
    with st.sidebar:
        st.markdown("### 🌿 Mindful Pro")
        st.caption(f"Model: `{Config.GROQ_MODEL}`")
        st.divider()

        # API key input if not set
        if not st.session_state.api_key:
            key = st.text_input("Groq API Key", type="password",
                                placeholder="gsk_...",
                                help="Get a free key at console.groq.com")
            if key:
                st.session_state.api_key = key
                st.rerun()
        else:
            st.success("✅ Groq connected", icon="✅")

        st.divider()

        # Navigation
        pages = {
            "💬 Chat":         "Chat",
            "🌅 Daily Check-in":"Checkin",
            "📊 Mood Chart":    "Chart",
            "🙏 Gratitude":     "Gratitude",
            "🎯 Goals":         "Goals",
            "🧘 Toolkit":       "Toolkit",
            "📈 Progress":      "Progress",
        }
        for label, key in pages.items():
            if st.button(label, use_container_width=True,
                         type="primary" if st.session_state.page==key else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.divider()

        # Streak widget
        streak = db.get_streak()
        if streak["current"] > 0:
            fire = "🔥" * min(streak["current"], 7)
            st.markdown(f"""
            <div style="text-align:center;padding:12px 0">
                <div style="font-size:1.6rem">{fire}</div>
                <div style="font-size:0.8rem;color:#8892a4;margin-top:4px">
                    {streak['current']}-day streak
                </div>
            </div>""", unsafe_allow_html=True)

        # Clear chat
        if st.session_state.page == "Chat":
            st.divider()
            if st.button("🔄 Clear conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.session_id   = datetime.now().strftime("%Y%m%d%H%M%S")
                st.rerun()

        st.divider()
        st.caption("All data stored locally. Nothing leaves your device except Groq API calls.")


def render_crisis_box():
    st.markdown("""
    <div class="crisis-box">
        <h4 style="color:#e05b5b;margin:0 0 12px 0">🚨 You're not alone — please reach out now</h4>
        <p style="color:#e8ecf4;margin:0 0 12px 0;font-size:0.9rem">
        What you're feeling right now is not permanent. Trained people are ready to help.</p>
    </div>""", unsafe_allow_html=True)
    cols = st.columns(2)
    for i,(name,number) in enumerate(Config.CRISIS_LINES):
        with cols[i%2]:
            st.markdown(f"""
            <div class="card card-red" style="margin:4px 0;padding:10px 14px">
                <div style="font-size:0.82rem;color:#8892a4">{name}</div>
                <div style="font-size:1rem;font-weight:500;color:#f0b429">{number}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_chat(db: Database):
    st.markdown("## 💬 Talk to Mindful Pro")

    # Affirmation banner
    st.markdown(f'<div class="affirmation">"{st.session_state.affirmation}"</div>',
                unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.warning("Add your Groq API key in the sidebar to start chatting. "
                   "Get a free key at [console.groq.com](https://console.groq.com).")
        return

    # Render history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div>
                <div class="user-label">You</div>
                <div class="user-msg">{msg["content"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div>
                <div class="bot-label">🌿 Mindful Pro · {Config.GROQ_MODEL}</div>
                <div class="bot-msg">{msg["content"].replace(chr(10),"<br>")}</div>
            </div>""", unsafe_allow_html=True)

    # Input area
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    user_input = st.chat_input("What's on your mind?", key="chat_input")

    if user_input:
        analysis = st.session_state.brain.analyze(user_input)

        # Crisis check
        if analysis["is_crisis"]:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            render_crisis_box()
            crisis_reply = ("I'm really concerned about what you're sharing. Your safety is the only priority right now. "
                            "Please reach out to one of the crisis lines above — trained counselors are available 24/7. "
                            "You don't have to face this alone, and what you're feeling right now is not permanent. 💙")
            st.session_state.chat_history.append({"role":"assistant","content":crisis_reply})
            db.log_turn("user", user_input, st.session_state.session_id)
            db.log_turn("assistant", crisis_reply, st.session_state.session_id)
            st.rerun()
            return

        # Build and send to Groq
        messages = build_messages(st.session_state.chat_history, user_input, analysis)

        with st.spinner(""):
            reply = groq_chat(messages, st.session_state.api_key)

        st.session_state.chat_history.append({"role":"user",      "content":user_input})
        st.session_state.chat_history.append({"role":"assistant", "content":reply})
        db.log_turn("user",      user_input, st.session_state.session_id)
        db.log_turn("assistant", reply,      st.session_state.session_id)

        # Contextual tip
        em = analysis["primary_emotion"]
        tips = {"anxious":"Try typing **breathe** or visit the Toolkit for grounding exercises.",
                "overwhelmed":"The Toolkit has box breathing and grounding to help right now.",
                "exhausted":"A short meditation might help — check the Toolkit.",
                "sad":"The Gratitude practice can gently lift mood over time."}
        if em in tips:
            st.info(f"💡 {tips[em]}")

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  DAILY CHECK-IN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_checkin(db: Database):
    st.markdown("## 🌅 Daily Check-in")
    st.markdown("Track how you feel each day. Consistency builds self-awareness over time.")

    # Check if done today
    history = db.get_mood_history(1)
    if history and history[-1]["date"] == date.today().isoformat():
        st.success("✅ Check-in complete for today!")
        entry = history[-1]
        c1,c2,c3,c4 = st.columns(4)
        with c1: render_metric("Mood",   f"{entry['score']}/10", "")
        with c2: render_metric("Emotion", entry['emotion'].title(), "")
        with c3: render_metric("Energy",  f"{entry['energy']}/5", "")
        with c4: render_metric("Sleep",   f"{entry['sleep']}h", "")
        if entry.get("note"):
            st.markdown(f'<div class="card card-accent"><em>"{entry["note"]}"</em></div>',
                        unsafe_allow_html=True)
        if st.button("Update today's check-in"):
            st.session_state.checkin_done_today = False
            st.rerun()
        return

    with st.form("checkin_form"):
        st.markdown("#### How are you feeling today?")

        mood = st.slider("Overall mood", 1, 10, 5,
                         help="1 = very bad, 10 = amazing")
        mood_labels = {1:"😣 Very bad",2:"😢 Bad",3:"😔 Low",4:"😐 Neutral",5:"🙂 Okay",
                       6:"🙂 Good",7:"😊 Pretty good",8:"😃 Great",9:"🤩 Excellent",10:"✨ Amazing"}
        st.caption(mood_labels.get(mood,""))

        col1, col2 = st.columns(2)
        with col1:
            emotion_raw = st.selectbox("Dominant emotion", Config.EMOTIONS)
            emotion = emotion_raw.split(" ",1)[1].lower() if " " in emotion_raw else emotion_raw.lower()
        with col2:
            energy = st.slider("Energy level", 1, 5, 3,
                               help="1 = depleted, 5 = energised")

        sleep = st.number_input("Hours slept last night", 0.0, 24.0, 7.0, 0.5)
        note  = st.text_input("One sentence about today (optional)",
                              placeholder="e.g. Tough meeting, but managed well")

        submitted = st.form_submit_button("✅ Save check-in", use_container_width=True)

    if submitted:
        db.log_mood(mood, emotion, note, energy, sleep)
        st.session_state.checkin_done_today = True

        streak = db.get_streak()
        st.balloons()
        st.success(f"Saved! 🔥 {streak['current']}-day streak")

        # Personalised insight
        if mood <= 3:
            st.info("💙 Mood is low today — that's okay. Try one small kind act for yourself.")
        elif mood >= 8:
            st.info("✨ You're doing well — what's contributing? Worth noting for reference.")
        if sleep < 6:
            st.warning("😴 Under 6 hours of sleep significantly affects mood and anxiety levels.")
        if energy <= 2:
            st.info("⚡ Low energy? Even a 5-minute walk outside can shift your state.")

        st.rerun()


def render_metric(label, value, delta=""):
    st.markdown(f"""
    <div class="metric-tile">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f'<div style="font-size:0.8rem;color:#52c788;margin-top:4px">{delta}</div>' if delta else ''}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MOOD CHART PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_chart(db: Database):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    st.markdown("## 📊 Mood Chart")

    days = st.select_slider("Time window", [7,14,30], value=14,
                            format_func=lambda x: f"{x} days")
    history = db.get_mood_history(days)

    if not history:
        st.markdown("""
        <div class="card card-accent" style="text-align:center;padding:40px">
            <div style="font-size:2rem;margin-bottom:12px">📋</div>
            <div>No check-in data yet.</div>
            <div style="color:#8892a4;font-size:0.9rem;margin-top:6px">
                Do your first daily check-in to see trends here.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    dates   = [r["date"] for r in history]
    scores  = [r["score"] or 0 for r in history]
    energies= [r["energy"] or 0 for r in history]
    sleeps  = [r["sleep"] or 0 for r in history]
    emotions= [r["emotion"] or "neutral" for r in history]

    # Summary metrics
    col1,col2,col3,col4 = st.columns(4)
    with col1: render_metric("Avg Mood",   f"{sum(scores)/len(scores):.1f}/10","")
    with col2: render_metric("Avg Energy", f"{sum(energies)/len(energies):.1f}/5","")
    with col3: render_metric("Avg Sleep",  f"{sum(sleeps)/len(sleeps):.1f}h","")
    with col4:
        from collections import Counter
        top_em = Counter(emotions).most_common(1)[0][0].title() if emotions else "—"
        render_metric("Top Emotion", top_em, "")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if PLOTLY:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Mood Score", "Energy Level", "Sleep Hours"),
                            vertical_spacing=0.08)
        # Mood
        fig.add_trace(go.Scatter(
            x=dates, y=scores, mode="lines+markers",
            line=dict(color="#5b8ff0", width=2.5),
            marker=dict(size=7, color="#5b8ff0"),
            fill="tozeroy", fillcolor="rgba(91,143,240,0.08)",
            name="Mood",
        ), row=1, col=1)
        # Energy
        fig.add_trace(go.Bar(
            x=dates, y=energies,
            marker_color="#52c788", opacity=0.8, name="Energy",
        ), row=2, col=1)
        # Sleep
        fig.add_trace(go.Bar(
            x=dates, y=sleeps,
            marker_color="#f0b429", opacity=0.8, name="Sleep",
        ), row=3, col=1)
        fig.update_layout(
            height=520, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892a4", family="DM Sans"),
            margin=dict(l=0,r=0,t=32,b=0),
        )
        for i in range(1,4):
            fig.update_xaxes(showgrid=False, row=i, col=1)
            fig.update_yaxes(gridcolor="#2a3347", row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # ASCII fallback
        st.markdown("**Mood History** (install `plotly` for charts)")
        for r in history:
            bar = "█" * (r["score"] or 0) + "░" * (10-(r["score"] or 0))
            st.text(f"  {r['date'][-5:]}  {r['emotion'][:8]:8s}  [{bar}] {r['score']}/10")

    # Emotion breakdown
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### Emotion breakdown")
    from collections import Counter
    em_count = Counter(emotions)
    cols = st.columns(min(len(em_count), 4))
    for i,(em,count) in enumerate(em_count.most_common(4)):
        with cols[i]:
            pct = int(count/len(emotions)*100)
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:14px">
                <div style="font-size:1.5rem">{_em_emoji(em)}</div>
                <div style="font-size:0.9rem;margin:4px 0">{em.title()}</div>
                <div style="font-size:0.75rem;color:#8892a4">{count}× ({pct}%)</div>
            </div>""", unsafe_allow_html=True)


def _em_emoji(em):
    return {"anxious":"😰","sad":"😢","angry":"😠","overwhelmed":"😓",
            "exhausted":"😴","confused":"🤔","hopeful":"🙂","happy":"😊"}.get(em,"😐")


# ─────────────────────────────────────────────────────────────────────────────
#  GRATITUDE PAGE
# ─────────────────────────────────────────────────────────────────────────────

GRATITUDE_PROMPTS = [
    "Something small that made you smile today",
    "Someone who made your life easier recently",
    "A challenge that taught you something valuable",
    "Something about your body or health you appreciate",
    "A memory that brings you warmth",
    "Something in your environment you often overlook",
    "A personal quality you're glad to have",
    "A simple pleasure you had recently",
    "Something that went better than expected",
]

def page_gratitude(db: Database):
    st.markdown("## 🙏 Gratitude Practice")
    st.markdown("Specific gratitude — naming *why* something matters — rewires the brain for positivity within 21 days.")

    tab1, tab2 = st.tabs(["✍️ Add entries", "📚 Past entries"])

    with tab1:
        prompts = random.sample(GRATITUDE_PROMPTS, 3)
        entries = []
        with st.form("gratitude_form"):
            for i, prompt in enumerate(prompts, 1):
                val = st.text_input(f"{i}. {prompt}", placeholder="Be specific…",
                                    key=f"grat_{i}")
                entries.append(val)
            submitted = st.form_submit_button("💛 Save gratitudes", use_container_width=True)

        if submitted:
            saved = [e for e in entries if e.strip()]
            if saved:
                db.log_gratitude(saved)
                st.success(f"✅ {len(saved)} gratitude entries saved!")
                st.balloons()
            else:
                st.warning("Please add at least one entry.")

    with tab2:
        recent = db.get_gratitude(30)
        if not recent:
            st.info("No entries yet — add some above!")
        else:
            # Group by date
            from collections import defaultdict
            by_date = defaultdict(list)
            for g in recent:
                by_date[g["date"]].append(g["entry"])
            for d in sorted(by_date.keys(), reverse=True)[:14]:
                st.markdown(f"**{d}**")
                for entry in by_date[d]:
                    st.markdown(f"""
                    <div class="card card-amber" style="margin:4px 0;padding:10px 16px;font-size:0.92rem">
                        💛 {entry}
                    </div>""", unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  GOALS PAGE
# ─────────────────────────────────────────────────────────────────────────────

GOAL_CATS = ["Mental health","Relationships","Work / Study","Physical health","Habits","Creativity","Other"]

def page_goals(db: Database):
    st.markdown("## 🎯 Goals")

    tab1, tab2 = st.tabs(["📋 Active goals", "➕ Add goal"])

    with tab1:
        active    = db.get_goals(completed=False)
        completed = db.get_goals(completed=True)

        if not active:
            st.markdown("""
            <div class="card" style="text-align:center;padding:32px">
                <div style="font-size:1.6rem">🎯</div>
                <div style="margin-top:8px;color:#8892a4">No active goals yet. Add one in the next tab.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for g in active:
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="card card-accent" style="padding:12px 16px">
                        <div style="font-size:0.95rem">{g['title']}</div>
                        <div style="font-size:0.75rem;color:#8892a4;margin-top:4px">
                            {g['category']} · added {g['created'][:10]}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    if st.button("✅", key=f"complete_{g['id']}", help="Mark complete"):
                        db.complete_goal(g["id"])
                        st.rerun()
                with col3:
                    if st.button("🗑️", key=f"delete_{g['id']}", help="Delete"):
                        db.delete_goal(g["id"])
                        st.rerun()

        if completed:
            st.markdown(f"<div style='margin-top:20px;color:#8892a4;font-size:0.85rem'>"
                        f"✅ {len(completed)} goal(s) completed</div>", unsafe_allow_html=True)
            with st.expander("Show completed"):
                for g in completed[-10:]:
                    st.markdown(f"""
                    <div style="color:#52c788;font-size:0.9rem;padding:6px 0;
                                text-decoration:line-through;opacity:0.7">
                        ✔ {g['title']} <span style="color:#8892a4">{g['completed_date']}</span>
                    </div>""", unsafe_allow_html=True)

    with tab2:
        with st.form("goal_form"):
            title = st.text_input("Goal",
                                  placeholder="Be specific: 'Meditate 5 min every morning' not 'meditate more'")
            category = st.selectbox("Category", GOAL_CATS)
            submitted = st.form_submit_button("➕ Add goal", use_container_width=True)
        if submitted and title.strip():
            db.add_goal(title.strip(), category)
            st.success(f"✅ Goal added: \"{title}\"")
            st.rerun()
        elif submitted:
            st.warning("Please enter a goal title.")


# ─────────────────────────────────────────────────────────────────────────────
#  TOOLKIT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_toolkit():
    st.markdown("## 🧘 Therapy Toolkit")
    st.markdown("Evidence-based techniques for immediate relief and long-term resilience.")

    techniques = [
        ("🫁 4-7-8 Breathing",    "Anxiety · Stress · Sleep",    "90 sec",
         "Activates the parasympathetic nervous system. Inhale 4s · Hold 7s · Exhale 8s. Repeat 3–4 cycles.",
         "478"),
        ("📦 Box Breathing",       "Acute stress · Panic",        "2 min",
         "Used by Navy SEALs to reset under pressure. Inhale 4s · Hold 4s · Exhale 4s · Hold 4s.",
         "box"),
        ("🧘 5-4-3-2-1 Grounding","Panic · Overwhelm · Anxiety", "3-5 min",
         "Anchors you in the present moment. Name 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste.",
         "54321"),
        ("📝 CBT Thought Record",  "Negative thoughts · Worry",   "5-10 min",
         "Challenge distorted thinking by examining evidence for and against automatic thoughts.",
         "cbt"),
        ("😴 Progressive Relaxation","Tension · Insomnia · Anxiety","5-10 min",
         "Systematically tense and release muscle groups from feet to face, releasing stored tension.",
         "pmr"),
    ]

    cols = st.columns(2)
    for i,(name,tags,dur,desc,key) in enumerate(techniques):
        with cols[i%2]:
            st.markdown(f"""
            <div class="card card-accent" style="margin:6px 0;min-height:120px">
                <div style="font-size:1.05rem;font-weight:500">{name}</div>
                <div style="margin:6px 0">
                    <span class="emotion-badge">{tags}</span>
                    <span class="emotion-badge">{dur}</span>
                </div>
                <div style="font-size:0.85rem;color:#8892a4;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Start →", key=f"start_{key}", use_container_width=True):
                st.session_state[f"active_technique"] = key
                st.rerun()

    # Active technique
    active = st.session_state.get("active_technique")
    if active:
        st.divider()
        if   active == "478":   render_breathing_guide("4-7-8",[(4,"🔵 Breathe IN"),(7,"⏸ HOLD"),(8,"🟢 Breathe OUT")], 3)
        elif active == "box":   render_breathing_guide("Box",  [(4,"🔵 IN"),(4,"⏸ HOLD"),(4,"🟢 OUT"),(4,"⏸ HOLD")], 4)
        elif active == "54321": render_grounding()
        elif active == "cbt":   render_cbt()
        elif active == "pmr":   render_pmr()
        if st.button("✖ Close technique"):
            del st.session_state["active_technique"]
            st.rerun()


def render_breathing_guide(name, steps, rounds):
    st.markdown(f"### {name} Breathing")
    st.info(f"Do {rounds} rounds. Focus only on your breath.")
    pattern = " → ".join([f"{label} ({secs}s)" for secs,label in steps])
    st.markdown(f"""
    <div class="card card-green">
        <div style="font-size:0.9rem;line-height:2">{pattern}</div>
        <div style="margin-top:12px;color:#8892a4;font-size:0.85rem">
            Repeat this {rounds} times. Even one round activates your calm response.
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("**Why it works:** Extending the exhale stimulates the vagus nerve, which signals "
                "your nervous system to shift from 'threat' mode into rest-and-digest mode.")


def render_grounding():
    st.markdown("### 5-4-3-2-1 Grounding")
    st.info("Anxiety lives in the future. This exercise pulls you back to the present moment.")
    prompts = [("👁 5 things you SEE","Look around — notice details you usually ignore."),
               ("✋ 4 things you can TOUCH","Feel textures, temperatures, surfaces."),
               ("👂 3 things you HEAR","Background sounds, your own breathing."),
               ("👃 2 things you SMELL","Subtle scents in the air around you."),
               ("👅 1 thing you TASTE","Whatever's present in your mouth right now.")]
    for sense,hint in prompts:
        with st.expander(sense):
            st.caption(hint)
            for i in range(int(sense[0])):
                st.text_input(f"  {i+1}.", key=f"ground_{sense[0]}_{i}", placeholder="…")
    st.success("✨ You're here. Present. Grounded.")


def render_cbt():
    st.markdown("### CBT Thought Record")
    st.info("Challenge distorted thinking by examining the actual evidence.")
    with st.form("cbt_form"):
        situation = st.text_area("What happened? (situation)", height=70)
        thought   = st.text_input("Automatic thought that came up")
        intensity = st.slider("How intense is this feeling? (0-100)", 0, 100, 70)
        ev_for    = st.text_area("Evidence FOR this thought", height=70)
        ev_ag     = st.text_area("Evidence AGAINST this thought", height=70)
        balanced  = st.text_area("More balanced alternative thought", height=70)
        new_int   = st.slider("New feeling intensity (0-100)", 0, 100, 50)
        submitted = st.form_submit_button("Save record")
    if submitted and thought:
        diff = intensity - new_int
        if diff > 0:
            st.success(f"✨ {diff}-point reduction in intensity. That's real progress.")
        elif diff == 0:
            st.info("Intensity unchanged — that's okay. Awareness itself is progress.")
        st.markdown(f"""
        <div class="card card-green" style="margin-top:12px">
            <div style="color:#8892a4;font-size:0.8rem;margin-bottom:4px">ORIGINAL THOUGHT [{intensity}%]</div>
            <div style="text-decoration:line-through;color:#8892a4">{thought}</div>
            <div style="color:#8892a4;font-size:0.8rem;margin:12px 0 4px">BALANCED THOUGHT [{new_int}%]</div>
            <div style="color:#e8ecf4">{balanced}</div>
        </div>""", unsafe_allow_html=True)


def render_pmr():
    st.markdown("### Progressive Muscle Relaxation")
    st.info("Work through each muscle group — tense for 5s, then fully release for 10s.")
    groups = ["Feet & toes","Calves","Thighs","Abdomen","Chest","Hands & forearms",
              "Upper arms","Shoulders","Neck","Face"]
    for i,group in enumerate(groups):
        st.markdown(f"""
        <div class="card" style="padding:10px 16px;margin:4px 0">
            <span style="color:#5b8ff0;font-weight:500">{i+1}.</span> {group}
            <span style="color:#8892a4;font-size:0.82rem;float:right">Tense 5s → Release 10s</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("**Why it works:** Deliberately tensing muscles makes the subsequent relaxation "
                "deeper, releasing physical tension your body holds during stress.")


# ─────────────────────────────────────────────────────────────────────────────
#  PROGRESS PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_progress(db: Database):
    try:
        import plotly.graph_objects as go
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    st.markdown("## 📈 Progress & Insights")

    streak = db.get_streak()
    stats7 = db.get_mood_stats(7)
    stats30= db.get_mood_stats(30)

    # Streak row
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="streak-number">{streak['current']}</div>
            <div class="metric-label">Day streak 🔥</div>
        </div>""", unsafe_allow_html=True)
    with c2: render_metric("Best streak",   str(streak["longest"]), "")
    with c3: render_metric("Total days",    str(streak["total"]),   "")
    with c4: render_metric("Avg mood (7d)", f"{stats7['avg_score']}/10","")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Goals summary
    active    = len(db.get_goals(completed=False))
    completed = len(db.get_goals(completed=True))
    total_g   = active + completed
    c1,c2,c3 = st.columns(3)
    with c1: render_metric("Goals set",       str(total_g),   "")
    with c2: render_metric("Goals completed", str(completed), "")
    with c3:
        pct = int(completed/total_g*100) if total_g else 0
        render_metric("Completion rate", f"{pct}%", "")

    # Gratitude count
    grats = db.get_gratitude(30)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: render_metric("Gratitudes (30d)", str(len(grats)), "")
    with c2: render_metric("Avg mood (30d)",   f"{stats30['avg_score']}/10", "")

    # Wellness score
    st.divider()
    st.markdown("#### Wellness Score")
    mood30  = stats30["avg_score"]
    s_score = int(min(100, max(0,
        (mood30/10)*40 +
        (min(streak["current"],30)/30)*30 +
        (min(len(grats),30)/30)*20 +
        (min(completed,10)/10)*10
    )))
    label = ("🌟 Thriving"   if s_score>=80 else
             "✅ Doing well" if s_score>=65 else
             "💛 Managing"   if s_score>=50 else
             "💙 Keep going" if s_score>=30 else
             "🙏 Needs care")
    st.markdown(f"""
    <div class="card card-accent">
        <div style="display:flex;align-items:center;gap:20px">
            <div>
                <div style="font-family:'DM Serif Display',serif;font-size:3rem;
                            color:#5b8ff0;line-height:1">{s_score}</div>
                <div style="color:#8892a4;font-size:0.78rem;text-transform:uppercase;
                            letter-spacing:0.08em">out of 100</div>
            </div>
            <div>
                <div style="font-size:1.2rem;font-weight:500">{label}</div>
                <div style="color:#8892a4;font-size:0.85rem;margin-top:6px;max-width:380px">
                    Weighted from mood average, streak, gratitude practice, and goal completion.
                </div>
            </div>
        </div>
        <div class="mood-bar-wrap" style="margin-top:16px">
            <div class="mood-bar-fill" style="width:{s_score}%"></div>
        </div>
    </div>""", unsafe_allow_html=True)

    if PLOTLY and stats30["count"] > 0:
        history = db.get_mood_history(30)
        if history:
            from collections import Counter
            emotions = [r["emotion"] for r in history if r.get("emotion")]
            em_count = Counter(emotions)
            fig = go.Figure(go.Bar(
                x=list(em_count.keys()),
                y=list(em_count.values()),
                marker_color=["#5b8ff0","#52c788","#f0b429","#e05b5b",
                              "#a78bfa","#38bdf8","#fb923c"][:len(em_count)],
                text=list(em_count.values()),
                textposition="auto",
            ))
            fig.update_layout(
                title="Emotions over last 30 days",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892a4", family="DM Sans"),
                height=260,
                margin=dict(l=0,r=0,t=36,b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#2a3347"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()
    init_state()
    db = get_db()
    render_sidebar(db)

    page = st.session_state.page

    if   page == "Chat":     page_chat(db)
    elif page == "Checkin":  page_checkin(db)
    elif page == "Chart":    page_chart(db)
    elif page == "Gratitude":page_gratitude(db)
    elif page == "Goals":    page_goals(db)
    elif page == "Toolkit":  page_toolkit()
    elif page == "Progress": page_progress(db)


if __name__ == "__main__":
    main()