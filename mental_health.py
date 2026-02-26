"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🌟 MINDFUL PRO - Advanced AI Mental Wellness Companion 🌟                 ║
║                                                                              ║
║   • Thinks deeply about context and nuance                                  ║
║   • Responds like a real therapist friend                                   ║
║   • Learns your patterns and adapts over time                               ║
║   • Evidence-based therapeutic interventions (CBT / DBT / Mindfulness)     ║
║   • LM Studio local AI  +  Groq cloud backup                               ║
║   • .env file support for API keys                                          ║
║                                                                              ║
║   Version: 2.1.0 | Production Ready                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import json
import os
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

# ── Load .env file automatically ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    _DOTENV_LOADED = True
except ImportError:
    _DOTENV_LOADED = False  # fall back to system env vars

# ── Optional NLP imports ──────────────────────────────────────────────────────
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central configuration — reads secrets from .env / environment variables."""

    # ── LM Studio ─────────────────────────────────────────────────────────────
    LM_STUDIO_BASE_URL   = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234")
    LM_STUDIO_MODELS_URL = f"{LM_STUDIO_BASE_URL}/v1/models"
    LM_STUDIO_CHAT_URL   = f"{LM_STUDIO_BASE_URL}/v1/chat/completions"

    # ── Groq ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
    # Use a current supported Groq model (llama-3.1-70b-versatile is deprecated)
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    # ── Database ──────────────────────────────────────────────────────────────
    DB_PATH = os.getenv("DB_PATH", "mindful_pro.db")

    # ── AI generation parameters ──────────────────────────────────────────────
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
    MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "500"))

    # ── App metadata ──────────────────────────────────────────────────────────
    APP_NAME = "MINDFUL PRO"
    VERSION  = "2.1.0"

    # ── Crisis hotlines ───────────────────────────────────────────────────────
    CRISIS_HOTLINES = {
        "india": [
            {"name": "AASRA",                 "number": "+91-9820466726",  "hours": "24/7"},
            {"name": "iCall",                 "number": "+91-9152987821",  "hours": "Mon-Sat 8am-10pm"},
            {"name": "NIMHANS",               "number": "+91-80-26995000", "hours": "24/7"},
            {"name": "Vandrevala Foundation", "number": "+91-9999666555",  "hours": "24/7"},
        ],
        "usa": [
            {"name": "988 Suicide & Crisis Lifeline", "number": "988",               "hours": "24/7"},
            {"name": "Crisis Text Line",              "number": "Text HOME to 741741","hours": "24/7"},
            {"name": "SAMHSA",                        "number": "1-800-662-4357",    "hours": "24/7"},
        ],
        "uk": [
            {"name": "Samaritans", "number": "116 123",       "hours": "24/7"},
            {"name": "CALM",       "number": "0800 58 58 58", "hours": "5pm-midnight"},
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LM STUDIO MODEL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class LMStudioModelManager:
    """
    Manages LM Studio connection, model listing, and interactive model selection.

    Features:
    - Auto-detect & list all models loaded in LM Studio
    - Interactive selection via the 'models' chat command
    - Per-session model persistence
    - Graceful fallback when LM Studio is offline
    """

    def __init__(self):
        self.selected_model: Optional[str] = None
        self.available_models: List[Dict]  = []
        self.is_connected: bool            = False
        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        try:
            resp = requests.get(Config.LM_STUDIO_MODELS_URL, timeout=4)
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                self.available_models = models
                self.is_connected     = True
                if models:
                    self.selected_model = models[0]["id"]
                return True
        except Exception:
            pass
        self.is_connected = False
        return False

    def refresh(self) -> bool:
        return self._connect()

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_models(self) -> List[Dict]:
        self.refresh()
        return self.available_models

    def print_models(self):
        models = self.list_models()

        print("\n╔══════════════════════════════════════════════════════════════════╗")
        print("║           🤖  LM STUDIO — AVAILABLE MODELS                      ║")
        print("╚══════════════════════════════════════════════════════════════════╝\n")

        if not self.is_connected:
            print("  ❌  Cannot reach LM Studio at", Config.LM_STUDIO_BASE_URL)
            print("      • Open LM Studio and start the Local Server on port 1234.\n")
            return

        if not models:
            print("  ⚠️  LM Studio is running but no model is loaded.")
            print("      Go to Local Server tab → select a model → Start Server.\n")
            return

        print(f"  {len(models)} model(s) loaded:\n")
        for idx, m in enumerate(models, 1):
            mid      = m.get("id", "unknown")
            owned_by = m.get("owned_by", "")
            created  = m.get("created", "")
            marker   = "▶" if mid == self.selected_model else " "
            print(f"  [{marker}] {idx}.  {mid}")
            if owned_by:
                print(f"           Owner   : {owned_by}")
            if created:
                try:
                    ts = datetime.fromtimestamp(int(created)).strftime("%Y-%m-%d")
                    print(f"           Created : {ts}")
                except Exception:
                    pass
            print()

        print(f"  Currently selected → {self.selected_model or 'None'}\n")

    # ── Interactive selection ─────────────────────────────────────────────────

    def interactive_select(self):
        self.print_models()

        if not self.is_connected or not self.available_models:
            return

        print("  Enter the number to switch models, or press Enter to keep current.\n")
        choice = input("  Choice: ").strip()

        if not choice:
            print(f"\n  ✔ Keeping: {self.selected_model}\n")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.available_models):
                self.selected_model = self.available_models[idx]["id"]
                print(f"\n  ✔ Model switched to: {self.selected_model}\n")
            else:
                print("\n  ⚠️  Invalid number — no change.\n")
        except ValueError:
            print("\n  ⚠️  Please enter a number — no change.\n")

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
    ) -> Tuple[str, float]:
        """Send messages to the selected LM Studio model. Returns (text, ms)."""
        if not self.is_connected or not self.selected_model:
            return "", 0.0

        payload = {
            "model":       self.selected_model,
            "messages":    messages,
            "temperature": temperature if temperature is not None else Config.TEMPERATURE,
            "max_tokens":  max_tokens  if max_tokens  is not None else Config.MAX_TOKENS,
        }

        try:
            t0   = time.time()
            resp = requests.post(Config.LM_STUDIO_CHAT_URL, json=payload, timeout=90)
            ms   = (time.time() - t0) * 1000

            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return text, ms

            return f"[LM Studio HTTP {resp.status_code}]", ms
        except requests.exceptions.Timeout:
            return "[LM Studio timed out — try a smaller model]", 0.0
        except Exception as e:
            return f"[LM Studio error: {e}]", 0.0

    # ── Status ────────────────────────────────────────────────────────────────

    def status_line(self) -> str:
        if not self.is_connected:
            return "❌  LM Studio: Offline  (open LM Studio → start server)"
        if not self.available_models:
            return "⚠️   LM Studio: Connected but no model loaded"
        return f"✅  LM Studio: {self.selected_model}"


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentBrain:

    def __init__(self):
        self.conversation_memory: List[Dict] = []
        self.user_profile = {
            "communication_style": "unknown",
            "recurring_themes":    [],
            "effective_techniques":[],
            "crisis_history":      [],
            "emotion_patterns":    {},
            "total_sessions":      0,
            "first_session":       None,
            "last_session":        None,
        }
        self._load_profile()

    def understand(self, text: str) -> Dict:
        surface  = self._analyze_surface(text)
        deeper   = self._analyze_deeper(text, surface)
        context  = self._analyze_context(text)
        patterns = self._analyze_patterns(surface, deeper)
        needs    = self._identify_needs(text, surface, deeper, context)

        understanding = {
            **surface, **deeper, **context, **patterns,
            "needs":     needs,
            "timestamp": datetime.now(),
        }
        self._learn(understanding)
        return understanding

    def _analyze_surface(self, text: str) -> Dict:
        t = text.lower()

        emotion_map = {
            "anxious":     (["anxious","nervous","worried","panic","scared","afraid","stressed",
                              "tense","uneasy","restless","fear","dread"],
                            ["very","extremely","really","so","too"]),
            "sad":         (["sad","depressed","down","hopeless","empty","lonely","miserable",
                              "worthless","numb","despair","grief"],
                            ["very","so","completely","totally"]),
            "angry":       (["angry","frustrated","furious","annoyed","mad","irritated",
                              "rage","pissed","resentful","bitter"],
                            ["so","really","extremely","incredibly"]),
            "overwhelmed": (["overwhelmed","drowning","too much","can't handle","breaking",
                              "crushing","swamped","buried"],
                            ["completely","totally","so"]),
            "confused":    (["confused","lost","don't know","unsure","uncertain","unclear"],
                            ["really","so","completely"]),
            "hopeful":     (["hopeful","better","improving","progress","good","happy"],
                            ["much","really","so"]),
        }

        intensities: Dict[str, int] = {}
        for emotion, (keywords, boosters) in emotion_map.items():
            count = sum(1 for kw in keywords if kw in t)
            if count:
                score = min(100, count * 30)
                if any(b in t for b in boosters):
                    score = min(100, score + 20)
                intensities[emotion] = score

        primary = max(intensities, key=intensities.get) if intensities else "neutral"

        is_question        = "?" in text or any(t.startswith(q) for q in
                             ["what","how","why","when","where","who","can you","could you",
                              "would you","do you","is there"])
        seeking_advice     = any(p in t for p in ["what should","how do i","help me","advice",
                                                   "suggest","recommend","what can i","how can i"])
        seeking_validation = any(p in t for p in ["is it okay","am i wrong","is this normal",
                                                   "does this make sense"])
        venting            = not is_question and len(text.split()) > 20 and not seeking_advice

        if TEXTBLOB_AVAILABLE:
            try:
                blob         = TextBlob(text)
                sentiment    = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            except Exception:
                sentiment, subjectivity = self._simple_sentiment(text), 0.5
        else:
            sentiment, subjectivity = self._simple_sentiment(text), 0.5

        return {
            "primary_emotion":    primary,
            "all_emotions":       list(intensities.keys()),
            "emotion_intensity":  intensities.get(primary, 0),
            "is_question":        is_question,
            "seeking_advice":     seeking_advice,
            "seeking_validation": seeking_validation,
            "venting":            venting,
            "sentiment_score":    sentiment,
            "subjectivity":       subjectivity,
            "word_count":         len(text.split()),
            "original_text":      text,
        }

    def _simple_sentiment(self, text: str) -> float:
        pos = ["good","great","happy","better","hope","wonderful","amazing","love"]
        neg = ["bad","awful","terrible","worse","horrible","hate","worst","sad"]
        t   = text.lower()
        p   = sum(1 for w in pos if w in t)
        n   = sum(1 for w in neg if w in t)
        return 0.0 if (p + n) == 0 else (p - n) / (p + n)

    def _analyze_deeper(self, text: str, surface: Dict) -> Dict:
        t = text.lower()

        distortion_map = {
            "catastrophizing":   ["disaster","terrible","awful","worst","ruin","destroy"],
            "black_and_white":   ["always","never","everyone","nobody","everything","nothing"],
            "mind_reading":      ["they think","probably thinks","must think","they don't like"],
            "fortune_telling":   ["will never","going to fail","won't work","never going to"],
            "labeling":          ["i'm a failure","i am such a","they're all"],
            "should_statements": ["should have","shouldn't have","i must","i have to"],
            "personalization":   ["my fault","because of me","i caused"],
        }
        distortions = [d for d, pats in distortion_map.items() if any(p in t for p in pats)]

        underlying_map = {
            "self_worth":    ["not good enough","failure","worthless","useless",
                              "inadequate","don't deserve","fake","fraud"],
            "control":       ["can't control","out of control","helpless","powerless"],
            "belonging":     ["don't belong","outsider","different from","nobody understands"],
            "perfectionism": ["must be perfect","have to be perfect","not allowed to fail"],
            "abandonment":   ["leave me","alone","reject","abandoned"],
            "safety":        ["unsafe","danger","threat","scared of"],
            "burnout":       ["exhausted","can't anymore","giving up","too tired","drained"],
        }
        issues = [iss for iss, pats in underlying_map.items() if any(p in t for p in pats)]

        needs_map = {
            "safety":    ["safe","danger","threat","scared","fear"],
            "belonging": ["alone","lonely","connect","friend","relationship"],
            "esteem":    ["respect","worth","value","confidence","proud"],
            "autonomy":  ["control","choice","decide","free","independent"],
            "meaning":   ["purpose","meaning","why","point","matter"],
        }
        unmet = [n for n, sigs in needs_map.items() if any(s in t for s in sigs)]

        return {
            "cognitive_distortions": distortions,
            "underlying_issues":     issues,
            "unmet_needs":           unmet,
            "complexity_level":      "high" if len(issues) > 2 else "moderate" if issues else "low",
        }

    def _analyze_context(self, text: str) -> Dict:
        depth       = len(self.conversation_memory)
        is_followup = depth > 0
        opening_up  = shutting_down = False

        if is_followup and depth >= 2:
            prev_len = len(self.conversation_memory[-1].get("original_text","").split())
            cur_len  = len(text.split())
            opening_up    = cur_len > prev_len * 1.3
            shutting_down = cur_len < prev_len * 0.5

        phases = {0: "introduction", 1: "introduction", 2: "building_rapport",
                  3: "building_rapport", 4: "exploring", 5: "exploring"}
        phase = phases.get(depth, "deeper_work")

        prev_emotions = [m.get("primary_emotion") for m in self.conversation_memory[-3:]] \
                        if is_followup else []

        return {
            "conversation_depth": depth,
            "is_followup":        is_followup,
            "opening_up":         opening_up,
            "shutting_down":      shutting_down,
            "phase":              phase,
            "previous_emotions":  prev_emotions,
        }

    def _analyze_patterns(self, surface: Dict, deeper: Dict) -> Dict:
        patterns = {}
        if self.user_profile["total_sessions"] >= 3:
            ec = self.user_profile["emotion_patterns"]
            if ec:
                patterns["recurring_emotion"] = max(ec, key=ec.get)
            themes = self.user_profile["recurring_themes"]
            if themes:
                tc = {}
                for th in themes:
                    tc[th] = tc.get(th, 0) + 1
                patterns["recurring_theme"] = max(tc, key=tc.get)
        return patterns

    def _identify_needs(self, text, surface, deeper, context) -> List[str]:
        needs = []
        if "safety" in deeper["underlying_issues"]:                               needs.append("safety")
        if surface["seeking_validation"] or surface["primary_emotion"] \
                in ["sad","overwhelmed","anxious"]:                                needs.append("validation")
        if surface["seeking_advice"]:                                             needs.append("practical_guidance")
        if deeper["cognitive_distortions"]:                                       needs.append("reframing")
        if surface["venting"]:                                                    needs.append("listening")
        if surface["is_question"]:                                                needs.append("information")
        if any(d in deeper["cognitive_distortions"]
               for d in ["catastrophizing","fortune_telling"]):                    needs.append("anxiety_management")
        if "self_worth" in deeper["underlying_issues"]:                           needs.append("self_compassion")
        if "burnout"    in deeper["underlying_issues"]:                           needs.append("boundaries")
        if not needs:                                                             needs.append("general_support")
        return needs

    def _learn(self, u: Dict):
        e = u["primary_emotion"]
        self.user_profile["emotion_patterns"][e] = \
            self.user_profile["emotion_patterns"].get(e, 0) + 1
        for iss in u.get("underlying_issues", []):
            self.user_profile["recurring_themes"].append(iss)
        self.user_profile["total_sessions"] += 1
        self.user_profile["last_session"]    = datetime.now().isoformat()
        if not self.user_profile["first_session"]:
            self.user_profile["first_session"] = datetime.now().isoformat()
        self.conversation_memory.append(u)
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
        self._save_profile()

    def _load_profile(self):
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            row  = conn.execute("SELECT profile_data FROM user_profile WHERE id=1").fetchone()
            if row:
                self.user_profile = json.loads(row[0])
            conn.close()
        except Exception:
            pass

    def _save_profile(self):
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO user_profile (id, profile_data, updated_at) VALUES (1,?,?)",
                    (json.dumps(self.user_profile), datetime.now().isoformat()),
                )
                conn.commit()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# INTELLIGENT RESPONDER
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentResponder:

    def __init__(self, lm_manager: LMStudioModelManager):
        self.lm         = lm_manager
        self.groq_ready = bool(Config.GROQ_API_KEY)

    def generate(self, understanding: Dict) -> Tuple[str, str]:
        """Returns (response_text, source_label)."""
        # 1) Try LM Studio
        if self.lm.is_connected and self.lm.selected_model:
            text, ms = self._try_lm_studio(understanding)
            if text:
                return text, f"LM Studio [{self.lm.selected_model}] ({ms:.0f}ms)"

        # 2) Try Groq
        if self.groq_ready:
            text, ms = self._try_groq(understanding)
            if text:
                return text, f"Groq [{Config.GROQ_MODEL}] ({ms:.0f}ms)"

        # 3) Smart rule-based fallback
        return self._intelligent_fallback(understanding), "Built-in (no AI connected)"

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_system_prompt(self, u: Dict) -> str:
        emotion     = u["primary_emotion"]
        intensity   = u.get("emotion_intensity", 0)
        phase       = u.get("phase", "introduction")
        needs       = u.get("needs", [])
        distortions = u.get("cognitive_distortions", [])

        base = """You are Mindful Pro, an emotionally intelligent mental health companion.

Personality:
• Warm, genuine, caring — like a wise friend who truly listens
• Professional but not clinical; explain psychology in plain language
• Adaptive — match the user's communication style
• Honest — acknowledge when professional help is needed

Core rules:
• Validate emotions FIRST, always
• Say "I notice / It sounds like / I hear" — never "you should"
• Give specific, actionable guidance — no generic platitudes
• Explain WHY techniques work (builds trust and motivation)
• End with a gentle question or prompt when appropriate
• NEVER diagnose, prescribe, or claim to replace a therapist"""

        context = f"""

Situation:
• Detected emotion  : {emotion} (intensity {intensity}%)
• Conversation phase: {phase}
• Primary needs     : {', '.join(needs) or 'general support'}"""

        if distortions:
            context += f"\n• Cognitive distortions : {', '.join(distortions)}"

        phase_hints = {
            "introduction":    "Be welcoming. Build safety. Don't rush to fix.",
            "building_rapport":"Show you're listening. Ask gentle follow-ups.",
            "exploring":       "Start offering insights. Gently challenge thinking.",
            "deeper_work":     "Explore root causes. Challenge more directly but kindly.",
        }
        context += f"\n\nApproach: {phase_hints.get(phase, 'Be supportive.')}"

        if "validation"        in needs: context += "\n• Validate feelings deeply BEFORE anything else."
        if "practical_guidance"in needs: context += "\n• Offer 2-3 specific, immediately actionable steps."
        if "reframing"         in needs: context += "\n• Use Socratic questions to guide a perspective shift."
        if "listening"         in needs: context += "\n• The person is venting — acknowledge first, fix second."

        style = """

Response style:
• 2-3 short paragraphs, ~150-200 words total
• Warm, conversational — like texting a therapist friend
• No jargon, no corporate-speak, no generic bullet-lists
• Contractions, varied sentence length — sound human
• If recommending a technique, briefly explain the psychology behind it"""

        return base + context + style

    # ── LM Studio ────────────────────────────────────────────────────────────

    def _try_lm_studio(self, u: Dict) -> Tuple[str, float]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(u)},
            {"role": "user",   "content": u["original_text"]},
        ]
        text, ms = self.lm.chat(messages)
        if text and not text.startswith("[LM Studio"):
            return text, ms
        return "", 0.0

    # ── Groq ─────────────────────────────────────────────────────────────────

    def _try_groq(self, u: Dict) -> Tuple[str, float]:
        messages = [
            {"role": "system", "content": self._build_system_prompt(u)},
            {"role": "user",   "content": u["original_text"]},
        ]
        try:
            t0   = time.time()
            resp = requests.post(
                Config.GROQ_URL,
                headers={
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       Config.GROQ_MODEL,
                    "messages":    messages,
                    "temperature": Config.TEMPERATURE,
                    "max_tokens":  Config.MAX_TOKENS,
                },
                timeout=30,
            )
            ms = (time.time() - t0) * 1000
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip(), ms
            err = resp.json().get("error", {}).get("message", resp.text[:120])
            print(f"\n  ⚠️  Groq error ({resp.status_code}): {err}\n")
        except requests.exceptions.Timeout:
            print("\n  ⚠️  Groq timed out.\n")
        except Exception as e:
            print(f"\n  ⚠️  Groq exception: {e}\n")
        return "", 0.0

    # ── Rule-based fallback ───────────────────────────────────────────────────

    def _intelligent_fallback(self, u: Dict) -> str:
        emotion     = u["primary_emotion"]
        intensity   = u.get("emotion_intensity", 0)
        needs       = u.get("needs", [])
        distortions = u.get("cognitive_distortions", [])
        issues      = u.get("underlying_issues", [])
        text        = u["original_text"]

        openings = {
            "anxious":     ["I can feel the anxiety in your words.",
                            "That racing-mind feeling is exhausting, isn't it?",
                            "Anxiety has its grip on you right now."],
            "sad":         ["There's a heaviness in what you're sharing.",
                            "I hear the pain in this.",
                            "That weight you're carrying is real."],
            "angry":       ["That frustration is real and valid.",
                            "Something important isn't being heard or respected."],
            "overwhelmed": ["It sounds like everything is hitting you at once.",
                            "That drowning feeling — I hear it."],
            "confused":    ["When thoughts are tangled it's hard to know where to start.",
                            "That uncertainty is uncomfortable."],
            "hopeful":     ["I can hear some light coming through.",
                            "There's something shifting for you."],
            "neutral":     ["I'm listening.", "Tell me more.", "I'm here with you in this."],
        }
        parts = [random.choice(openings.get(emotion, openings["neutral"]))]

        if "validation"        in needs and intensity > 60: parts.append(self._validation_text(emotion, issues))
        if "practical_guidance"in needs:                    parts.append(self._practical_text(emotion))
        if "reframing"         in needs and distortions:    parts.append(self._reframing_text(distortions[0]))
        if "information"       in needs:                    parts.append(self._information_text(text))
        if "listening"         in needs and len(parts) == 1:parts.append(self._listening_text())
        if len(parts) == 1:
            parts.append("What you're dealing with is real. Even talking about it takes courage.")

        closings = (
            ["What feels most important to you right now?",
             "Is there more you'd like to share?",
             "What would help most in this moment?"]
            if u.get("phase") == "introduction"
            else ["How does that land with you?", "Does this resonate?",
                  "Want to explore that further?"]
        )
        parts.append(random.choice(closings))
        return "\n\n".join(parts)

    def _validation_text(self, emotion, issues):
        v = {
            "anxious":     "Anxiety is your brain trying to protect you — it's not a flaw. What you're feeling makes complete sense.",
            "sad":         "Sadness this deep is real and valid. You're not broken — you're human dealing with something genuinely hard.",
            "angry":       "Your anger is information. It's telling you something important isn't right.",
            "overwhelmed": "When everything feels like too much, it IS too much. That's not weakness — that's reality.",
        }
        base = v.get(emotion, "What you're feeling is completely valid.")
        if "self_worth" in issues:
            base += " And your worth? That's not up for debate — it just IS."
        return base

    def _practical_text(self, emotion):
        g = {
            "anxious":     "Right now, try 5-4-3-2-1 grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. It works because anxiety lives in the future — grounding pulls you back to NOW.",
            "sad":         "Depression tricks you into doing nothing, which makes it worse. Try one tiny action — step outside for two minutes, send a single text. Motivation follows action, not the other way around.",
            "overwhelmed": "Brain-dump everything worrying you, then circle only what's genuinely urgent TODAY. Pick the single most important thing and do only that. Everything else can wait.",
            "angry":       "Cool down first — cold water on your wrists, 10 jumping jacks, or box breathing (4 counts in, 4 hold, 4 out, 4 hold). You can't think clearly while flooded with emotion.",
        }
        return g.get(emotion, "Let's break this down into one small, concrete next step you can take today.")

    def _reframing_text(self, distortion):
        r = {
            "catastrophizing":   "I notice worst-case thinking here. Ask yourself: what's the MOST LIKELY outcome — not worst, not best, just realistic?",
            "black_and_white":   "The words 'always' and 'never' are rarely accurate. What's a more nuanced take on this?",
            "mind_reading":      "You're predicting what others think. What's the actual evidence for that — not the fear, the evidence?",
            "fortune_telling":   "You're predicting a negative future. You've probably been wrong about predictions before. What if you're wrong about this one?",
            "labeling":          "You're applying a fixed label to yourself. You're not a static thing — you're a person going through something hard.",
            "should_statements": "Those 'shoulds' create pressure. Try replacing 'I should' with 'I could' — notice how the feeling shifts.",
            "personalization":   "You're owning things that aren't fully yours. What parts are actually within your control?",
        }
        return r.get(distortion, "I notice a pattern in how you're framing this. What would it look like to see it differently?")

    def _information_text(self, text):
        t = text.lower()
        if any(w in t for w in ["think","thought","mind"]):
            return "Your brain runs ~60,000 thoughts a day and defaults to negative ones — that's its problem-scanning mode, not reality. Thoughts aren't facts; they're guesses shaped by mood and past experience. And we can retrain those patterns."
        if any(w in t for w in ["why","point","meaning","purpose"]):
            return "Big 'why' questions often surface when we're in pain. Meaning isn't found passively — it's built through connection, contribution, and doing things that matter to you, even in small ways."
        return "That's worth exploring carefully. Want to dig into how this connects to what you're feeling?"

    def _listening_text(self):
        return random.choice([
            "I hear you. Sometimes the most helpful thing is just being truly heard.",
            "Thank you for trusting me with this. I'm here with you in it.",
            "That's a lot to carry. I'm listening — take all the space you need.",
            "You don't have to have it figured out. Talking about it already matters.",
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisDetector:
    _CRITICAL = ["suicide","kill myself","end my life","want to die","end it all",
                 "take my life","not worth living","better off dead"]
    _HIGH     = ["self harm","hurt myself","cut myself","burn myself","overdose",
                 "harm myself","injure myself"]
    _MODERATE = ["can't go on","give up on life","no point living","want to disappear"]

    @staticmethod
    def check(text: str, u: Dict) -> Dict:
        t        = text.lower()
        critical = any(kw in t for kw in CrisisDetector._CRITICAL)
        high     = any(kw in t for kw in CrisisDetector._HIGH)
        moderate = any(kw in t for kw in CrisisDetector._MODERATE)
        emotion  = u.get("primary_emotion")
        intensity= u.get("emotion_intensity", 0)

        if   critical or (high and intensity > 70):                         level = "critical"
        elif high     or (moderate and intensity > 80):                     level = "high"
        elif moderate and emotion in ["sad","hopeless"] and intensity > 70: level = "moderate"
        else:
            return {"is_crisis": False, "level": "none"}

        return {
            "is_crisis": True,
            "level":     level,
            "keywords":  [kw for kw in CrisisDetector._CRITICAL + CrisisDetector._HIGH if kw in t],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THERAPY TOOLKIT
# ═══════════════════════════════════════════════════════════════════════════════

class TherapyToolkit:

    @staticmethod
    def breathing_478():
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║     🫁  4-7-8 BREATHING  —  90-Second Calm Down          ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        print("Activates your parasympathetic nervous system — your body's natural calm-down switch.\n")
        input("Find a comfortable position and press Enter when ready...")
        print()
        for i in range(1, 4):
            print(f"  ── Round {i} / 3 ──")
            print("  Breathe IN through your nose ...  (4 counts)")
            time.sleep(4)
            print("  HOLD ..............................  (7 counts)")
            time.sleep(7)
            print("  Breathe OUT through your mouth ..  (8 counts)")
            time.sleep(8)
            print()
        print("✨ Notice any shift — even small changes matter.\n")

    @staticmethod
    def grounding_54321():
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║     🧘  5-4-3-2-1 GROUNDING TECHNIQUE                    ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        print("Anxiety lives in the future. This pulls you back to NOW.\n")
        prompts = [("👁  5 things you SEE", 5), ("✋ 4 things you can TOUCH", 4),
                   ("👂 3 things you HEAR", 3), ("👃 2 things you SMELL", 2),
                   ("👅 1 thing you TASTE", 1)]
        for prompt, count in prompts:
            print(prompt + ":")
            for i in range(count):
                input(f"   {i+1}. ")
            print()
        print("✨ You're here. You're present. You're grounded.\n")

    @staticmethod
    def thought_record():
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║     📝  CBT THOUGHT RECORD                                ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        situation   = input("Situation:\n→ ")
        thought     = input("\nAutomatic thought:\n→ ")
        emotion_pct = input("\nEmotion / intensity (0-100):\n→ ")
        ev_for      = input("\nEvidence FOR the thought:\n→ ")
        ev_against  = input("\nEvidence AGAINST the thought:\n→ ")
        alternative = input("\nBalanced alternative thought:\n→ ")
        new_emotion = input("\nNew emotion intensity (0-100):\n→ ")
        print("\n─── Summary ───")
        print(f"Original: {thought} [{emotion_pct}%]")
        print(f"Balanced: {alternative} [{new_emotion}%]")
        print("\n✨ Notice the shift. Even small reductions in intensity matter.\n")

    @staticmethod
    def show_all_techniques():
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║     🎯  THERAPY TECHNIQUES LIBRARY                        ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")
        menu = {
            "1": ("4-7-8 Breathing",     "Anxiety, stress, sleep issues",  "90 seconds"),
            "2": ("5-4-3-2-1 Grounding", "Panic, overwhelm, anxiety",      "3–5 minutes"),
            "3": ("CBT Thought Record",  "Negative/distorted thinking",    "5–10 minutes"),
        }
        for k, (name, use_for, duration) in menu.items():
            print(f"  {k}.  {name}")
            print(f"       For  : {use_for}")
            print(f"       Time : {duration}\n")
        choice = input("  Enter a number to begin, or press Enter to skip:\n  → ").strip()
        if   choice == "1": TherapyToolkit.breathing_478()
        elif choice == "2": TherapyToolkit.grounding_54321()
        elif choice == "3": TherapyToolkit.thought_record()
        elif choice:        print("\n  ✨ More techniques coming soon!\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:

    def __init__(self):
        self._init()

    def _init(self):
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp       TEXT    NOT NULL,
                        user_message    TEXT    NOT NULL,
                        emotion         TEXT,
                        intensity       INTEGER,
                        bot_response    TEXT,
                        crisis_detected INTEGER DEFAULT 0,
                        model_used      TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_profile (
                        id           INTEGER PRIMARY KEY,
                        profile_data TEXT    NOT NULL,
                        updated_at   TEXT    NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"  ⚠️  DB init error: {e}")

    def log(self, user_msg: str, u: Dict, response: str, crisis: bool, model: str = ""):
        try:
            with sqlite3.connect(Config.DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO conversations "
                    "(timestamp,user_message,emotion,intensity,bot_response,crisis_detected,model_used) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(), user_msg,
                     u.get("primary_emotion","?"), u.get("emotion_intensity", 0),
                     response, 1 if crisis else 0, model),
                )
                conn.commit()
        except Exception:
            pass

    def get_stats(self, days: int = 7) -> Dict:
        try:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            with sqlite3.connect(Config.DB_PATH) as conn:
                total      = conn.execute("SELECT COUNT(*) FROM conversations WHERE timestamp>?", (since,)).fetchone()[0]
                er         = conn.execute("SELECT emotion, COUNT(*) c FROM conversations WHERE timestamp>? GROUP BY emotion ORDER BY c DESC LIMIT 1", (since,)).fetchone()
                avg_int    = conn.execute("SELECT AVG(intensity) FROM conversations WHERE timestamp>?", (since,)).fetchone()[0] or 0
                crisis_cnt = conn.execute("SELECT COUNT(*) FROM conversations WHERE crisis_detected=1 AND timestamp>?", (since,)).fetchone()[0]
            return {
                "total": total,
                "top_emotion": er[0] if er else "N/A",
                "top_emotion_count": er[1] if er else 0,
                "avg_intensity": round(avg_int, 1),
                "crisis_count": crisis_cnt,
                "days": days,
            }
        except Exception:
            return {"total": 0, "top_emotion": "N/A", "top_emotion_count": 0,
                    "avg_intensity": 0, "crisis_count": 0, "days": days}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

class MindfulPro:

    def __init__(self):
        print("\n  Starting Mindful Pro v2.1.0 …")
        self.lm         = LMStudioModelManager()
        self.brain      = IntelligentBrain()
        self.responder  = IntelligentResponder(self.lm)
        self.crisis_det = CrisisDetector()
        self.toolkit    = TherapyToolkit()
        self.db         = DatabaseManager()
        self._welcome()

    def _welcome(self):
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                              ║")
        print("║         🌟  MINDFUL PRO  —  Mental Wellness Companion  v2.1.0                ║")
        print("║                                                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")

        if _DOTENV_LOADED:
            print("  📄  .env loaded successfully")
        else:
            print("  ℹ️   python-dotenv not installed (run: pip install python-dotenv)")

        print(f"  {self.lm.status_line()}")

        if self.responder.groq_ready:
            print(f"  ✅  Groq: Connected  [{Config.GROQ_MODEL}]")
        else:
            print("  ℹ️   Groq: No API key — add GROQ_API_KEY to your .env")

        if self.brain.user_profile["total_sessions"] > 0:
            print(f"\n  👋  Welcome back!  {self.brain.user_profile['total_sessions']} conversations so far.")

        print("\n" + "─" * 78 + "\n")

    def chat(self, user_input: str):
        u            = self.brain.understand(user_input)
        crisis_check = self.crisis_det.check(user_input, u)

        if crisis_check["is_crisis"]:
            self._handle_crisis(crisis_check)
            self.db.log(user_input, u, "[CRISIS RESPONSE]", True, "crisis-handler")
            return

        response, source = self.responder.generate(u)
        self._display_response(u, response, source)
        self.db.log(user_input, u, response, False, source)

    def _handle_crisis(self, crisis: Dict):
        print("\n" + "═" * 78)
        print("\n  🚨  I'm really concerned about what you're sharing right now.\n")
        print("      Your safety is the priority. Please reach out immediately:\n")
        for region, lines in Config.CRISIS_HOTLINES.items():
            flag = {"india": "🇮🇳  INDIA", "usa": "🇺🇸  USA", "uk": "🇬🇧  UK"}.get(region, region.upper())
            print(f"  {flag}:")
            for h in lines:
                print(f"     • {h['name']}: {h['number']}  ({h['hours']})")
            print()
        print("  💙  You matter. These feelings can and do change.")
        print("      Please reach out — right now if possible.\n")
        print("═" * 78 + "\n")

    def _display_response(self, u: Dict, response: str, source: str):
        print("\n" + "─" * 78)

        emotion   = u["primary_emotion"]
        intensity = u.get("emotion_intensity", 0)

        if emotion != "neutral" and intensity > 30:
            emojis = {"anxious": "😰", "sad": "😢", "angry": "😠",
                      "overwhelmed": "😓", "confused": "🤔", "hopeful": "🙂"}
            bar = "█" * int(intensity / 10) + "░" * (10 - int(intensity / 10))
            print(f"\n  {emojis.get(emotion,'💬')}  {emotion.title()}  [{bar}] {intensity}%")

        print(f"  🤖  {source}\n")
        print(self._wrap(response))

        needs = u.get("needs", [])
        if "anxiety_management" in needs or emotion == "anxious":
            print("\n  💡  Tip: type 'breathe' for a 90-second calm-down exercise")
        elif "validation" in needs and intensity > 70:
            print("\n  💡  Tip: type 'techniques' to see evidence-based coping tools")

        print("\n" + "─" * 78 + "\n")

    def _wrap(self, text: str, width: int = 74) -> str:
        out = []
        for para in text.split("\n\n"):
            if para.lstrip().startswith(("**", "•", "#", "-", "1.", "2.", "3.")):
                out.append(para)
                continue
            words, line, lines = para.split(), "", []
            for w in words:
                if len(line) + len(w) + 1 <= width:
                    line += w + " "
                else:
                    if line: lines.append(line.rstrip())
                    line = w + " "
            if line: lines.append(line.rstrip())
            out.append("\n".join(lines))
        return "\n\n".join(out)

    def show_stats(self):
        s = self.db.get_stats(7)
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                     📊  YOUR WELLNESS INSIGHTS  (last 7 days)                ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
        if s["total"] < 3:
            print(f"  Conversations this week : {s['total']}")
            print(f"  {3 - s['total']} more needed to generate pattern insights.\n")
            return
        print(f"  Total conversations  : {s['total']}")
        print(f"  Most common emotion  : {s['top_emotion'].title()} ({s['top_emotion_count']}×)")
        print(f"  Average intensity    : {s['avg_intensity']}%")
        if s["crisis_count"]:
            print(f"  ⚠️   Crisis moments  : {s['crisis_count']}  — please consider professional support")
        penalty = (20 if s["avg_intensity"] > 70 else 10 if s["avg_intensity"] > 50 else 0)
        score   = max(0, min(100, 70 - penalty - s["crisis_count"] * 15
                             - (15 if s["top_emotion"] in ["sad","hopeless","anxious"] else 0)))
        label   = ("Doing well!" if score >= 70 else "Managing." if score >= 50
                   else "Struggling — extra support recommended." if score >= 30
                   else "Needs support — please talk to a professional.")
        print(f"\n  🎯  Wellness Score: {score}/100 — {label}\n")

    def interactive(self):
        print("  Just talk, or use a command:\n")
        print("    💬  Anything         — share what's on your mind")
        print("    🤖  models           — list & switch LM Studio model")
        print("    🫁  breathe          — 90-second breathing exercise")
        print("    🎯  techniques       — therapy tools menu")
        print("    📊  stats            — your wellness insights")
        print("    ❓  help             — all features")
        print("    👋  exit             — end session\n")

        while True:
            try:
                raw = input("You: ").strip()
                if not raw:
                    continue
                cmd = raw.lower()

                if   cmd == "exit":
                    print("\n  💙  Take good care of yourself. I'm here whenever you need me.\n")
                    break
                elif cmd in ("models","model","select model","change model"):
                    self.lm.interactive_select()
                elif cmd in ("breathe","breathing","breath"):
                    self.toolkit.breathing_478()
                elif cmd == "ground":
                    self.toolkit.grounding_54321()
                elif cmd in ("techniques","tools","exercises"):
                    self.toolkit.show_all_techniques()
                elif cmd in ("stats","progress","insights"):
                    self.show_stats()
                elif cmd == "help":
                    self._show_help()
                else:
                    self.chat(raw)

            except KeyboardInterrupt:
                print("\n\n  💙  Take care!\n")
                break
            except Exception as e:
                print(f"\n  ⚠️  Unexpected error: {e}\n")

    def _show_help(self):
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                        📖  MINDFUL PRO — ALL FEATURES                        ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
        sections = [
            ("🤖 LM STUDIO", [
                "Type 'models' to list all models loaded in LM Studio",
                "Select any model interactively — persists for the session",
                "Active model is shown with every response (with latency)",
                "Groq is used as cloud backup when LM Studio is offline",
            ]),
            ("📄 .ENV FILE", [
                "GROQ_API_KEY=your_key_here",
                "Optional overrides: GROQ_MODEL, TEMPERATURE, MAX_TOKENS, DB_PATH",
                "Requires: pip install python-dotenv",
            ]),
            ("🧠 INTELLIGENCE", [
                "Multi-layer understanding: emotion, distortions, unmet needs",
                "Learns emotion patterns and adapts over sessions",
                "Phase-aware responses (intro → rapport → exploration → deep work)",
                "Crisis detection with immediate helpline resources",
            ]),
            ("🎯 THERAPY TOOLS", [
                "4-7-8 Breathing          → 'breathe'",
                "5-4-3-2-1 Grounding     → 'ground'",
                "CBT Thought Record       → 'techniques' → 3",
            ]),
            ("📊 TRACKING", [
                "Wellness score (0-100) based on 7-day patterns",
                "All data stored locally in SQLite — nothing leaves your machine",
            ]),
        ]
        for title, items in sections:
            print(f"  {title}")
            for item in items:
                print(f"     • {item}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        MindfulPro().interactive()
    except KeyboardInterrupt:
        print("\n\n  💙  Take care of yourself!\n")
    except Exception as e:
        print(f"\n  ❌  Critical error: {e}")
        print("      Please restart the application.\n")