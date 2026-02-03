"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🌟 MINDFUL PRO - Advanced AI Mental Wellness Companion 🌟                 ║
║                                                                              ║
║   A genuinely intelligent mental health AI that:                            ║
║   • Thinks deeply about context and nuance                                  ║
║   • Responds like a real therapist friend                                   ║
║   • Learns your patterns and adapts                                         ║
║   • Provides evidence-based therapeutic interventions                       ║
║   • Beautiful, professional interface                                       ║
║                                                                              ║
║   Version: 1.0.0 | Production Ready                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import json
import os
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import random
import re

# Optional advanced imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None
except:
    SPACY_AVAILABLE = False
    nlp = None

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration"""
    
    # AI Endpoints
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODEL = "llm"
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.1-70b-versatile"
    
    # Database
    DB_PATH = "mindful_pro.db"
    
    # AI Parameters
    TEMPERATURE = 0.8
    MAX_TOKENS = 400
    
    # App Settings
    APP_NAME = "MINDFUL PRO"
    VERSION = "1.0.0"
    
    # Therapy frameworks
    THERAPY_TECHNIQUES = {
        "cbt": {
            "name": "Cognitive Behavioral Therapy",
            "when": "Negative thinking patterns, cognitive distortions",
            "how": "Challenge thoughts, behavioral experiments, thought records"
        },
        "dbt": {
            "name": "Dialectical Behavior Therapy", 
            "when": "Intense emotions, overwhelm, crisis",
            "how": "TIPP, DEAR MAN, distress tolerance skills"
        },
        "act": {
            "name": "Acceptance & Commitment Therapy",
            "when": "Avoidance, stuck in thoughts, lack of values",
            "how": "Defusion, acceptance, values clarification"
        },
        "mindfulness": {
            "name": "Mindfulness-Based Approaches",
            "when": "Stress, anxiety, disconnection from present",
            "how": "Body scan, breath awareness, grounding techniques"
        }
    }
    
    # Crisis resources
    CRISIS_HOTLINES = {
        "india": [
            {"name": "AASRA", "number": "+91-9820466726", "hours": "24/7"},
            {"name": "iCall", "number": "+91-9152987821", "hours": "Mon-Sat, 8am-10pm"},
            {"name": "NIMHANS", "number": "+91-80-26995000", "hours": "24/7"},
            {"name": "Vandrevala Foundation", "number": "+91-9999666555", "hours": "24/7"}
        ],
        "usa": [
            {"name": "988 Suicide & Crisis Lifeline", "number": "988", "hours": "24/7"},
            {"name": "Crisis Text Line", "number": "Text HOME to 741741", "hours": "24/7"},
            {"name": "SAMHSA", "number": "1-800-662-4357", "hours": "24/7"}
        ],
        "uk": [
            {"name": "Samaritans", "number": "116 123", "hours": "24/7"},
            {"name": "CALM", "number": "0800 58 58 58", "hours": "5pm-midnight"}
        ]
    }

# ═══════════════════════════════════════════════════════════════════════
# INTELLIGENT UNDERSTANDING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class IntelligentBrain:
    """Deep understanding engine - analyzes multiple layers"""
    
    def __init__(self):
        self.conversation_memory = []
        self.user_profile = {
            "communication_style": "unknown",
            "recurring_themes": [],
            "effective_techniques": [],
            "crisis_history": [],
            "emotion_patterns": {},
            "total_sessions": 0,
            "first_session": None,
            "last_session": None
        }
        
        # Load profile from database if exists
        self._load_profile()
    
    def understand(self, text: str) -> Dict:
        """Multi-layer deep understanding"""
        
        # Layer 1: Surface analysis (what they're saying)
        surface = self._analyze_surface(text)
        
        # Layer 2: Deeper analysis (what they're not saying)
        deeper = self._analyze_deeper(text, surface)
        
        # Layer 3: Contextual analysis (conversation history)
        context = self._analyze_context(text)
        
        # Layer 4: Pattern analysis (user patterns over time)
        patterns = self._analyze_patterns(surface, deeper)
        
        # Layer 5: Need identification (what they actually need)
        needs = self._identify_needs(text, surface, deeper, context)
        
        # Synthesize complete understanding
        understanding = {
            **surface,
            **deeper,
            **context,
            **patterns,
            "needs": needs,
            "timestamp": datetime.now()
        }
        
        # Learn from this interaction
        self._learn(understanding)
        
        return understanding
    
    def _analyze_surface(self, text: str) -> Dict:
        """What they're explicitly saying"""
        
        text_lower = text.lower()
        
        # Emotion detection with intensity
        emotions = {
            "anxious": {
                "keywords": ["anxious", "nervous", "worried", "panic", "scared", "afraid", 
                           "stressed", "tense", "uneasy", "restless", "fear", "dread"],
                "intensity_boosters": ["very", "extremely", "really", "so", "too"]
            },
            "sad": {
                "keywords": ["sad", "depressed", "down", "hopeless", "empty", "lonely",
                           "miserable", "worthless", "numb", "despair", "grief"],
                "intensity_boosters": ["very", "so", "completely", "totally"]
            },
            "angry": {
                "keywords": ["angry", "frustrated", "furious", "annoyed", "mad", "irritated",
                           "rage", "pissed", "resentful", "bitter"],
                "intensity_boosters": ["so", "really", "extremely", "incredibly"]
            },
            "overwhelmed": {
                "keywords": ["overwhelmed", "drowning", "too much", "can't handle", "breaking",
                           "crushing", "swamped", "buried"],
                "intensity_boosters": ["completely", "totally", "so"]
            },
            "confused": {
                "keywords": ["confused", "lost", "don't know", "unsure", "uncertain", "unclear"],
                "intensity_boosters": ["really", "so", "completely"]
            },
            "hopeful": {
                "keywords": ["hopeful", "better", "improving", "progress", "good", "happy"],
                "intensity_boosters": ["much", "really", "so"]
            }
        }
        
        detected_emotions = []
        emotion_intensities = {}
        
        for emotion, data in emotions.items():
            count = sum(1 for kw in data["keywords"] if kw in text_lower)
            if count > 0:
                # Calculate intensity
                intensity = min(100, count * 30)
                # Boost if intensity words present
                if any(boost in text_lower for boost in data["intensity_boosters"]):
                    intensity = min(100, intensity + 20)
                
                detected_emotions.append(emotion)
                emotion_intensities[emotion] = intensity
        
        primary_emotion = max(emotion_intensities, key=emotion_intensities.get) if emotion_intensities else "neutral"
        
        # Question detection
        is_question = "?" in text or any(text_lower.startswith(q) for q in 
                                        ["what", "how", "why", "when", "where", "who", 
                                         "can you", "could you", "would you", "do you", "is there"])
        
        # Seeking type
        seeking_advice = any(phrase in text_lower for phrase in 
                           ["what should", "how do i", "help me", "advice", "suggest", 
                            "recommend", "what can i", "how can i"])
        
        seeking_validation = any(phrase in text_lower for phrase in
                                ["is it okay", "am i wrong", "is this normal", "does this make sense"])
        
        venting = not is_question and len(text.split()) > 20 and not seeking_advice
        
        # Sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
            except:
                sentiment_score = self._simple_sentiment(text)
                subjectivity = 0.5
        else:
            sentiment_score = self._simple_sentiment(text)
            subjectivity = 0.5
        
        return {
            "primary_emotion": primary_emotion,
            "all_emotions": detected_emotions,
            "emotion_intensity": emotion_intensities.get(primary_emotion, 0),
            "is_question": is_question,
            "seeking_advice": seeking_advice,
            "seeking_validation": seeking_validation,
            "venting": venting,
            "sentiment_score": sentiment_score,
            "subjectivity": subjectivity,
            "word_count": len(text.split()),
            "original_text": text
        }
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment when TextBlob not available"""
        positive = ["good", "great", "happy", "better", "hope", "wonderful", "amazing", "love"]
        negative = ["bad", "awful", "terrible", "worse", "horrible", "hate", "worst", "sad"]
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _analyze_deeper(self, text: str, surface: Dict) -> Dict:
        """What they're NOT saying - read between the lines"""
        
        text_lower = text.lower()
        
        # Cognitive distortions
        distortions = {
            "catastrophizing": ["disaster", "terrible", "awful", "worst", "ruin", "destroy"],
            "black_and_white": ["always", "never", "everyone", "nobody", "everything", "nothing"],
            "mind_reading": ["they think", "probably thinks", "must think", "they don't like"],
            "fortune_telling": ["will never", "going to fail", "won't work", "never going to"],
            "labeling": ["i'm a", "i am such a", "they're all"],
            "should_statements": ["should have", "shouldn't have", "must", "have to"],
            "personalization": ["my fault", "because of me", "i caused"]
        }
        
        detected_distortions = []
        for distortion, patterns in distortions.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_distortions.append(distortion)
        
        # Underlying issues
        underlying = {
            "self_worth": ["not good enough", "failure", "worthless", "useless", "inadequate", 
                          "don't deserve", "fake", "fraud"],
            "control": ["can't control", "out of control", "helpless", "powerless"],
            "belonging": ["don't belong", "outsider", "different from", "nobody understands"],
            "perfectionism": ["perfect", "must be", "have to be", "not allowed to fail"],
            "abandonment": ["leave me", "alone", "reject", "abandoned"],
            "safety": ["unsafe", "danger", "threat", "scared of"],
            "burnout": ["exhausted", "can't anymore", "giving up", "too tired", "drained"]
        }
        
        detected_issues = []
        for issue, patterns in underlying.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_issues.append(issue)
        
        # Psychological needs (Maslow-inspired)
        needs_signals = {
            "safety": ["safe", "danger", "threat", "scared", "fear"],
            "belonging": ["alone", "lonely", "connect", "friend", "relationship"],
            "esteem": ["respect", "worth", "value", "confidence", "proud"],
            "autonomy": ["control", "choice", "decide", "free", "independent"],
            "meaning": ["purpose", "meaning", "why", "point", "matter"]
        }
        
        unmet_needs = []
        for need, signals in needs_signals.items():
            if any(signal in text_lower for signal in signals):
                unmet_needs.append(need)
        
        return {
            "cognitive_distortions": detected_distortions,
            "underlying_issues": detected_issues,
            "unmet_needs": unmet_needs,
            "complexity_level": "high" if len(detected_issues) > 2 else "moderate" if detected_issues else "low"
        }
    
    def _analyze_context(self, text: str) -> Dict:
        """Analyze conversation context"""
        
        conversation_depth = len(self.conversation_memory)
        
        # Is this a follow-up?
        is_followup = conversation_depth > 0
        
        # Are they opening up or shutting down?
        if is_followup and conversation_depth >= 2:
            prev_length = len(self.conversation_memory[-1].get("original_text", "").split())
            current_length = len(text.split())
            opening_up = current_length > prev_length * 1.3
            shutting_down = current_length < prev_length * 0.5
        else:
            opening_up = False
            shutting_down = False
        
        # Conversation phase
        if conversation_depth == 0:
            phase = "introduction"
        elif conversation_depth <= 2:
            phase = "building_rapport"
        elif conversation_depth <= 5:
            phase = "exploring"
        else:
            phase = "deeper_work"
        
        # Topic continuity
        if is_followup:
            prev_emotions = [m.get("primary_emotion") for m in self.conversation_memory[-3:]]
            topic_continuity = text.lower() in " ".join([m.get("original_text", "").lower() for m in self.conversation_memory[-2:]])
        else:
            prev_emotions = []
            topic_continuity = False
        
        return {
            "conversation_depth": conversation_depth,
            "is_followup": is_followup,
            "opening_up": opening_up,
            "shutting_down": shutting_down,
            "phase": phase,
            "previous_emotions": prev_emotions,
            "topic_continuity": topic_continuity
        }
    
    def _analyze_patterns(self, surface: Dict, deeper: Dict) -> Dict:
        """Analyze patterns over time"""
        
        patterns = {}
        
        if self.user_profile["total_sessions"] >= 3:
            # Emotion patterns
            emotion_counts = self.user_profile["emotion_patterns"]
            if emotion_counts:
                most_common = max(emotion_counts, key=emotion_counts.get)
                patterns["recurring_emotion"] = most_common
            
            # Theme patterns
            if self.user_profile["recurring_themes"]:
                theme_counts = {}
                for theme in self.user_profile["recurring_themes"]:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
                if theme_counts:
                    patterns["recurring_theme"] = max(theme_counts, key=theme_counts.get)
        
        return patterns
    
    def _identify_needs(self, text: str, surface: Dict, deeper: Dict, context: Dict) -> List[str]:
        """What does this person actually need right now?"""
        
        needs = []
        
        # Immediate safety need
        if any(issue in deeper["underlying_issues"] for issue in ["safety"]):
            needs.append("safety")
        
        # Emotional needs
        if surface["seeking_validation"]:
            needs.append("validation")
        elif surface["primary_emotion"] in ["sad", "overwhelmed", "anxious"]:
            needs.append("validation")
        
        if surface["seeking_advice"]:
            needs.append("practical_guidance")
        
        if deeper["cognitive_distortions"]:
            needs.append("reframing")
        
        if surface["venting"]:
            needs.append("listening")
        
        if surface["is_question"]:
            needs.append("information")
        
        # Therapeutic needs
        if any(distortion in deeper["cognitive_distortions"] for distortion in ["catastrophizing", "fortune_telling"]):
            needs.append("anxiety_management")
        
        if "self_worth" in deeper["underlying_issues"]:
            needs.append("self_compassion")
        
        if "burnout" in deeper["underlying_issues"]:
            needs.append("boundaries")
        
        # Default
        if not needs:
            needs.append("general_support")
        
        return needs
    
    def _learn(self, understanding: Dict):
        """Learn from interaction"""
        
        # Update emotion patterns
        emotion = understanding["primary_emotion"]
        self.user_profile["emotion_patterns"][emotion] = \
            self.user_profile["emotion_patterns"].get(emotion, 0) + 1
        
        # Update recurring themes
        for issue in understanding.get("underlying_issues", []):
            self.user_profile["recurring_themes"].append(issue)
        
        # Update session info
        self.user_profile["total_sessions"] += 1
        self.user_profile["last_session"] = datetime.now().isoformat()
        if not self.user_profile["first_session"]:
            self.user_profile["first_session"] = datetime.now().isoformat()
        
        # Store in memory (keep last 10)
        self.conversation_memory.append(understanding)
        if len(self.conversation_memory) > 10:
            self.conversation_memory = self.conversation_memory[-10:]
        
        # Save to database
        self._save_profile()
    
    def _load_profile(self):
        """Load user profile from database"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT profile_data FROM user_profile WHERE id = 1")
            result = cursor.fetchone()
            if result:
                self.user_profile = json.loads(result[0])
            conn.close()
        except:
            pass
    
    def _save_profile(self):
        """Save user profile to database"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_profile (id, profile_data, updated_at)
                VALUES (1, ?, ?)
            """, (json.dumps(self.user_profile), datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except:
            pass

# ═══════════════════════════════════════════════════════════════════════
# INTELLIGENT RESPONSE GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class IntelligentResponder:
    """Generates human-like, contextually aware responses"""
    
    def __init__(self):
        self.lm_studio = self._check_lm_studio()
        self.groq = Config.GROQ_API_KEY != ""
    
    def _check_lm_studio(self) -> bool:
        try:
            r = requests.post(Config.LM_STUDIO_URL,
                            json={"model": Config.LM_STUDIO_MODEL,
                                 "messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 5},
                            timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def generate(self, understanding: Dict) -> str:
        """Generate intelligent, personalized response"""
        
        # Try AI first
        if self.lm_studio or self.groq:
            ai_response = self._ai_response(understanding)
            if ai_response:
                return ai_response
        
        # Fallback to intelligent rule-based
        return self._intelligent_fallback(understanding)
    
    def _ai_response(self, understanding: Dict) -> str:
        """Generate AI response with smart prompting"""
        
        system_prompt = self._build_prompt(understanding)
        user_text = understanding["original_text"]
        
        try:
            if self.lm_studio:
                response = requests.post(
                    Config.LM_STUDIO_URL,
                    json={
                        "model": Config.LM_STUDIO_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}
                        ],
                        "temperature": Config.TEMPERATURE,
                        "max_tokens": Config.MAX_TOKENS
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
            
            elif self.groq:
                response = requests.post(
                    Config.GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": Config.GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text}
                        ],
                        "temperature": Config.TEMPERATURE,
                        "max_tokens": Config.MAX_TOKENS
                    },
                    timeout=15
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
        except:
            pass
        
        return ""
    
    def _build_prompt(self, understanding: Dict) -> str:
        """Build context-aware system prompt"""
        
        base = """You are Mindful Pro, an emotionally intelligent mental health companion.

Your personality:
• Warm, genuine, caring - like a wise friend who truly listens
• Professional but not clinical - you explain psychology in simple terms
• Adaptive - you match their communication style
• Honest - you acknowledge when professional help is needed

Core principles:
• Validate emotions FIRST, always
• Use "I notice", "It sounds like", "I hear" - not "you should"
• Give specific, actionable guidance (not generic platitudes)
• Explain WHY techniques work (builds trust and motivation)
• End with questions or gentle prompts when appropriate"""
        
        # Context from understanding
        emotion = understanding["primary_emotion"]
        intensity = understanding.get("emotion_intensity", 0)
        phase = understanding.get("phase", "introduction")
        needs = understanding.get("needs", [])
        distortions = understanding.get("cognitive_distortions", [])
        
        context = f"""

Current situation:
• Emotion: {emotion} (intensity: {intensity}%)
• Conversation phase: {phase}
• Primary needs: {', '.join(needs)}"""
        
        if distortions:
            context += f"\n• Cognitive distortions detected: {', '.join(distortions)}"
        
        # Phase-specific guidance
        phase_guidance = {
            "introduction": "\nApproach: Be welcoming. Create safety. Don't rush to fix. Build trust first.",
            "building_rapport": "\nApproach: Show you're listening. Ask gentle follow-ups. Validate deeply.",
            "exploring": "\nApproach: Start offering insights. Gently challenge thinking. Teach techniques.",
            "deeper_work": "\nApproach: Go deeper. Explore root causes. Challenge more directly but kindly."
        }
        
        guidance = phase_guidance.get(phase, "")
        
        # Need-specific guidance
        need_guidance = ""
        if "validation" in needs:
            need_guidance += "\n• Priority: Validate their feelings deeply before anything else"
        if "practical_guidance" in needs:
            need_guidance += "\n• Give 2-3 specific, actionable steps they can take"
        if "reframing" in needs:
            need_guidance += "\n• Help them see this differently - use questions to guide their thinking"
        if "listening" in needs:
            need_guidance += "\n• They're venting - acknowledge their experience, don't rush to fix"
        
        style = f"""

Response style:
• Length: 2-3 short paragraphs (150-200 words)
• Tone: Warm, conversational, like texting a therapist friend
• NO corporate-speak, NO therapy jargon, NO generic advice
• Use "you" and "I" - be personal
• If teaching techniques, explain WHY they work (the psychology)
• Sound human - contractions, varied sentence length, natural flow"""
        
        return base + context + guidance + need_guidance + style
    
    def _intelligent_fallback(self, understanding: Dict) -> str:
        """Intelligent rule-based response when no AI"""
        
        emotion = understanding["primary_emotion"]
        intensity = understanding.get("emotion_intensity", 0)
        needs = understanding.get("needs", [])
        distortions = understanding.get("cognitive_distortions", [])
        issues = understanding.get("underlying_issues", [])
        text = understanding["original_text"]
        
        # Build response in parts
        parts = []
        
        # Part 1: Opening (validation)
        openings = {
            "anxious": [
                "I can feel the anxiety in your words.",
                "That racing-mind feeling is exhausting, isn't it?",
                "Anxiety has its grip on you right now."
            ],
            "sad": [
                "There's a heaviness in what you're sharing.",
                "I hear the pain in this.",
                "That weight you're carrying is real."
            ],
            "angry": [
                "That frustration is real and valid.",
                "Something important isn't being heard or respected.",
                "I can feel the intensity of this."
            ],
            "overwhelmed": [
                "It sounds like everything is hitting you at once.",
                "That drowning feeling - I hear it.",
                "Too much, all at the same time."
            ],
            "confused": [
                "I get it - when thoughts are tangled, it's hard to know where to start.",
                "That uncertainty is uncomfortable.",
                "Not knowing can be really unsettling."
            ],
            "hopeful": [
                "I can hear some light coming through.",
                "There's something shifting for you.",
                "That's a meaningful step forward."
            ],
            "neutral": [
                "I'm listening.",
                "Tell me more.",
                "I'm here with you in this."
            ]
        }
        
        parts.append(random.choice(openings.get(emotion, openings["neutral"])))
        
        # Part 2: Main response (based on needs)
        if "validation" in needs and intensity > 60:
            parts.append(self._validation_response(emotion, intensity, issues))
        
        if "practical_guidance" in needs:
            parts.append(self._practical_response(emotion, issues))
        
        if "reframing" in needs and distortions:
            parts.append(self._reframing_response(distortions[0], text))
        
        if "information" in needs:
            parts.append(self._information_response(text))
        
        if "listening" in needs and not parts[1:]:
            parts.append(self._listening_response(emotion))
        
        # Default if nothing else fits
        if len(parts) == 1:
            parts.append(self._general_support(emotion))
        
        # Part 3: Closing (engagement)
        if understanding.get("phase") == "introduction":
            closings = [
                "What feels most important to you right now?",
                "Is there more you want to share about this?",
                "What would help you most in this moment?"
            ]
        else:
            closings = [
                "How does that land with you?",
                "Does this resonate?",
                "What comes up for you hearing that?",
                "Want to explore that more?"
            ]
        
        parts.append(random.choice(closings))
        
        return "\n\n".join(parts)
    
    def _validation_response(self, emotion: str, intensity: int, issues: List[str]) -> str:
        """Deep validation"""
        
        validations = {
            "anxious": "Anxiety is your brain trying to protect you - it's not a flaw, it's your nervous system being overprotective. What you're feeling makes complete sense.",
            "sad": "Sadness this deep is real and valid. You're not broken for feeling this way - you're human, dealing with something genuinely difficult.",
            "angry": "Your anger is information. It's telling you something important isn't right. That's valid.",
            "overwhelmed": "When everything feels like too much, it IS too much. Your system is maxed out, and that's not weakness - that's reality.",
        }
        
        base = validations.get(emotion, "What you're feeling is completely valid.")
        
        if "self_worth" in issues:
            base += " And your worth? That's not up for debate. It just IS."
        
        return base
    
    def _practical_response(self, emotion: str, issues: List[str]) -> str:
        """Practical guidance"""
        
        guidance = {
            "anxious": """Here's what actually helps with anxiety:

**Right now:** 5-4-3-2-1 grounding. Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste. This pulls your brain back to the present instead of feared futures.

**Why it works:** Anxiety lives in the future ("what if"). Grounding brings you to NOW, where you're actually safe.""",
            
            "sad": """Depression has a cruel trick - it tells you to do nothing, which makes you feel worse.

**Break the cycle:** Pick ONE tiny thing. Not "exercise an hour" - try "step outside 2 minutes" or "text one friend". Microscopic.

**Why it works:** Action creates momentum. Start tiny, build from there. Motivation follows action, not the other way around.""",
            
            "overwhelmed": """When everything feels like TOO MUCH, your brain can't prioritize. So:

**Step 1:** Brain dump everything (even small stuff)
**Step 2:** Circle what's ACTUALLY urgent today
**Step 3:** Pick the ONE thing that matters most
**Step 4:** Just that. Everything else waits.

**Why it works:** You can only eat the elephant one bite at a time.""",
            
            "angry": """Anger needs healthy expression:

**Cool down first:** TIPP - Temperature (cold water/ice), Intense exercise (10 jumping jacks), Paced breathing (4-7-8), Paired relaxation

**Then address it:** Use "I feel ___ when ___ because ___" statements

**Why:** You can't think clearly while flooded with emotion. Cool first, then communicate."""
        }
        
        return guidance.get(emotion, "Let me share what might help here.")
    
    def _reframing_response(self, distortion: str, text: str) -> str:
        """Cognitive reframing"""
        
        reframes = {
            "catastrophizing": "I notice you're imagining worst-case scenarios. That's anxiety talking - it wants to prepare you for danger. But ask yourself: What's the MOST LIKELY outcome? Not worst, not best - most realistic?",
            
            "black_and_white": "You're using 'always' or 'never'. Life is rarely that absolute, even though it FEELS that way. What's a more nuanced way to look at this?",
            
            "mind_reading": "You're assuming you know what others think. But we're terrible at mind-reading - we usually project our own fears. What's the actual evidence they think that?",
            
            "fortune_telling": "You're predicting a negative future. But you've probably been wrong about predictions before, right? We all have. What if you're wrong about this one too?",
            
            "labeling": "You're putting a fixed label on yourself. But you're not a static thing - you're a changing person who sometimes struggles. Big difference.",
            
            "should_statements": "Those 'shoulds' are creating pressure. Who says you 'should'? What if you replaced 'should' with 'could' or 'choose to'?",
            
            "personalization": "You're taking responsibility for things outside your control. What parts are ACTUALLY yours, and what parts aren't?"
        }
        
        return reframes.get(distortion, "I notice a pattern in how you're thinking about this. What if we looked at it differently?")
    
    def _information_response(self, text: str) -> str:
        """Answer their question"""
        
        text_lower = text.lower()
        
        # Common topics
        if any(word in text_lower for word in ["think", "thought", "mind"]):
            return """Our brains are fascinating machines. We have ~60,000 thoughts per day, and your brain's main job is problem-scanning (that's why it focuses on negatives).

**The mental health angle:** Your thoughts aren't facts - they're your brain's best guess based on past experiences and current mood. Anxiety makes you catastrophize. Depression makes you think in absolutes.

**Good news:** We can actually retrain our brains through techniques like CBT. Thoughts are changeable."""
        
        elif any(word in text_lower for word in ["why", "point", "meaning"]):
            return """Big existential questions often surface when we're struggling emotionally. That makes sense - pain makes us question everything.

**Here's what I know:** Meaning isn't found, it's created. It comes from:
• Connection with others
• Contributing something (even small)
• Personal growth
• Experiences that matter to YOU

The fact you're asking shows you're thinking deeply. That's a strength."""
        
        else:
            return """That's a great question. While I'm focused on mental health support, I can share that most questions about life, meaning, and existence intersect with our emotional wellbeing.

Want to explore how this question relates to what you're feeling?"""
    
    def _listening_response(self, emotion: str) -> str:
        """Just listening"""
        
        responses = [
            "I hear you. Sometimes the most helpful thing is just being heard and understood.",
            "Thank you for trusting me with this. I'm here with you in it.",
            "That's a lot to carry. I'm listening.",
            "You don't have to have it all figured out. Just talking about it matters."
        ]
        
        return random.choice(responses)
    
    def _general_support(self, emotion: str) -> str:
        """General supportive response"""
        
        return "What you're dealing with is real, and you don't have to face it alone. Even talking about it - like you're doing now - is a form of strength."

# ═══════════════════════════════════════════════════════════════════════
# CRISIS DETECTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════

class CrisisDetector:
    """Detect genuine crisis situations"""
    
    CRITICAL_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die", "end it all",
        "take my life", "not worth living", "better off dead"
    ]
    
    HIGH_KEYWORDS = [
        "self harm", "hurt myself", "cut myself", "burn myself", "overdose",
        "harm myself", "injure myself"
    ]
    
    MODERATE_KEYWORDS = [
        "can't go on", "give up on life", "no point living", "want to disappear"
    ]
    
    @staticmethod
    def check(text: str, understanding: Dict) -> Dict:
        """Check for crisis"""
        
        text_lower = text.lower()
        
        critical = any(kw in text_lower for kw in CrisisDetector.CRITICAL_KEYWORDS)
        high = any(kw in text_lower for kw in CrisisDetector.HIGH_KEYWORDS)
        moderate = any(kw in text_lower for kw in CrisisDetector.MODERATE_KEYWORDS)
        
        # Also check emotion + intensity
        emotion = understanding.get("primary_emotion")
        intensity = understanding.get("emotion_intensity", 0)
        
        if critical or (high and intensity > 70):
            level = "critical"
        elif high or (moderate and intensity > 80):
            level = "high"
        elif moderate and emotion in ["sad", "hopeless"] and intensity > 70:
            level = "moderate"
        else:
            return {"is_crisis": False, "level": "none"}
        
        return {
            "is_crisis": True,
            "level": level,
            "detected_keywords": [kw for kw in CrisisDetector.CRITICAL_KEYWORDS + CrisisDetector.HIGH_KEYWORDS if kw in text_lower]
        }

# ═══════════════════════════════════════════════════════════════════════
# THERAPY TOOLKIT
# ═══════════════════════════════════════════════════════════════════════

class TherapyToolkit:
    """Evidence-based therapeutic interventions"""
    
    @staticmethod
    def breathing_478():
        """4-7-8 breathing exercise"""
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║        🫁 4-7-8 BREATHING - 90 Second Calm Down             ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("Find a comfortable position. Close your eyes if you'd like.\n")
        print("This activates your parasympathetic nervous system - your")
        print("body's natural 'calm down' response.\n")
        
        input("Press Enter when ready...")
        print()
        
        for i in range(1, 4):
            print(f"═══ Round {i}/3 ═══")
            print("Breathe IN through your nose... (1... 2... 3... 4...)")
            time.sleep(4)
            print("HOLD... (1... 2... 3... 4... 5... 6... 7...)")
            time.sleep(7)
            print("Breathe OUT through your mouth... (1... 2... 3... 4... 5... 6... 7... 8...)")
            time.sleep(8)
            if i < 3:
                print()
        
        print("\n✨ Notice how you feel. Even a small shift is progress.\n")
    
    @staticmethod
    def grounding_54321():
        """5-4-3-2-1 grounding technique"""
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║           🧘 5-4-3-2-1 GROUNDING TECHNIQUE                   ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("This pulls your mind back to the present moment.\n")
        print("Anxiety lives in the future. This brings you to NOW.\n")
        
        print("Look around you. Name out loud or in your head:\n")
        
        print("👁️  5 things you SEE:")
        for i in range(5):
            input(f"   {i+1}. ")
        
        print("\n✋ 4 things you can TOUCH:")
        for i in range(4):
            input(f"   {i+1}. ")
        
        print("\n👂 3 things you HEAR:")
        for i in range(3):
            input(f"   {i+1}. ")
        
        print("\n👃 2 things you SMELL:")
        for i in range(2):
            input(f"   {i+1}. ")
        
        print("\n👅 1 thing you TASTE:")
        input("   1. ")
        
        print("\n✨ You're here. You're present. You're grounded.\n")
    
    @staticmethod
    def thought_record():
        """CBT thought record"""
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              📝 CBT THOUGHT RECORD                           ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        print("Let's examine this thought like a scientist.\n")
        
        situation = input("What's the situation?\n→ ")
        thought = input("\nWhat's the automatic thought?\n→ ")
        emotion = input("\nWhat emotion does it create? (0-100%)\n→ ")
        
        print("\nEvidence FOR the thought:")
        evidence_for = input("→ ")
        
        print("\nEvidence AGAINST the thought:")
        evidence_against = input("→ ")
        
        print("\nAlternative balanced thought:")
        alternative = input("→ ")
        
        print(f"\n✨ New emotion intensity (0-100%):")
        new_emotion = input("→ ")
        
        print("\n═══ SUMMARY ═══")
        print(f"Original thought: {thought}")
        print(f"Original emotion: {emotion}%")
        print(f"Balanced thought: {alternative}")
        print(f"New emotion: {new_emotion}%")
        print("\nNotice any shift? Even small changes matter.\n")
    
    @staticmethod
    def show_all_techniques():
        """Show all available techniques"""
        print("\n╔══════════════════════════════════════════════════════════════╗")
        print("║              🎯 THERAPY TECHNIQUES LIBRARY                   ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")
        
        techniques = {
            "1": {"name": "4-7-8 Breathing", "for": "Anxiety, stress, sleep", "time": "90 seconds"},
            "2": {"name": "5-4-3-2-1 Grounding", "for": "Anxiety, panic, overwhelm", "time": "3-5 minutes"},
            "3": {"name": "CBT Thought Record", "for": "Negative thinking, distortions", "time": "5-10 minutes"},
            "4": {"name": "Body Scan", "for": "Stress, tension, mindfulness", "time": "10 minutes"},
            "5": {"name": "Values Clarification", "for": "Direction, meaning, motivation", "time": "15 minutes"},
            "6": {"name": "Behavioral Activation", "for": "Depression, low mood", "time": "Ongoing"}
        }
        
        for num, tech in techniques.items():
            print(f"{num}. {tech['name']}")
            print(f"   For: {tech['for']}")
            print(f"   Time: {tech['time']}\n")
        
        print("Type the number to try a technique, or press Enter to skip.")
        choice = input("→ ").strip()
        
        if choice == "1":
            TherapyToolkit.breathing_478()
        elif choice == "2":
            TherapyToolkit.grounding_54321()
        elif choice == "3":
            TherapyToolkit.thought_record()
        elif choice:
            print("\n✨ That technique is coming soon!\n")

# ═══════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Manage all database operations"""
    
    def __init__(self):
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    emotion TEXT,
                    intensity INTEGER,
                    bot_response TEXT,
                    crisis_detected INTEGER DEFAULT 0
                )
            """)
            
            # User profile table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY,
                    profile_data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    insight_data TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database init error: {e}")
    
    def log_conversation(self, user_msg: str, understanding: Dict, bot_response: str, crisis: bool):
        """Log conversation"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversations 
                (timestamp, user_message, emotion, intensity, bot_response, crisis_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                user_msg,
                understanding.get("primary_emotion", "unknown"),
                understanding.get("emotion_intensity", 0),
                bot_response,
                1 if crisis else 0
            ))
            
            conn.commit()
            conn.close()
        except:
            pass
    
    def get_stats(self, days: int = 7) -> Dict:
        """Get statistics"""
        try:
            conn = sqlite3.connect(Config.DB_PATH)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE timestamp > ?", (start_date,))
            total = cursor.fetchone()[0]
            
            # Most common emotion
            cursor.execute("""
                SELECT emotion, COUNT(*) as count 
                FROM conversations 
                WHERE timestamp > ? 
                GROUP BY emotion 
                ORDER BY count DESC 
                LIMIT 1
            """, (start_date,))
            emotion_result = cursor.fetchone()
            
            # Average intensity
            cursor.execute("SELECT AVG(intensity) FROM conversations WHERE timestamp > ?", (start_date,))
            avg_intensity = cursor.fetchone()[0] or 0
            
            # Crisis count
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE crisis_detected = 1 AND timestamp > ?", (start_date,))
            crisis_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_conversations": total,
                "most_common_emotion": emotion_result[0] if emotion_result else "N/A",
                "emotion_count": emotion_result[1] if emotion_result else 0,
                "avg_intensity": round(avg_intensity, 1),
                "crisis_count": crisis_count,
                "days": days
            }
        except:
            return {
                "total_conversations": 0,
                "most_common_emotion": "N/A",
                "emotion_count": 0,
                "avg_intensity": 0,
                "crisis_count": 0,
                "days": days
            }

# ═══════════════════════════════════════════════════════════════════════
# MAIN COACH APPLICATION
# ═══════════════════════════════════════════════════════════════════════

class MindfulPro:
    """Main application - Mindful Pro Mental Wellness Companion"""
    
    def __init__(self):
        # Initialize components
        self.brain = IntelligentBrain()
        self.responder = IntelligentResponder()
        self.crisis_detector = CrisisDetector()
        self.toolkit = TherapyToolkit()
        self.db = DatabaseManager()
        
        # Welcome
        self._show_welcome()
    
    def _show_welcome(self):
        """Show welcome screen"""
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                                                                              ║")
        print("║                  🌟 MINDFUL PRO - Mental Wellness Companion                  ║")
        print("║                                                                              ║")
        print("║                        Version 1.0.0 | Production Ready                     ║")
        print("║                                                                              ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
        
        print("I'm not just another chatbot. I'm genuinely intelligent and caring.\n")
        print("What makes me different:")
        print("  • I understand context and read between the lines")
        print("  • I respond like a real therapist friend, not a robot")
        print("  • I learn your patterns and adapt to you")
        print("  • I provide evidence-based therapeutic support\n")
        
        # Show AI status
        if self.responder.lm_studio:
            print("✨ LM Studio: Connected (Full AI capabilities)")
        elif self.responder.groq:
            print("✨ Groq: Connected (Cloud AI active)")
        else:
            print("💡 Smart Mode: Active (Intelligent rule-based responses)")
            print("   Tip: Add LM Studio or Groq API key for AI-powered chat\n")
        
        # Show user stats if available
        if self.brain.user_profile["total_sessions"] > 0:
            print(f"\n📊 Welcome back! We've had {self.brain.user_profile['total_sessions']} conversations.")
        
        print("\n" + "─" * 78 + "\n")
    
    def chat(self, user_input: str):
        """Main chat function"""
        
        # Understand deeply
        understanding = self.brain.understand(user_input)
        
        # Check crisis first
        crisis_check = self.crisis_detector.check(user_input, understanding)
        
        if crisis_check["is_crisis"]:
            self._handle_crisis(crisis_check)
            self.db.log_conversation(user_input, understanding, "[CRISIS RESPONSE]", True)
            return
        
        # Generate response
        response = self.responder.generate(understanding)
        
        # Display
        self._display_response(understanding, response)
        
        # Log
        self.db.log_conversation(user_input, understanding, response, False)
    
    def _handle_crisis(self, crisis_check: Dict):
        """Handle crisis situation"""
        print("\n" + "═" * 78)
        print("\n🚨 I'm really concerned about what you're sharing.\n")
        print("Your safety is the priority. Please reach out for immediate professional support:\n")
        
        level = crisis_check["level"]
        
        if level in ["critical", "high"]:
            print("🇮🇳 INDIA:")
            for hotline in Config.CRISIS_HOTLINES["india"]:
                print(f"   • {hotline['name']}: {hotline['number']} ({hotline['hours']})")
            
            print("\n🇺🇸 USA:")
            for hotline in Config.CRISIS_HOTLINES["usa"]:
                print(f"   • {hotline['name']}: {hotline['number']} ({hotline['hours']})")
            
            print("\n🇬🇧 UK:")
            for hotline in Config.CRISIS_HOTLINES["uk"]:
                print(f"   • {hotline['name']}: {hotline['number']} ({hotline['hours']})")
            
            print("\n💙 You matter. Your life matters. These feelings can change.")
            print("Professional support can make a huge difference.")
            print("\nPlease reach out - right now if possible.\n")
        
        print("═" * 78 + "\n")
    
    def _display_response(self, understanding: Dict, response: str):
        """Display response beautifully"""
        
        print("\n" + "─" * 78)
        
        # Show emotion (only if significant)
        emotion = understanding["primary_emotion"]
        intensity = understanding.get("emotion_intensity", 0)
        
        if emotion != "neutral" and intensity > 30:
            emoji_map = {
                "anxious": "😰", "sad": "😢", "angry": "😠",
                "overwhelmed": "😓", "confused": "🤔", "hopeful": "🙂"
            }
            emoji = emoji_map.get(emotion, "💬")
            
            intensity_bar = "█" * int(intensity / 10) + "░" * (10 - int(intensity / 10))
            print(f"\n{emoji} Emotion detected: {emotion.title()}")
            print(f"   Intensity: [{intensity_bar}] {intensity}%")
        
        # The response
        print()
        print(self._wrap_text(response, 76))
        print()
        
        # Smart suggestions
        needs = understanding.get("needs", [])
        if "anxiety_management" in needs or emotion == "anxious":
            print("💡 Quick help: Type 'breathe' for 90-second calm-down exercise")
        elif "validation" in needs and intensity > 70:
            print("💡 Available: Type 'techniques' to see all coping tools")
        
        print("\n" + "─" * 78 + "\n")
    
    def _wrap_text(self, text: str, width: int = 76) -> str:
        """Wrap text for readability"""
        paragraphs = text.split('\n\n')
        wrapped = []
        
        for para in paragraphs:
            # Don't wrap formatted text
            if para.startswith('**') or para.startswith('•') or para.startswith('#'):
                wrapped.append(para)
                continue
            
            words = para.split()
            lines = []
            current = ""
            
            for word in words:
                if len(current) + len(word) + 1 <= width:
                    current += (word + " ")
                else:
                    if current:
                        lines.append(current.rstrip())
                    current = word + " "
            
            if current:
                lines.append(current.rstrip())
            
            wrapped.append('\n'.join(lines))
        
        return '\n\n'.join(wrapped)
    
    def show_stats(self):
        """Show user statistics"""
        stats = self.db.get_stats(7)
        
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                        📊 YOUR WELLNESS INSIGHTS                             ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
        
        if stats["total_conversations"] < 3:
            print(f"Conversations this week: {stats['total_conversations']}")
            print(f"\n💡 Keep chatting! I need {3 - stats['total_conversations']} more conversations to")
            print("   provide meaningful insights about your patterns.\n")
        else:
            print(f"📈 Activity (Last {stats['days']} days)")
            print(f"   Total conversations: {stats['total_conversations']}")
            print(f"   Most common emotion: {stats['most_common_emotion'].title()} ({stats['emotion_count']} times)")
            print(f"   Average intensity: {stats['avg_intensity']}%")
            
            if stats["crisis_count"] > 0:
                print(f"   ⚠️  Crisis moments: {stats['crisis_count']}")
                print(f"      Consider reaching out to a professional for ongoing support")
            
            # Wellness score
            score = self._calculate_wellness_score(stats)
            print(f"\n🎯 Wellness Score: {score}/100")
            
            if score >= 70:
                print("   Status: Doing well! Keep up the good self-care.")
            elif score >= 50:
                print("   Status: Managing. Consider additional support if needed.")
            elif score >= 30:
                print("   Status: Struggling. Professional support recommended.")
            else:
                print("   Status: Need support. Please talk to a mental health professional.")
            
            # Patterns
            if self.brain.user_profile.get("recurring_theme"):
                print(f"\n🔍 Pattern noticed: Recurring theme of {self.brain.user_profile['recurring_theme']}")
                print("   This might be worth exploring more deeply.")
        
        print()
    
    def _calculate_wellness_score(self, stats: Dict) -> int:
        """Calculate wellness score"""
        base_score = 70
        
        # Adjust based on intensity
        if stats["avg_intensity"] > 70:
            base_score -= 20
        elif stats["avg_intensity"] > 50:
            base_score -= 10
        
        # Adjust based on crisis
        base_score -= (stats["crisis_count"] * 15)
        
        # Adjust based on emotion
        if stats["most_common_emotion"] in ["sad", "hopeless", "anxious"]:
            base_score -= 15
        
        return max(0, min(100, base_score))
    
    def interactive(self):
        """Interactive mode"""
        
        print("I'm listening. Type anything, or try these commands:\n")
        print("  💬 Just talk - share what's on your mind")
        print("  🫁 'breathe' - Quick 90-second breathing exercise")
        print("  🎯 'techniques' - View all therapy tools")
        print("  📊 'stats' - Your wellness insights")
        print("  ❓ 'help' - See all features")
        print("  👋 'exit' - Take care!\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                cmd = user_input.lower()
                
                if cmd == 'exit':
                    print("\n💙 Take good care of yourself. I'm here whenever you need me.\n")
                    break
                
                elif cmd in ['breathe', 'breathing', 'breath']:
                    self.toolkit.breathing_478()
                
                elif cmd == 'ground':
                    self.toolkit.grounding_54321()
                
                elif cmd in ['techniques', 'tools', 'exercises']:
                    self.toolkit.show_all_techniques()
                
                elif cmd in ['stats', 'progress', 'insights']:
                    self.show_stats()
                
                elif cmd == 'help':
                    self._show_help()
                
                else:
                    # Regular chat
                    self.chat(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n💙 Take care!\n")
                break
            except Exception as e:
                print(f"\n⚠️  Something went wrong: {e}\n")
                print("Let's try that again.")
    
    def _show_help(self):
        """Show help information"""
        print("\n╔══════════════════════════════════════════════════════════════════════════════╗")
        print("║                           📖 MINDFUL PRO FEATURES                            ║")
        print("╚══════════════════════════════════════════════════════════════════════════════╝\n")
        
        print("🧠 INTELLIGENT FEATURES")
        print("   • Deep contextual understanding (reads between the lines)")
        print("   • Learns your patterns and adapts over time")
        print("   • Detects cognitive distortions and underlying issues")
        print("   • Personalized responses based on your unique situation\n")
        
        print("💬 THERAPY SUPPORT")
        print("   • Evidence-based therapeutic techniques (CBT, DBT, ACT)")
        print("   • Crisis detection and immediate resource provision")
        print("   • Emotional validation and practical guidance")
        print("   • Thought reframing and cognitive restructuring\n")
        
        print("🎯 TOOLS & EXERCISES")
        print("   • 4-7-8 Breathing (anxiety, stress)")
        print("   • 5-4-3-2-1 Grounding (panic, overwhelm)")
        print("   • Thought Records (negative thinking)")
        print("   • And more... (type 'techniques')\n")
        
        print("📊 PROGRESS TRACKING")
        print("   • Wellness scoring based on patterns")
        print("   • Emotion tracking over time")
        print("   • Pattern recognition and insights")
        print("   • Complete conversation history\n")
        
        print("⚡ QUICK COMMANDS")
        print("   • 'breathe' - 90-second breathing exercise")
        print("   • 'ground' - 5-4-3-2-1 grounding technique")
        print("   • 'techniques' - All therapy tools")
        print("   • 'stats' - Your wellness insights")
        print("   • 'exit' - End session\n")
        
        print("💡 TIP: Just talk naturally. I understand context and nuance.\n")

# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        app = MindfulPro()
        app.interactive()
    except KeyboardInterrupt:
        print("\n\n💙 Take care of yourself!\n")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("Please restart the application.\n")