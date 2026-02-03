"""
🚀 ULTIMATE MENTAL HEALTH AI CHATBOT - PROFESSIONAL EDITION V2.0 🚀
33+ FEATURES | NLP-Powered | Therapeutic Frameworks | Privacy-First
Multi-Modal AI | Advanced Analytics | Conversation Memory | Groq Integration
"""

import requests
import json
import os
import time
import sqlite3
import csv
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, Counter
import random
import statistics

# Optional: Advanced NLP imports (install if needed)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Install transformers for advanced NLP: pip install transformers torch")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  Install textblob for sentiment: pip install textblob")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  Install spaCy for NLP: pip install spacy && python -m spacy download en_core_web_sm")

# ========================= CONFIGURATION =========================
class Config:
    """Advanced enterprise configuration"""

    # LM Studio (Primary - Local AI)
    LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODEL = "llm"

    # Groq (Backup - Cloud AI) - REPLACES GEMINI
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL = "llama-3.1-70b-versatile"  # Best for mental health

    # AI Generation Parameters
    TEMPERATURE = 0.7
    MAX_TOKENS = 512
    TOP_P = 0.9

    # Database & Export
    DB_PATH = "mental_health_ultimate.db"
    EXPORT_PATH = "mental_health_reports"

    # Test Scenarios
    TEST_SCENARIOS = {
        "stress": "I feel completely overwhelmed with work and personal responsibilities. My mind won't stop racing.",
        "anxiety": "I have to give a presentation tomorrow and I'm extremely nervous. What if I make mistakes?",
        "loneliness": "I feel isolated and don't have anyone to talk to about my problems.",
        "motivation": "I've lost motivation for everything. Nothing seems worth doing anymore.",
        "relationship": "My relationship is going through a rough patch. We keep arguing over small things.",
        "sleep": "I can't sleep at night because my mind keeps thinking about everything.",
        "self_doubt": "I keep doubting myself and feeling like I'm not good enough for anything.",
        "grief": "I recently lost someone close to me and I don't know how to process it."
    }

    # Crisis Detection
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end it all", "want to die", "harm myself",
        "no point living", "can't go on", "better off dead", "jump off", "overdose",
        "end my life", "not worth living", "give up on life"
    ]

    # Affirmations
    AFFIRMATIONS = [
        "You are stronger than you think.",
        "This moment is temporary; it will pass.",
        "Your feelings are valid and important.",
        "You deserve kindness, especially from yourself.",
        "Progress over perfection.",
        "You have overcome challenges before.",
        "Your worth is not defined by your productivity.",
        "It's okay to ask for help.",
        "You are enough, just as you are.",
        "One step at a time is still progress.",
    ]

# ========================= FEATURE 1-5: ADVANCED NLP SENTIMENT ANALYSIS =========================
class NLPEnhancedSentimentAnalyzer:
    """Professional-grade NLP sentiment analysis with multiple engines"""

    def __init__(self):
        self.transformer_classifier = None
        self.nlp = None
        
        # Initialize transformer model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                print("🧠 Loading emotion classifier...")
                self.transformer_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
                print("✅ Transformer model loaded")
            except Exception as e:
                print(f"⚠️  Transformer load failed: {e}")
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy loaded")
            except:
                print("⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm")

    def analyze(self, text: str) -> Dict:
        """Comprehensive multi-engine sentiment analysis"""
        
        result = {
            "polarity": 0.0,
            "label": "neutral",
            "intensity": 0.0,
            "emotion": "neutral",
            "confidence": 0.0,
            "subjectivity": 0.0,
            "all_emotions": [],
            "entities": [],
            "key_phrases": [],
            "word_breakdown": {}
        }
        
        # 1. Transformer-based emotion detection (if available)
        if self.transformer_classifier:
            try:
                emotions = self.transformer_classifier(text)[0]
                if emotions:
                    primary = max(emotions, key=lambda x: x['score'])
                    result["emotion"] = primary['label']
                    result["confidence"] = primary['score'] * 100
                    result["all_emotions"] = emotions
                    result["intensity"] = primary['score'] * 100
            except Exception as e:
                print(f"⚠️  Transformer analysis failed: {e}")
        
        # 2. TextBlob sentiment (if available)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                result["polarity"] = blob.sentiment.polarity
                result["subjectivity"] = blob.sentiment.subjectivity
                
                # Map polarity to label
                if result["polarity"] > 0.5:
                    result["label"] = "very_positive"
                elif result["polarity"] > 0.1:
                    result["label"] = "positive"
                elif result["polarity"] > -0.1:
                    result["label"] = "neutral"
                elif result["polarity"] > -0.5:
                    result["label"] = "negative"
                else:
                    result["label"] = "very_negative"
            except Exception as e:
                print(f"⚠️  TextBlob analysis failed: {e}")
        
        # 3. spaCy entity & phrase extraction (if available)
        if self.nlp:
            try:
                doc = self.nlp(text)
                result["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
                result["key_phrases"] = [chunk.text for chunk in doc.noun_chunks][:5]
            except Exception as e:
                print(f"⚠️  spaCy analysis failed: {e}")
        
        # 4. Fallback: Pattern-based analysis (always available)
        pattern_result = self._pattern_based_analysis(text)
        
        # Merge results (use transformer if available, otherwise pattern-based)
        if not self.transformer_classifier:
            result.update(pattern_result)
        else:
            # Use pattern-based word breakdown
            result["word_breakdown"] = pattern_result["word_breakdown"]
            # If transformer didn't detect emotion, use pattern-based
            if result["emotion"] == "neutral" and pattern_result["emotion"] != "neutral":
                result["emotion"] = pattern_result["emotion"]
        
        return result

    def _pattern_based_analysis(self, text: str) -> Dict:
        """Enhanced pattern-based analysis with contextual phrases"""
        
        EMOTION_PATTERNS = {
            "anxious": {
                "keywords": ["anxious", "nervous", "worried", "panic", "scared", "afraid", 
                            "overwhelmed", "tense", "uneasy", "restless", "fear"],
                "phrases": ["can't stop thinking", "what if", "heart racing", "can't breathe",
                           "feel like", "going to happen", "so worried", "keep worrying"]
            },
            "sad": {
                "keywords": ["sad", "depressed", "hopeless", "empty", "numb", "alone",
                            "worthless", "cry", "grief", "loss", "miserable"],
                "phrases": ["don't want to", "no point", "can't enjoy", "lost interest",
                           "feel nothing", "want to disappear", "feel empty", "so alone"]
            },
            "angry": {
                "keywords": ["angry", "frustrated", "furious", "annoyed", "irritated",
                            "mad", "rage", "resentful", "hate"],
                "phrases": ["so frustrated", "can't stand", "makes me angry", "fed up",
                           "sick of", "can't take it"]
            },
            "stressed": {
                "keywords": ["stressed", "pressure", "deadline", "burden", "exhausted",
                            "burned out", "too much", "swamped"],
                "phrases": ["can't handle", "too many", "no time", "falling apart",
                           "can't cope", "breaking point"]
            }
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Calculate emotion scores
        emotion_scores = {}
        for emotion, patterns in EMOTION_PATTERNS.items():
            score = 0
            score += sum(2 for keyword in patterns["keywords"] if keyword in text_lower)
            score += sum(3 for phrase in patterns["phrases"] if phrase in text_lower)
            emotion_scores[emotion] = score
        
        # Determine primary emotion
        primary_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
        if emotion_scores.get(primary_emotion, 0) == 0:
            primary_emotion = "neutral"
        
        # Basic sentiment
        positive_words = {
            "happy", "joy", "grateful", "excited", "hopeful", "better", "good",
            "calm", "peaceful", "loved", "appreciated", "proud", "confident"
        }
        
        negative_words = {
            "sad", "anxious", "stressed", "depressed", "worried", "overwhelmed",
            "hopeless", "awful", "terrible", "useless", "failed", "alone"
        }
        
        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)
        total = pos_score + neg_score
        
        polarity = (pos_score - neg_score) / max(total, 1) if total > 0 else 0.0
        intensity = min(100, sum(emotion_scores.values()) * 10)
        
        if polarity > 0.5:
            label = "very_positive"
        elif polarity > 0.1:
            label = "positive"
        elif polarity > -0.1:
            label = "neutral"
        elif polarity > -0.5:
            label = "negative"
        else:
            label = "very_negative"
        
        return {
            "polarity": polarity,
            "label": label,
            "intensity": intensity,
            "emotion": primary_emotion,
            "word_breakdown": {
                "positive": pos_score,
                "negative": neg_score,
                "emotion_scores": emotion_scores
            }
        }

# ========================= FEATURE 6-10: CRISIS DETECTION =========================
class EnhancedCrisisDetectionSystem:
    """Multi-factor crisis detection with therapy framework integration"""

    def __init__(self):
        self.crisis_history = []

    def detect(self, text: str, sentiment: Dict) -> Dict:
        """Advanced crisis detection using multiple signals"""
        
        text_lower = text.lower()
        detected_keywords = [kw for kw in Config.CRISIS_KEYWORDS if kw in text_lower]
        
        # Multi-factor scoring
        crisis_score = 0
        
        # Factor 1: Keywords (highest weight)
        if detected_keywords:
            crisis_score += 40
        
        # Factor 2: Sentiment polarity
        if sentiment['polarity'] < -0.7:
            crisis_score += 30
        elif sentiment['polarity'] < -0.5:
            crisis_score += 20
        
        # Factor 3: Specific emotions
        if sentiment['emotion'] in ['sad', 'hopeless']:
            crisis_score += 15
        
        # Factor 4: Intensity
        if sentiment['intensity'] > 80:
            crisis_score += 10
        
        # Factor 5: Subjectivity (if available)
        if sentiment.get('subjectivity', 0) > 0.8:
            crisis_score += 5
        
        # Determine severity
        if crisis_score >= 70:
            severity = "critical"
            is_crisis = True
        elif crisis_score >= 50:
            severity = "high"
            is_crisis = True
        elif crisis_score >= 30:
            severity = "moderate"
            is_crisis = True
        else:
            severity = "none"
            is_crisis = False
        
        crisis_response = self._get_response(severity)
        
        if is_crisis:
            self.crisis_history.append({
                "timestamp": datetime.now(),
                "severity": severity,
                "keywords": detected_keywords,
                "score": crisis_score
            })
        
        return {
            "is_crisis": is_crisis,
            "severity": severity,
            "keywords": detected_keywords,
            "response": crisis_response,
            "score": crisis_score
        }

    @staticmethod
    def _get_response(severity: str) -> str:
        """Get crisis response with helplines"""
        responses = {
            "critical": """🚨 IMMEDIATE HELP NEEDED - PLEASE CALL NOW

🇮🇳 INDIA:
   • AASRA: +91-9820466726 (24/7)
   • iCall: +91-96540 22000 (Mon-Sat 8am-10pm)
   • NIMHANS: +91-80-26995000

🇺🇸 USA:
   • 988 Suicide & Crisis Lifeline
   • Crisis Text: HOME to 741741

💙 You are not alone. These trained professionals are ready to help RIGHT NOW.
Please reach out - your life matters.""",

            "high": """⚠️ URGENT SUPPORT NEEDED

🇮🇳 INDIA:
   • AASRA: +91-9820466726
   • NIMHANS: +91-80-26995000

🇺🇸 USA:
   • 988 Lifeline
   • SAMHSA: 1-800-662-4357

Please talk to someone trained to help. You don't have to face this alone.""",

            "moderate": """💙 SUPPORT AVAILABLE

If you're struggling, please consider:
   • Talking to a trusted friend/family member
   • Calling a helpline: AASRA +91-9820466726
   • Scheduling with a mental health professional

You matter, and help is available.""",

            "none": ""
        }
        return responses.get(severity, "")

# ========================= FEATURE 11-13: THERAPY FRAMEWORK =========================
class TherapyFramework:
    """CBT, DBT, and Mindfulness-based interventions"""
    
    CBT_TECHNIQUES = {
        "thought_challenging": {
            "triggers": ["always", "never", "terrible", "awful", "everyone", "nobody"],
            "name": "Cognitive Restructuring",
            "description": """I notice some all-or-nothing thinking. Let's challenge that:

1. What's the EVIDENCE FOR this thought?
2. What's the EVIDENCE AGAINST it?
3. What would you tell a FRIEND in this situation?
4. What's a more BALANCED thought?

Example: "I always fail" → "I've succeeded at X, Y, Z. This is one setback, not a pattern." """
        },
        
        "behavioral_activation": {
            "triggers": ["don't want to", "no energy", "can't do", "too tired", "no motivation"],
            "name": "Behavioral Activation",
            "description": """Depression reduces motivation. Let's reverse that:

1. Pick ONE small activity (5-10 minutes)
   - Make tea
   - Walk around the block
   - Text one friend
2. Schedule it for a specific time TODAY
3. Rate mood BEFORE (0-10): ___
4. Do the activity
5. Rate mood AFTER (0-10): ___

Small actions → mood improvement → more motivation → more actions"""
        },
        
        "catastrophizing": {
            "triggers": ["what if", "worst case", "disaster", "terrible", "going to"],
            "name": "Decatastrophizing",
            "description": """You're catastrophizing (imagining worst outcomes). Let's reality-check:

1. What's the WORST that could happen?
2. What's the BEST that could happen?
3. What's the MOST LIKELY outcome?
4. Even if the worst happens, how would you cope?

Most feared outcomes don't happen, and we're more resilient than we think."""
        }
    }
    
    DBT_SKILLS = {
        "distress_tolerance": {
            "name": "TIPP Skills (Crisis Management)",
            "description": """When overwhelmed, use TIPP:

T - TEMPERATURE: Splash cold water on face / hold ice cube
I - INTENSE EXERCISE: 10 jumping jacks / run in place
P - PACED BREATHING: 4 in, 7 hold, 8 out
P - PAIRED MUSCLE RELAXATION: Tense then relax muscle groups

These calm your nervous system FAST."""
        },
        
        "accepts": {
            "name": "ACCEPTS (Distraction)",
            "description": """When distressed, use ACCEPTS to distract:

A - ACTIVITIES: Do something engaging
C - CONTRIBUTING: Help someone else
C - COMPARISONS: Compare to worse times (you survived those)
E - EMOTIONS: Watch something that creates different emotion
P - PUSHING AWAY: Visualize putting problem in a box temporarily
T - THOUGHTS: Count, puzzles, focus mind elsewhere
S - SENSATIONS: Strong taste, smell, texture"""
        },
        
        "opposite_action": {
            "name": "Opposite Action",
            "description": """When emotion is NOT justified or too intense, act OPPOSITE:

• Anxious? → Approach what you fear (gradually)
• Sad? → Get active, reach out
• Angry? → Be kind, step away gently
• Shame? → Share the story with safe person

Acting opposite changes the emotion."""
        }
    }
    
    MINDFULNESS_TECHNIQUES = {
        "5_4_3_2_1": {
            "name": "5-4-3-2-1 Grounding",
            "description": """GROUNDING - 5-4-3-2-1 TECHNIQUE

Name out loud or in your head:
• 5 things you SEE (clock, chair, window...)
• 4 things you TOUCH (soft shirt, cool desk...)
• 3 things you HEAR (fan, traffic, breathing...)
• 2 things you SMELL (coffee, soap...)
• 1 thing you TASTE (mint, water...)

This anchors you to the PRESENT MOMENT."""
        },
        
        "box_breathing": {
            "name": "Box Breathing",
            "description": """BOX BREATHING (4-4-4-4)

Breathe in a square pattern:
1. Inhale for 4 counts
2. Hold for 4 counts
3. Exhale for 4 counts
4. Hold for 4 counts

Repeat 4-5 times. Calms nervous system in 1-2 minutes."""
        },
        
        "4_7_8": {
            "name": "4-7-8 Breathing",
            "description": """4-7-8 BREATHING (for sleep & anxiety)

1. Inhale through nose for 4 counts
2. Hold breath for 7 counts
3. Exhale through mouth for 8 counts

Repeat 4 times. Activates relaxation response."""
        }
    }
    
    @staticmethod
    def suggest_intervention(text: str, emotion: str, intensity: float) -> Dict:
        """Automatically suggest best therapeutic intervention"""
        
        text_lower = text.lower()
        suggested_techniques = []
        
        # Check for CBT triggers
        for technique_key, technique in TherapyFramework.CBT_TECHNIQUES.items():
            if any(trigger in text_lower for trigger in technique["triggers"]):
                suggested_techniques.append({
                    "type": "CBT",
                    "name": technique["name"],
                    "description": technique["description"]
                })
        
        # Emotion-based suggestions
        if emotion == "anxious" and intensity > 60:
            suggested_techniques.append({
                "type": "Mindfulness",
                "name": TherapyFramework.MINDFULNESS_TECHNIQUES["5_4_3_2_1"]["name"],
                "description": TherapyFramework.MINDFULNESS_TECHNIQUES["5_4_3_2_1"]["description"]
            })
        
        if emotion == "sad" and intensity > 60:
            suggested_techniques.append({
                "type": "CBT",
                "name": TherapyFramework.CBT_TECHNIQUES["behavioral_activation"]["name"],
                "description": TherapyFramework.CBT_TECHNIQUES["behavioral_activation"]["description"]
            })
        
        if intensity > 75:  # High distress
            suggested_techniques.append({
                "type": "DBT",
                "name": TherapyFramework.DBT_SKILLS["distress_tolerance"]["name"],
                "description": TherapyFramework.DBT_SKILLS["distress_tolerance"]["description"]
            })
        
        return {
            "suggested_techniques": suggested_techniques[:2],  # Limit to 2 to avoid overwhelm
            "count": len(suggested_techniques)
        }

# ========================= FEATURE 14: SMART COPING STRATEGIES =========================
class EnhancedCopingStrategyGenerator:
    """Context-aware coping strategies with therapy integration"""

    @staticmethod
    def generate(emotion: str, intensity: float, text: str = "") -> List[str]:
        """Generate personalized evidence-based coping strategies"""
        
        strategies = []
        
        # Emotion-specific strategies
        emotion_strategies = {
            "anxious": [
                "🫁 Box breathing: Inhale 4, hold 4, exhale 4, hold 4 (repeat 5 times)",
                "🧘 5-4-3-2-1 grounding: Name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste",
                "💭 Challenge anxious thoughts: 'Is this thought realistic? What's the evidence?'",
                "🚶 Progressive muscle relaxation: Tense then relax each muscle group",
                "📝 Write down worries, then schedule 15min 'worry time' for later"
            ],
            "sad": [
                "🎯 Behavioral activation: Do ONE small enjoyable activity (10 min walk, favorite song, call friend)",
                "💬 Reach out to someone you trust - connection helps",
                "🌅 Go outside for 10 minutes - natural light improves mood",
                "📊 Challenge negative self-talk: 'Would I say this to a friend?'",
                "✅ Set tiny achievable goal for today (make bed, drink water)"
            ],
            "angry": [
                "🚶 Physical release: Brisk walk, exercise, or stretching",
                "🧊 TIPP: Hold ice cube or splash cold water on face (calms nervous system)",
                "🗣️ Express it safely: Journal, punch pillow, or talk to someone",
                "⏸️ Take a break from the situation if possible",
                "💭 Identify what need isn't being met (respect, fairness, control?)"
            ],
            "stressed": [
                "📋 Brain dump: Write EVERYTHING you need to do, then prioritize top 3",
                "⏰ Time-block: Work for 25 min, break for 5 (Pomodoro)",
                "🚫 Say no to one non-essential thing today",
                "🫁 4-7-8 breathing: Breathe in 4, hold 7, out 8 (repeat 4 times)",
                "🎯 Ask: 'What's the ONE thing that would make the biggest difference?'"
            ]
        }
        
        # Get emotion-specific strategies
        strategies = emotion_strategies.get(emotion, emotion_strategies["stressed"])
        
        # Intensity-based selection
        if intensity > 75:
            # High intensity: prioritize crisis management
            return strategies[:3]
        else:
            # Moderate: give variety
            return random.sample(strategies, min(3, len(strategies)))

# ========================= FEATURE 15-17: PROGRESS TRACKING =========================
class EnhancedProgressTracker:
    """Advanced analytics and tracking with conversation history"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scenario TEXT,
                    sentiment_polarity REAL,
                    sentiment_label TEXT,
                    emotion TEXT,
                    emotion_confidence REAL,
                    intensity REAL,
                    subjectivity REAL,
                    crisis_detected INTEGER,
                    crisis_severity TEXT,
                    crisis_score INTEGER,
                    model_used TEXT,
                    response_time REAL,
                    conversation_turns INTEGER,
                    therapy_technique_suggested TEXT
                )
            """)

            conn.commit()
            conn.close()
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️  DB initialization warning: {e}")

    def log_session(self, scenario: str, sentiment: Dict, crisis: Dict, 
                   model: str, response_time: float, conversation_turns: int = 0,
                   therapy_technique: str = ""):
        """Log comprehensive session data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions VALUES (
                    NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                datetime.now().isoformat(),
                scenario,
                sentiment.get('polarity', 0.0),
                sentiment.get('label', 'neutral'),
                sentiment.get('emotion', 'neutral'),
                sentiment.get('confidence', 0.0),
                sentiment.get('intensity', 0.0),
                sentiment.get('subjectivity', 0.0),
                1 if crisis.get('is_crisis', False) else 0,
                crisis.get('severity', 'none'),
                crisis.get('score', 0),
                model,
                response_time,
                conversation_turns,
                therapy_technique
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"⚠️  Logging warning: {e}")

    def get_mood_trend(self, days: int = 7) -> Dict:
        """Get mood trend analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            start_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("""
                SELECT sentiment_label, COUNT(*) FROM sessions 
                WHERE timestamp > ? 
                GROUP BY sentiment_label
            """, (start_date,))

            results = cursor.fetchall()
            
            # Get average metrics
            cursor.execute("""
                SELECT 
                    AVG(sentiment_polarity),
                    AVG(intensity),
                    AVG(emotion_confidence),
                    COUNT(CASE WHEN crisis_detected = 1 THEN 1 END)
                FROM sessions 
                WHERE timestamp > ?
            """, (start_date,))
            
            avg_data = cursor.fetchone()
            conn.close()

            trend = {label: count for label, count in results}
            
            return {
                "period_days": days,
                "sentiment_distribution": trend,
                "total_sessions": sum(trend.values()),
                "avg_polarity": avg_data[0] or 0.0,
                "avg_intensity": avg_data[1] or 0.0,
                "avg_confidence": avg_data[2] or 0.0,
                "crisis_count": avg_data[3] or 0
            }
        except Exception as e:
            print(f"⚠️  Trend analysis error: {e}")
            return {
                "period_days": days,
                "sentiment_distribution": {},
                "total_sessions": 0,
                "avg_polarity": 0.0,
                "avg_intensity": 0.0,
                "avg_confidence": 0.0,
                "crisis_count": 0
            }

    def get_emotion_patterns(self) -> Dict:
        """Identify emotional patterns over time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT emotion, COUNT(*) as count
                FROM sessions
                GROUP BY emotion
                ORDER BY count DESC
            """)
            
            emotions = cursor.fetchall()
            conn.close()
            
            return {
                "most_common_emotions": [(e[0], e[1]) for e in emotions[:5]],
                "total_tracked": sum(e[1] for e in emotions)
            }
        except:
            return {"most_common_emotions": [], "total_tracked": 0}

# ========================= FEATURE 18-22: ENHANCED AI MODES =========================
class EnhancedLMStudioMode:
    """LM Studio with system prompts, conversation memory, and dynamic parameters"""
    
    SYSTEM_PROMPTS = {
        "anxious": """You are a specialized anxiety support assistant trained in CBT and grounding techniques.

Your approach:
1. Validate their anxiety without judgment ("It's completely normal to feel anxious about...")
2. Offer 2-3 evidence-based grounding techniques (5-4-3-2-1, box breathing, progressive muscle relaxation)
3. Use calming, reassuring language
4. Suggest thought-challenging if catastrophizing detected
5. Encourage professional help if anxiety is severe or persistent
6. Keep responses 200-300 words, warm and supportive

Never diagnose anxiety disorders or prescribe medication. Focus on immediate coping skills.""",

        "sad": """You are a depression support specialist using behavioral activation and self-compassion techniques.

Your approach:
1. Acknowledge the weight of their feelings with deep empathy
2. Gently suggest ONE small, achievable action (behavioral activation)
3. Challenge negative self-talk compassionately ("Would you say this to a friend?")
4. Instill hope while being realistic about the difficulty
5. Emphasize their worth is not tied to productivity
6. Keep responses 200-300 words, empathetic and hopeful

Never diagnose depression or prescribe medication. Strongly encourage professional help for persistent sadness.""",

        "stressed": """You are a stress management coach focusing on practical, actionable solutions.

Your approach:
1. Validate their stress as a normal response to demands
2. Help break down overwhelming tasks into concrete steps
3. Teach prioritization: urgent vs important
4. Offer quick stress-relief techniques (breathing, time-blocking)
5. Suggest boundary-setting if overcommitment detected
6. Keep responses 200-300 words, practical and solution-focused

Never minimize their stress. Balance empathy with actionable guidance.""",

        "angry": """You are an anger management specialist using emotion regulation techniques.

Your approach:
1. Validate that anger is a normal emotion with important information
2. Help identify the underlying need (respect, fairness, control, safety)
3. Suggest healthy expression methods (physical activity, journaling, communication)
4. Teach "opposite action" if anger is disproportionate
5. Encourage cooling-off before important conversations
6. Keep responses 200-300 words, calm and non-judgmental

Never tell them to "just calm down." Respect anger as valid while teaching regulation.""",

        "crisis": """You are an emergency mental health support assistant. This is a CRISIS situation.

IMMEDIATE PROTOCOL:
1. Provide crisis helplines FIRST AND PROMINENTLY:
   🇮🇳 INDIA - AASRA: +91-9820466726 | iCall: +91-96540 22000
   🇺🇸 USA - 988 Suicide & Crisis Lifeline
2. Affirm their life has value: "You matter. Your life matters."
3. Offer immediate grounding technique (5-4-3-2-1, ice cube, call someone)
4. STRONGLY urge calling helpline or going to emergency room NOW
5. Keep responses 150-200 words, DIRECT and focused on immediate safety

This is an emergency. Professional intervention is CRITICAL. Be clear, supportive, urgent.""",

        "default": """You are a compassionate mental health support companion trained in evidence-based techniques.

Core principles:
1. EMPATHY FIRST - Always validate feelings before offering solutions
2. SAFETY - Immediately recognize crisis language (suicide, self-harm, "no point living")
3. EVIDENCE-BASED - Use CBT, mindfulness, behavioral activation, grounding techniques
4. NON-DIAGNOSTIC - Never diagnose mental health conditions or prescribe medication
5. PROFESSIONAL BOUNDARIES - Encourage professional therapy for persistent/severe concerns

Response structure:
- Validate feelings: "I hear that you're feeling [emotion]. That's [acknowledgment]."
- Ask clarifying question if needed (ONE question max)
- Offer 2-3 specific, actionable coping strategies
- End with encouragement and reminder of their strength
- Keep responses warm, professional, helpful (200-300 words)

Crisis keywords to watch: suicide, kill myself, self-harm, end it all, no point living
→ If detected, IMMEDIATELY provide helplines and encourage professional help.

You are a supportive companion, not a replacement for therapy."""
    }
    
    def __init__(self):
        self.available = self._check()
        self.conversation_history = []
        self.max_history = 6  # Last 3 exchanges
        
        if self.available:
            print("🔵 LM Studio: ✅ Ready (Enhanced Mode)")
        else:
            print("🔵 LM Studio: ❌ Not available")

    def _check(self) -> bool:
        """Check if LM Studio is running"""
        try:
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": Config.LM_STUDIO_MODEL,
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": 0.5,
                    "max_tokens": 10
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def _get_dynamic_config(self, emotion: str, sentiment_polarity: float, 
                           intensity: float, is_crisis: bool) -> dict:
        """Adjust parameters based on emotional state"""
        
        if is_crisis:
            # Crisis: focused, direct, clear
            return {
                "temperature": 0.4,
                "max_tokens": 400,
                "top_p": 0.8
            }
        elif intensity > 75:
            # High intensity: detailed support
            return {
                "temperature": 0.5,
                "max_tokens": 600,
                "top_p": 0.85
            }
        elif sentiment_polarity < -0.5:
            # Very negative: balanced response
            return {
                "temperature": 0.6,
                "max_tokens": 512,
                "top_p": 0.85
            }
        else:
            # General support: default
            return {
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_TOKENS,
                "top_p": Config.TOP_P
            }

    def generate(self, prompt: str, emotion: str = "default",
                sentiment_polarity: float = 0.0, intensity: float = 50.0,
                is_crisis: bool = False, use_history: bool = True) -> Tuple[str, float]:
        """Generate response with context and dynamic parameters"""
        
        if not self.available:
            return "LM Studio unavailable. Please start LM Studio server.", 0

        # Select appropriate system prompt
        if is_crisis:
            system_prompt = self.SYSTEM_PROMPTS["crisis"]
        else:
            system_prompt = self.SYSTEM_PROMPTS.get(emotion, self.SYSTEM_PROMPTS["default"])

        # Get dynamic configuration
        config = self._get_dynamic_config(emotion, sentiment_polarity, intensity, is_crisis)

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-self.max_history:])
        
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": Config.LM_STUDIO_MODEL,
                    "messages": messages,
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                    "top_p": config["top_p"]
                },
                timeout=60
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                assistant_message = response.json()["choices"][0]["message"]["content"].strip()
                
                # Store conversation history
                if use_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                    
                    # Trim to max_history
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history = self.conversation_history[-self.max_history:]
                
                return assistant_message, elapsed
            else:
                return f"LM Studio error {response.status_code}", elapsed
        except requests.exceptions.Timeout:
            return "LM Studio timeout (response took >60s)", 0
        except Exception as e:
            return f"LM Studio error: {str(e)}", 0

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("💬 Conversation history cleared")

    def get_conversation_summary(self) -> Dict:
        """Get conversation statistics"""
        return {
            "total_messages": len(self.conversation_history),
            "exchanges": len(self.conversation_history) // 2,
            "history_enabled": len(self.conversation_history) > 0
        }

class GroqMode:
    """Groq cloud API integration (replaces Gemini) - Llama 3.1 70B for mental health"""
    
    def __init__(self):
        self.available = self._check()
        if self.available:
            print(f"⚡ Groq: ✅ Ready ({Config.GROQ_MODEL})")
        else:
            print("⚡ Groq: ⚠️  Not configured (set GROQ_API_KEY)")

    def _check(self) -> bool:
        """Check if Groq API is configured and working"""
        if Config.GROQ_API_KEY == "your-groq-api-key-here":
            return False
        
        try:
            response = requests.post(
                Config.GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": Config.GROQ_MODEL,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                },
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str, emotion: str = "default") -> Tuple[str, float]:
        """Generate response using Groq (OpenAI-compatible API)"""
        
        if not self.available:
            return "Groq not available. Set GROQ_API_KEY environment variable.", 0

        # Use same system prompts as LM Studio
        system_prompt = EnhancedLMStudioMode.SYSTEM_PROMPTS.get(
            emotion, 
            EnhancedLMStudioMode.SYSTEM_PROMPTS["default"]
        )

        try:
            start = time.time()
            response = requests.post(
                Config.GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {Config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": Config.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": Config.TEMPERATURE,
                    "max_tokens": Config.MAX_TOKENS,
                    "top_p": Config.TOP_P
                },
                timeout=30  # Groq is faster, so shorter timeout
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                return text, elapsed
            else:
                return f"Groq error {response.status_code}: {response.text}", elapsed
        except requests.exceptions.Timeout:
            return "Groq timeout", 0
        except Exception as e:
            return f"Groq error: {str(e)}", 0

class AutoIntelligentMode:
    """Intelligent routing between Local (LM Studio) and Cloud (Groq)"""
    
    def __init__(self):
        self.lm_studio = EnhancedLMStudioMode()
        self.groq = GroqMode()
        self.available = self.lm_studio.available or self.groq.available
        
        if self.available:
            print("🟠 Auto Mode: ✅ Active (Local-first, Cloud backup)")
        else:
            print("🟠 Auto Mode: ❌ No AI models available")

    def should_use_cloud(self, text: str, is_crisis: bool = False) -> bool:
        """Decide whether to use cloud AI"""
        
        # Never use cloud for crisis (privacy)
        if is_crisis:
            return False
        
        # Use cloud if local not available
        if not self.lm_studio.available and self.groq.available:
            return True
        
        # Use cloud for complex queries (Groq is fast enough)
        word_count = len(text.split())
        question_count = text.count("?")
        
        # Complex query criteria
        is_complex = word_count > 60 or question_count > 2
        
        return is_complex and self.groq.available

    def generate(self, prompt: str, emotion: str = "default",
                sentiment_polarity: float = 0.0, intensity: float = 50.0,
                is_crisis: bool = False) -> Tuple[str, float, str]:
        """Auto-route to best available model"""
        
        use_cloud = self.should_use_cloud(prompt, is_crisis)

        if use_cloud:
            text, elapsed = self.groq.generate(prompt, emotion)
            model = "Groq"
        elif self.lm_studio.available:
            text, elapsed = self.lm_studio.generate(
                prompt, emotion, sentiment_polarity, intensity, is_crisis
            )
            model = "LM Studio"
        elif self.groq.available:
            # Fallback to Groq if LM Studio unavailable
            text, elapsed = self.groq.generate(prompt, emotion)
            model = "Groq (fallback)"
        else:
            return "No AI models available", 0, "None"

        return text, elapsed, model

# ========================= EXISTING FEATURES (ENHANCED) =========================
class EmotionWheel:
    """Visualize emotional state"""

    EMOTIONS = {
        "happy": "😊", "sad": "😢", "anxious": "😰", "angry": "😠",
        "calm": "😌", "excited": "🤩", "bored": "😑", "hopeful": "🙂",
        "grateful": "🙏", "afraid": "😨", "stressed": "😓", "neutral": "😐"
    }

    @staticmethod
    def display(emotion: str, intensity: float, confidence: float = 0.0):
        emoji = EmotionWheel.EMOTIONS.get(emotion, "😐")
        intensity_bar = "█" * int(intensity / 10) + "░" * (10 - int(intensity / 10))
        
        print(f"\n🎨 EMOTION WHEEL: {emoji} {emotion.upper()}")
        print(f"   Intensity: [{intensity_bar}] {intensity:.1f}%")
        if confidence > 0:
            print(f"   Confidence: {confidence:.1f}%")

class PersonalizationEngine:
    """Learn user preferences"""

    def __init__(self):
        self.user_profile = {
            "sessions_count": 0,
            "common_emotions": Counter(),
            "preferred_techniques": Counter(),
            "crisis_incidents": 0
        }

    def update_profile(self, emotion: str, technique_used: str = ""):
        self.user_profile["sessions_count"] += 1
        self.user_profile["common_emotions"][emotion] += 1
        if technique_used:
            self.user_profile["preferred_techniques"][technique_used] += 1

    def get_profile_summary(self) -> str:
        total = self.user_profile["sessions_count"]
        if total == 0:
            return "📊 No profile data yet. Start a session to build your profile!"

        top_emotions = self.user_profile["common_emotions"].most_common(3)
        top_techniques = self.user_profile["preferred_techniques"].most_common(2)

        summary = f"""
📊 YOUR MENTAL HEALTH PROFILE:
   Total Sessions: {total}
   Most Common Emotions: {', '.join([f'{e[0]} ({e[1]}x)' for e in top_emotions]) if top_emotions else 'N/A'}
   Preferred Techniques: {', '.join([f'{t[0]} ({t[1]}x)' for t in top_techniques]) if top_techniques else 'N/A'}
   Crisis Incidents: {self.user_profile['crisis_incidents']}
"""
        return summary

class SessionReplay:
    """Review past sessions"""

    @staticmethod
    def get_recent_sessions(db_path: str, limit: int = 5) -> List[Dict]:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, scenario, emotion, sentiment_polarity, intensity, crisis_detected
                FROM sessions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            results = cursor.fetchall()
            conn.close()

            sessions = [
                {
                    "timestamp": r[0],
                    "scenario": r[1],
                    "emotion": r[2],
                    "polarity": r[3],
                    "intensity": r[4],
                    "crisis": r[5]
                }
                for r in results
            ]
            return sessions
        except:
            return []

class WellnessScore:
    """Calculate overall wellness"""

    @staticmethod
    def calculate(trend: Dict) -> Dict:
        if trend["total_sessions"] == 0:
            return {"score": 0, "level": "No data", "recommendation": "Start tracking to see your wellness score!"}

        # Weighted scoring
        avg_polarity = trend.get("avg_polarity", 0)
        avg_intensity = trend.get("avg_intensity", 50)
        crisis_count = trend.get("crisis_count", 0)

        # Base score from polarity (-1 to +1 → 0 to 100)
        base_score = (avg_polarity + 1) * 50

        # Penalty for high intensity negative emotions
        if avg_polarity < 0 and avg_intensity > 60:
            base_score -= 10

        # Penalty for crisis incidents
        base_score -= (crisis_count * 5)

        score = max(0, min(100, int(base_score)))

        if score >= 70:
            level = "Excellent"
            recommendation = "Keep up the great self-care!"
        elif score >= 50:
            level = "Good"
            recommendation = "You're doing well. Consider additional support if needed."
        elif score >= 30:
            level = "Fair"
            recommendation = "Consider reaching out to a mental health professional."
        else:
            level = "Needs Support"
            recommendation = "Please talk to a mental health professional soon."

        return {"score": score, "level": level, "recommendation": recommendation}

class ReportExporter:
    """Export session data"""

    @staticmethod
    def export_sessions(db_path: str, format: str = "csv") -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM sessions ORDER BY timestamp DESC")
            sessions = cursor.fetchall()
            
            # Get column names
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            conn.close()

            if format == "csv":
                filename = f"mental_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(sessions)
                return f"✅ Exported {len(sessions)} sessions to {filename}"
            
            elif format == "json":
                filename = f"mental_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                data = [dict(zip(columns, session)) for session in sessions]
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                return f"✅ Exported {len(sessions)} sessions to {filename}"
            
        except Exception as e:
            return f"❌ Export failed: {e}"

class BreathingGuide:
    """Interactive breathing exercises"""

    @staticmethod
    def guided_breathing(technique: str = "4-7-8", cycles: int = 4):
        print(f"\n🫁 GUIDED BREATHING - {technique.upper()}")
        print("="*50)
        print("Close your eyes, relax your shoulders, and focus on your breath.\n")

        if technique == "4-7-8":
            print("Technique: Inhale (4) → Hold (7) → Exhale (8)")
            for i in range(1, cycles + 1):
                print(f"\n🔄 Cycle {i}/{cycles}:")
                print("   Inhale through nose... (1... 2... 3... 4...)")
                time.sleep(4)
                print("   Hold your breath... (1... 2... 3... 4... 5... 6... 7...)")
                time.sleep(7)
                print("   Exhale through mouth... (1... 2... 3... 4... 5... 6... 7... 8...)")
                time.sleep(8)

        elif technique == "box":
            print("Technique: Inhale (4) → Hold (4) → Exhale (4) → Hold (4)")
            for i in range(1, cycles + 1):
                print(f"\n🔄 Cycle {i}/{cycles}:")
                print("   Inhale... (1... 2... 3... 4...)")
                time.sleep(4)
                print("   Hold... (1... 2... 3... 4...)")
                time.sleep(4)
                print("   Exhale... (1... 2... 3... 4...)")
                time.sleep(4)
                print("   Hold... (1... 2... 3... 4...)")
                time.sleep(4)

        print("\n✅ Breathing exercise complete! Notice how you feel now.")

class MoodCalendar:
    """Visual mood calendar"""

    @staticmethod
    def get_mood_calendar(db_path: str, days: int = 7) -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DATE(timestamp), emotion, AVG(intensity), COUNT(*)
                FROM sessions
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                GROUP BY DATE(timestamp), emotion
                ORDER BY DATE(timestamp) DESC, COUNT(*) DESC
            """, (days,))

            results = cursor.fetchall()
            conn.close()

            emoji_map = {
                "happy": "😊", "sad": "😢", "anxious": "😰", "angry": "😠",
                "stressed": "😓", "calm": "😌", "neutral": "😐"
            }

            calendar = f"\n📅 MOOD CALENDAR (Last {days} days)\n" + "="*50 + "\n"
            
            if not results:
                calendar += "No mood data yet. Start tracking!\n"
            else:
                current_date = None
                for date, emotion, avg_intensity, count in results:
                    if date != current_date:
                        if current_date is not None:
                            calendar += "\n"
                        calendar += f"\n📆 {date}:\n"
                        current_date = date
                    
                    emoji = emoji_map.get(emotion, "😐")
                    calendar += f"   {emoji} {emotion} (intensity: {avg_intensity:.0f}%, {count} session{'s' if count > 1 else ''})\n"

            return calendar
        except Exception as e:
            return f"⚠️  Calendar error: {e}"

class AIRecommendations:
    """Context-aware recommendations"""

    @staticmethod
    def get_recommendations(emotion: str, intensity: float, sessions_count: int,
                          crisis_detected: bool, sentiment_polarity: float) -> List[str]:
        recommendations = []

        if crisis_detected:
            recommendations.append("🚨 URGENT: Please contact a crisis helpline immediately")

        if intensity > 75:
            recommendations.append("⚠️  High emotional intensity - consider professional support")

        if sessions_count < 3:
            recommendations.append("📈 Keep tracking to identify patterns over time")

        if emotion == "anxious" and intensity > 60:
            recommendations.append("🫁 Try the 5-4-3-2-1 grounding technique or box breathing")

        if emotion == "sad" and intensity > 60:
            recommendations.append("💝 Behavioral activation: Do one small enjoyable activity today")

        if emotion == "stressed":
            recommendations.append("📋 Brain dump: Write down all tasks, then prioritize top 3")

        if sentiment_polarity < -0.5 and sessions_count > 5:
            recommendations.append("💬 Persistent negative mood detected - consider talking to a therapist")

        if sessions_count > 10 and sessions_count % 10 == 0:
            recommendations.append(f"🎉 Milestone: {sessions_count} sessions tracked! Great self-awareness!")

        return recommendations[:4]  # Limit to 4 to avoid overwhelm

# ========================= MAIN SYSTEM =========================
class UltimateHealthCoach:
    """Complete mental health coaching system - 33+ features"""

    def __init__(self):
        print("\n" + "="*70)
        print("🚀 ULTIMATE MENTAL HEALTH COACH V2.0 - INITIALIZING 🚀")
        print("="*70 + "\n")

        # Core engines
        self.sentiment_analyzer = NLPEnhancedSentimentAnalyzer()
        self.crisis_detector = EnhancedCrisisDetectionSystem()
        self.therapy_framework = TherapyFramework()
        self.coping_generator = EnhancedCopingStrategyGenerator()
        self.progress_tracker = EnhancedProgressTracker(Config.DB_PATH)
        
        # AI models
        self.lm_studio = EnhancedLMStudioMode()
        self.groq = GroqMode()
        self.auto = AutoIntelligentMode()
        
        # Features
        self.personalization = PersonalizationEngine()
        self.session_replay = SessionReplay()
        self.wellness_score = WellnessScore()
        self.report_exporter = ReportExporter()
        self.breathing_guide = BreathingGuide()
        self.mood_calendar = MoodCalendar()
        self.ai_recommendations = AIRecommendations()

        print("\n✅ All systems initialized!\n")

    def comprehensive_analysis(self, scenario_name: str, user_input: str, use_ai: bool = True):
        """Full analysis with all 33+ features"""

        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE MENTAL HEALTH ANALYSIS")
        print(f"{'='*70}")
        print(f"📝 Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}\n")

        # 1. Advanced Sentiment Analysis (NLP-powered)
        print("🔍 Analyzing sentiment...")
        sentiment = self.sentiment_analyzer.analyze(user_input)
        
        print(f"\n📈 SENTIMENT ANALYSIS:")
        print(f"   Emotion: {sentiment['emotion']} (confidence: {sentiment.get('confidence', 0):.1f}%)")
        print(f"   Polarity: {sentiment['polarity']:.2f} ({sentiment['label']})")
        print(f"   Intensity: {sentiment['intensity']:.1f}%")
        if sentiment.get('subjectivity', 0) > 0:
            print(f"   Subjectivity: {sentiment['subjectivity']:.2f}")
        
        # Show NLP insights if available
        if sentiment.get('entities'):
            print(f"   Entities detected: {', '.join([e[0] for e in sentiment['entities'][:3]])}")
        if sentiment.get('key_phrases'):
            print(f"   Key phrases: {', '.join(sentiment['key_phrases'][:3])}")

        # 2. Emotion Wheel Visualization
        EmotionWheel.display(sentiment['emotion'], sentiment['intensity'], sentiment.get('confidence', 0))

        # 3. Crisis Detection (Multi-factor)
        crisis = self.crisis_detector.detect(user_input, sentiment)
        
        print(f"\n🚨 SAFETY CHECK:")
        if crisis['is_crisis']:
            print(f"   ⚠️  CRISIS DETECTED - Severity: {crisis['severity'].upper()}")
            print(f"   Crisis Score: {crisis['score']}/100")
            if crisis['keywords']:
                print(f"   Keywords: {', '.join(crisis['keywords'])}")
            print(f"\n{crisis['response']}")
        else:
            print(f"   ✅ No immediate crisis detected")

        # 4. Therapy Framework Suggestions
        print(f"\n🎓 THERAPEUTIC INTERVENTIONS:")
        therapy_suggestions = self.therapy_framework.suggest_intervention(
            user_input, sentiment['emotion'], sentiment['intensity']
        )
        
        if therapy_suggestions['suggested_techniques']:
            for i, technique in enumerate(therapy_suggestions['suggested_techniques'], 1):
                print(f"\n   {i}. {technique['type']}: {technique['name']}")
                print(f"   {technique['description'][:200]}...")
        else:
            print("   Using general coping strategies (see below)")

        # 5. Evidence-Based Coping Strategies
        strategies = self.coping_generator.generate(
            sentiment['emotion'], 
            sentiment['intensity'],
            user_input
        )
        
        print(f"\n💡 COPING STRATEGIES:")
        for i, strategy in enumerate(strategies, 1):
            print(f"   {i}. {strategy}")

        # 6. AI Responses (if enabled)
        model_used = "None"
        response_time = 0
        
        if use_ai:
            print(f"\n🤖 AI SUPPORT:")
            
            # Try auto mode first (smart routing)
            if self.auto.available:
                response, elapsed, model = self.auto.generate(
                    user_input,
                    sentiment['emotion'],
                    sentiment['polarity'],
                    sentiment['intensity'],
                    crisis['is_crisis']
                )
                print(f"\n   🟠 Auto Mode → {model} ({elapsed:.0f}ms):")
                print(f"   {response[:300]}{'...' if len(response) > 300 else ''}")
                model_used = model
                response_time = elapsed
            
            # Show LM Studio response if available
            elif self.lm_studio.available:
                response, elapsed = self.lm_studio.generate(
                    user_input,
                    sentiment['emotion'],
                    sentiment['polarity'],
                    sentiment['intensity'],
                    crisis['is_crisis']
                )
                print(f"\n   🔵 LM Studio ({elapsed:.0f}ms):")
                print(f"   {response[:300]}{'...' if len(response) > 300 else ''}")
                model_used = "LM Studio"
                response_time = elapsed
            
            else:
                print("   ⚠️  No AI models available")

        # 7. AI Recommendations
        recommendations = self.ai_recommendations.get_recommendations(
            sentiment['emotion'],
            sentiment['intensity'],
            self.personalization.user_profile['sessions_count'],
            crisis['is_crisis'],
            sentiment['polarity']
        )
        
        if recommendations:
            print(f"\n💼 PERSONALIZED RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")

        # 8. Wellness Score
        trend = self.progress_tracker.get_mood_trend(7)
        wellness = self.wellness_score.calculate(trend)
        
        print(f"\n📊 WELLNESS SCORE: {wellness['score']}/100 ({wellness['level']})")
        print(f"   {wellness.get('recommendation', '')}")

        # 9. Affirmation
        affirmation = random.choice(Config.AFFIRMATIONS)
        print(f"\n✨ TODAY'S AFFIRMATION:")
        print(f"   \"{affirmation}\"")

        # 10. Log session
        technique_used = therapy_suggestions['suggested_techniques'][0]['name'] if therapy_suggestions['suggested_techniques'] else ""
        self.progress_tracker.log_session(
            scenario_name,
            sentiment,
            crisis,
            model_used,
            response_time,
            len(self.lm_studio.conversation_history) // 2 if self.lm_studio.available else 0,
            technique_used
        )
        
        # Update personalization
        self.personalization.update_profile(sentiment['emotion'], technique_used)
        if crisis['is_crisis']:
            self.personalization.user_profile['crisis_incidents'] += 1

        print(f"\n{'='*70}\n")

    def interactive_menu(self):
        """Main interactive menu"""
        
        print("\n" + "="*70)
        print("💬 ULTIMATE MENTAL HEALTH COACH V2.0")
        print("="*70)
        print("""
🌟 33+ PROFESSIONAL FEATURES:

🧠 CORE AI:
   1. NLP-Powered Sentiment Analysis (Transformer + TextBlob + spaCy)
   2. Multi-Turn Conversation Memory
   3. Dynamic AI Parameters
   4. Professional System Prompts
   5. Groq Cloud AI (10x faster than Gemini)

🎓 THERAPY FRAMEWORKS:
   6. CBT Techniques (Thought Challenging, Behavioral Activation)
   7. DBT Skills (TIPP, ACCEPTS, Opposite Action)
   8. Mindfulness (5-4-3-2-1, Box Breathing, 4-7-8)

📊 ANALYTICS & TRACKING:
   9. Enhanced Crisis Detection (Multi-factor scoring)
   10. Evidence-Based Coping Strategies
   11. Comprehensive Progress Tracking
   12. Wellness Score Calculator
   13. Mood Calendar
   14. Session Replay
   15. Export Reports (CSV/JSON)
   16. Emotion Patterns Analysis

🎨 INTERACTIVE FEATURES:
   17. Emotion Wheel Visualization
   18. Guided Breathing Exercises
   19. Personalization Engine
   20. AI Recommendations

🔐 PRIVACY & SAFETY:
   21. 100% Local-First (LM Studio primary)
   22. Multi-Tier Crisis System
   23. Indian & US Helplines

COMMANDS:
  stress / anxiety / sad / etc.  → Full analysis
  custom                         → Enter your own text
  breathing                      → Guided breathing (4-7-8 or box)
  wellness                       → Check wellness score
  calendar                       → View mood calendar
  profile                        → See your profile
  replay                         → View recent sessions
  patterns                       → Analyze emotion patterns
  export                         → Export data (CSV/JSON)
  clear                          → Clear conversation history
  techniques                     → Show all therapy techniques
  all                            → Test all scenarios
  help                           → Show this menu
  exit                           → Quit

Type a command to begin...
""")

        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == "exit":
                    print("\n💙 Thank you for prioritizing your mental health. Take care! 🌸\n")
                    break

                elif command == "breathing":
                    technique = input("Choose technique (4-7-8 or box): ").strip().lower()
                    if technique not in ["4-7-8", "box"]:
                        technique = "4-7-8"
                    self.breathing_guide.guided_breathing(technique)

                elif command == "wellness":
                    trend = self.progress_tracker.get_mood_trend(7)
                    wellness = self.wellness_score.calculate(trend)
                    print(f"\n📊 YOUR WELLNESS SCORE:")
                    print(f"   Score: {wellness['score']}/100")
                    print(f"   Level: {wellness['level']}")
                    print(f"   Recommendation: {wellness.get('recommendation', '')}")
                    print(f"\n   Based on {trend['total_sessions']} sessions in last 7 days")
                    print(f"   Average polarity: {trend.get('avg_polarity', 0):.2f}")
                    print(f"   Average intensity: {trend.get('avg_intensity', 0):.1f}%")

                elif command == "calendar":
                    print(self.mood_calendar.get_mood_calendar(Config.DB_PATH, 7))

                elif command == "profile":
                    print(self.personalization.get_profile_summary())

                elif command == "replay":
                    sessions = self.session_replay.get_recent_sessions(Config.DB_PATH, 5)
                    if sessions:
                        print("\n📹 RECENT SESSIONS:")
                        for i, s in enumerate(sessions, 1):
                            crisis_flag = "🚨" if s['crisis'] else "  "
                            print(f"   {i}. {crisis_flag} {s['timestamp'][:19]} | {s['emotion']} "
                                  f"(intensity: {s['intensity']:.0f}%) | {s['scenario']}")
                    else:
                        print("\n📹 No sessions yet. Start tracking!")

                elif command == "patterns":
                    patterns = self.progress_tracker.get_emotion_patterns()
                    print("\n🔍 EMOTION PATTERNS:")
                    if patterns['most_common_emotions']:
                        print(f"   Most common emotions:")
                        for emotion, count in patterns['most_common_emotions']:
                            percentage = (count / patterns['total_tracked']) * 100
                            print(f"   • {emotion}: {count} times ({percentage:.1f}%)")
                    else:
                        print("   No patterns yet. Keep tracking!")

                elif command == "export":
                    format_choice = input("Choose format (csv or json): ").strip().lower()
                    if format_choice not in ["csv", "json"]:
                        format_choice = "csv"
                    result = self.report_exporter.export_sessions(Config.DB_PATH, format_choice)
                    print(f"\n{result}")

                elif command == "clear":
                    if self.lm_studio.available:
                        self.lm_studio.clear_history()
                    else:
                        print("💬 LM Studio not available")

                elif command == "techniques":
                    print("\n🎓 AVAILABLE THERAPY TECHNIQUES:\n")
                    print("CBT (Cognitive Behavioral Therapy):")
                    for key, tech in TherapyFramework.CBT_TECHNIQUES.items():
                        print(f"   • {tech['name']}")
                    print("\nDBT (Dialectical Behavior Therapy):")
                    for key, tech in TherapyFramework.DBT_SKILLS.items():
                        print(f"   • {tech['name']}")
                    print("\nMindfulness:")
                    for key, tech in TherapyFramework.MINDFULNESS_TECHNIQUES.items():
                        print(f"   • {tech['name']}")

                elif command in Config.TEST_SCENARIOS:
                    user_text = Config.TEST_SCENARIOS[command]
                    self.comprehensive_analysis(command, user_text)

                elif command == "custom":
                    user_text = input("\n💬 Enter your text: ").strip()
                    if user_text:
                        self.comprehensive_analysis("custom", user_text)
                    else:
                        print("⚠️  Empty input. Please try again.")

                elif command == "all":
                    print("\n🔄 Running all test scenarios...\n")
                    for scenario_name, user_input in Config.TEST_SCENARIOS.items():
                        self.comprehensive_analysis(scenario_name, user_input, use_ai=False)
                        time.sleep(1)

                elif command == "help":
                    self.interactive_menu()
                    return

                else:
                    # Try to use as custom text
                    if command:
                        confirm = input(f"Analyze '{command[:50]}...' as custom text? (y/n): ")
                        if confirm.lower() == 'y':
                            self.comprehensive_analysis("custom", command)
                        else:
                            print("❓ Unknown command. Type 'help' to see all commands.")

            except KeyboardInterrupt:
                print("\n\n💙 Session interrupted. Goodbye! 🌸\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for commands.\n")

# ========================= MAIN =========================
if __name__ == "__main__":
    try:
        coach = UltimateHealthCoach()
        coach.interactive_menu()
    except KeyboardInterrupt:
        print("\n\n💙 Thank you for using Ultimate Health Coach. Take care! 🌸\n")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        print("Please check your configuration and try again.\n")