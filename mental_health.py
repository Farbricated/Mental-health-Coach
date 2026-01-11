"""
🚀 ULTIMATE MENTAL HEALTH AI CHATBOT - PROFESSIONAL ENTERPRISE EDITION 🚀
15+ UNIQUE & INNOVATIVE FEATURES
Multi-Modal AI | Advanced Analytics | Personalization Engine | Export Capabilities
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

# ========================= CONFIGURATION =========================
class Config:
    """Advanced enterprise configuration"""

    LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODEL = "llm"

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    TEMPERATURE = 0.7
    MAX_TOKENS = 512
    TOP_P = 0.9

    DB_PATH = "mental_health_enterprise.db"
    EXPORT_PATH = "mental_health_reports"

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

    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end it all", "want to die", "harm myself",
        "no point living", "can't go on", "better off dead", "jump off", "overdose"
    ]

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

# ========================= FEATURE 1: ADVANCED SENTIMENT ANALYSIS =========================
class AdvancedSentimentAnalyzer:
    """Sophisticated sentiment detection"""

    @staticmethod
    def analyze(text: str) -> Dict:
        positive_words = {
            "happy", "joy", "grateful", "excited", "hopeful", "better", "good",
            "calm", "peaceful", "loved", "appreciated", "proud", "confident",
            "beautiful", "wonderful", "amazing", "fantastic", "excellent", "improved"
        }

        negative_words = {
            "sad", "anxious", "stressed", "depressed", "worried", "overwhelmed",
            "hopeless", "awful", "terrible", "useless", "failed", "alone",
            "scared", "angry", "frustrated", "disgusted", "ashamed", "broken", "bored"
        }

        neutral_words = {"okay", "fine", "normal", "regular", "average"}

        text_lower = text.lower()
        words = text_lower.split()

        pos_score = sum(1 for word in words if word in positive_words)
        neg_score = sum(1 for word in words if word in negative_words)
        neu_score = sum(1 for word in words if word in neutral_words)

        total = pos_score + neg_score + neu_score

        word_breakdown = {
            "positive": pos_score,
            "negative": neg_score,
            "neutral": neu_score
        }

        if total == 0:
            return {
                "polarity": 0.0,
                "label": "neutral",
                "intensity": 0.0,
                "emotion": "neutral",
                "word_breakdown": word_breakdown
            }

        polarity = (pos_score - neg_score) / max(total, 1)
        intensity = (pos_score + neg_score) / max(total, 1) * 100

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

        if "grief" in text_lower or "loss" in text_lower:
            emotion = "sad"
        elif "anxious" in text_lower or "nervous" in text_lower:
            emotion = "anxious"
        elif "angry" in text_lower or "frustrated" in text_lower:
            emotion = "angry"
        elif "bored" in text_lower:
            emotion = "bored"
        elif "hopeful" in text_lower or "excited" in text_lower:
            emotion = "hopeful"
        else:
            emotion = label

        return {
            "polarity": polarity,
            "label": label,
            "intensity": intensity,
            "emotion": emotion,
            "word_breakdown": word_breakdown
        }

# ========================= FEATURE 2: CRISIS DETECTION =========================
class CrisisDetectionSystem:
    """Advanced crisis detection with severity levels"""

    def __init__(self):
        self.crisis_history = []

    def detect(self, text: str, sentiment: Dict) -> Dict:
        text_lower = text.lower()
        detected_keywords = [kw for kw in Config.CRISIS_KEYWORDS if kw in text_lower]

        severity = "none"
        has_crisis = False

        if detected_keywords and sentiment['polarity'] < -0.5:
            severity = "critical"
            has_crisis = True
        elif detected_keywords or sentiment['polarity'] < -0.7:
            severity = "high"
            has_crisis = True
        elif sentiment['polarity'] < -0.4:
            severity = "moderate"
            has_crisis = True

        crisis_response = self._get_response(severity)

        if has_crisis:
            self.crisis_history.append({
                "timestamp": datetime.now(),
                "severity": severity,
                "keywords": detected_keywords
            })

        return {
            "is_crisis": has_crisis,
            "severity": severity,
            "keywords": detected_keywords,
            "response": crisis_response
        }

    @staticmethod
    def _get_response(severity: str) -> str:
        responses = {
            "critical": """🚨 IMMEDIATE HELP NEEDED
🇮🇳 AASRA: +91-9820466726 | iCall: +91-96540 22000
🇺🇸 988 | Crisis Text: HOME to 741741
💙 You matter. Help NOW.""",
            "high": """⚠️ URGENT SUPPORT
🇮🇳 NIMHANS: +91-80-26995000
🇺🇸 SAMHSA: 1-800-662-4357
You're not alone.""",
            "moderate": """💙 SUPPORT AVAILABLE
Reach out to someone you trust or a professional.""",
            "none": ""
        }
        return responses.get(severity, "")

# ========================= FEATURE 3: SMART COPING STRATEGIES =========================
class CopingStrategyGenerator:
    """Generate personalized coping strategies"""

    COPING_STRATEGIES = {
        "anxiety": [
            "Box breathing: 4-in, 4-hold, 4-out, 4-hold",
            "Ground yourself: 5-4-3-2-1 technique",
            "Progressive muscle relaxation",
        ],
        "depression": [
            "Small daily goals",
            "Reach out to one person",
            "Gentle physical activity",
        ],
        "stress": [
            "Break tasks into steps",
            "Time blocking: 25/5",
            "Prioritize truly urgent",
        ],
        "bored": [
            "Try something new",
            "Connect with friends",
            "Learn online",
        ]
    }

    @staticmethod
    def generate(emotion: str) -> List[str]:
        strategies = CopingStrategyGenerator.COPING_STRATEGIES.get(emotion, [])
        if not strategies:
            strategies = CopingStrategyGenerator.COPING_STRATEGIES.get("stress", [])
        return random.sample(strategies, min(3, len(strategies)))

# ========================= FEATURE 4: PROGRESS TRACKING =========================
class ProgressTracker:
    """Track user progress over time"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    scenario TEXT,
                    sentiment_polarity REAL,
                    sentiment_label TEXT,
                    emotion TEXT,
                    crisis_detected INTEGER,
                    model_used TEXT,
                    response_time REAL
                )
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB warning: {e}")

    def log_session(self, scenario: str, sentiment: Dict, crisis: bool, 
                   model: str, response_time: float):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), scenario, sentiment['polarity'],
                  sentiment['label'], sentiment['emotion'], 1 if crisis else 0,
                  model, response_time))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Logging warning: {e}")

    def get_mood_trend(self, days: int = 7) -> Dict:
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
            conn.close()

            trend = {label: count for label, count in results}
            return {
                "period_days": days,
                "sentiment_distribution": trend,
                "total_sessions": sum(trend.values())
            }
        except:
            return {"period_days": days, "sentiment_distribution": {}, "total_sessions": 0}

# ========================= FEATURE 5-7: AI MODES =========================
class LMStudioMode:
    def __init__(self):
        self.available = self._check()
        if self.available:
            print("🔵 LM Studio: ✅ Ready")

    def _check(self) -> bool:
        try:
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={"model": "llm", "messages": [{"role": "user", "content": "hi"}],
                      "temperature": 0.5, "max_tokens": 10},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str) -> Tuple[str, float]:
        if not self.available:
            return "LM Studio unavailable", 0

        try:
            start = time.time()
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": "llm",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": Config.TEMPERATURE,
                    "max_tokens": Config.MAX_TOKENS,
                    "top_p": Config.TOP_P
                },
                timeout=60
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                text = response.json()["choices"][0]["message"]["content"].strip()
                return text, elapsed
            return f"Error {response.status_code}", elapsed
        except Exception as e:
            return f"Error: {str(e)}", 0

class GeminiMode:
    def __init__(self):
        self.available = self._check()
        if not self.available:
            print("🟢 Gemini: ⚠️  Not configured")

    def _check(self) -> bool:
        if Config.GEMINI_API_KEY == "your-api-key-here":
            return False
        try:
            response = requests.post(
                f"{Config.GEMINI_URL}?key={Config.GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": "hi"}]}]},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt: str) -> Tuple[str, float]:
        if not self.available:
            return "Gemini not available", 0

        try:
            start = time.time()
            response = requests.post(
                f"{Config.GEMINI_URL}?key={Config.GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30
            )
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                return text, elapsed
            return f"Error {response.status_code}", elapsed
        except Exception as e:
            return f"Error: {str(e)}", 0

class AutoIntelligentMode:
    def __init__(self):
        self.lm_studio = LMStudioMode()
        self.gemini = GeminiMode()
        self.available = self.lm_studio.available or self.gemini.available
        print("🟠 Auto Mode: ✅ Active")

    def should_use_gemini(self, text: str) -> bool:
        word_count = len(text.split())
        question_count = text.count("?")
        return word_count > 60 or question_count > 2

    def generate(self, prompt: str) -> Tuple[str, float, str]:
        use_gemini = self.should_use_gemini(prompt)

        if use_gemini and self.gemini.available:
            text, elapsed = self.gemini.generate(prompt)
            return text, elapsed, "Gemini"
        elif self.lm_studio.available:
            text, elapsed = self.lm_studio.generate(prompt)
            return text, elapsed, "LM Studio"

        return "No models available", 0, "None"

# ========================= FEATURE 8: EMOTION WHEEL VISUALIZATION =========================
class EmotionWheel:
    """NEW: Visualize emotional state"""

    EMOTIONS = {
        "happy": "😊",
        "sad": "😢",
        "anxious": "😰",
        "angry": "😠",
        "calm": "😌",
        "excited": "🤩",
        "bored": "😑",
        "hopeful": "🙂",
        "grateful": "🙏",
        "afraid": "😨"
    }

    @staticmethod
    def display(emotion: str, intensity: float):
        emoji = EmotionWheel.EMOTIONS.get(emotion, "😐")
        intensity_bar = "█" * int(intensity / 10) + "░" * (10 - int(intensity / 10))
        print(f"\nEMOTION WHEEL: {emoji} {emotion.upper()}")
        print(f"[{intensity_bar}] {intensity:.1f}%")

# ========================= FEATURE 9: PERSONALIZATION ENGINE =========================
class PersonalizationEngine:
    """NEW: Learn user preferences and customize responses"""

    def __init__(self):
        self.user_profile = {
            "preferred_model": "auto",
            "favorite_strategy": None,
            "response_style": "supportive",
            "sessions_count": 0,
            "common_emotions": Counter()
        }

    def update_profile(self, emotion: str):
        self.user_profile["sessions_count"] += 1
        self.user_profile["common_emotions"][emotion] += 1

    def get_profile_summary(self) -> str:
        total = self.user_profile["sessions_count"]
        if total == 0:
            return "No profile data yet."

        top_emotion = self.user_profile["common_emotions"].most_common(1)
        summary = f"""
📊 YOUR PROFILE:
  Sessions: {total}
  Most Common: {top_emotion[0][0] if top_emotion else 'N/A'}
  Preferred Model: {self.user_profile['preferred_model']}
"""
        return summary

# ========================= FEATURE 10: SESSION REPLAY =========================
class SessionReplay:
    """NEW: Replay and learn from past sessions"""

    @staticmethod
    def get_recent_sessions(db_path: str, limit: int = 5) -> List[Dict]:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, scenario, emotion, sentiment_polarity
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
                    "polarity": r[3]
                }
                for r in results
            ]
            return sessions
        except:
            return []

# ========================= FEATURE 11: WELLNESS SCORE =========================
class WellnessScore:
    """NEW: Calculate overall wellness score"""

    @staticmethod
    def calculate(trend: Dict) -> Dict:
        if trend["total_sessions"] == 0:
            return {"score": 0, "level": "No data"}

        sentiment_dist = trend["sentiment_distribution"]

        positive_count = sentiment_dist.get("very_positive", 0) + sentiment_dist.get("positive", 0)
        negative_count = sentiment_dist.get("very_negative", 0) + sentiment_dist.get("negative", 0)
        neutral_count = sentiment_dist.get("neutral", 0)

        total = positive_count + negative_count + neutral_count

        if total == 0:
            score = 50
        else:
            score = int((positive_count / total * 100) + (neutral_count / total * 30))

        if score >= 70:
            level = "Excellent"
        elif score >= 50:
            level = "Good"
        elif score >= 30:
            level = "Fair"
        else:
            level = "Needs Support"

        return {"score": score, "level": level}

# ========================= FEATURE 12: EXPORT REPORTS =========================
class ReportExporter:
    """NEW: Export sessions to CSV"""

    @staticmethod
    def export_sessions(db_path: str, filename: str = "mental_health_report.csv"):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM sessions")
            sessions = cursor.fetchall()
            conn.close()

            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Scenario", "Polarity", "Label", "Emotion", "Crisis", "Model", "Time(ms)"])
                writer.writerows(sessions)

            return f"✅ Exported to {filename}"
        except Exception as e:
            return f"❌ Export failed: {e}"

# ========================= FEATURE 13: BREATHING GUIDE =========================
class BreathingGuide:
    """NEW: Interactive breathing exercise"""

    @staticmethod
    def guided_breathing(technique: str = "4-7-8"):
        print(f"\n🫁 GUIDED BREATHING - {technique.upper()}")
        print("="*50)

        if technique == "4-7-8":
            print("Inhale (4) → Hold (7) → Exhale (8)")
            for i in range(1, 5):
                print(f"\nCycle {i}/4:")
                print("  Inhale... (4 seconds)")
                time.sleep(4)
                print("  Hold... (7 seconds)")
                time.sleep(7)
                print("  Exhale... (8 seconds)")
                time.sleep(8)

        print("\n✅ Breathing exercise complete!")

# ========================= FEATURE 14: MOOD CALENDAR =========================
class MoodCalendar:
    """NEW: Visual mood tracking calendar"""

    @staticmethod
    def get_mood_calendar(db_path: str, days: int = 7) -> str:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT DATE(timestamp), sentiment_label, COUNT(*)
                FROM sessions
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp) DESC
                LIMIT ?
            """, (days,))

            results = cursor.fetchall()
            conn.close()

            mood_map = {
                "very_positive": "🟢",
                "positive": "🟢",
                "neutral": "🟡",
                "negative": "🔴",
                "very_negative": "🔴"
            }

            calendar = "\n📅 MOOD CALENDAR (Last 7 days)\n"
            for date, mood, count in results:
                emoji = mood_map.get(mood, "⚫")
                calendar += f"{emoji} {date}: {mood} ({count} sessions)\n"

            return calendar
        except:
            return "No calendar data available"

# ========================= FEATURE 15: AI RECOMMENDATIONS =========================
class AIRecommendations:
    """NEW: AI-powered personalized recommendations"""

    @staticmethod
    def get_recommendations(emotion: str, intensity: float, sessions_count: int) -> List[str]:
        recommendations = []

        if intensity > 75:
            recommendations.append("🚨 High intensity detected - consider talking to a professional")

        if sessions_count < 3:
            recommendations.append("📈 Continue using the app to track patterns")

        if emotion == "anxious":
            recommendations.append("💡 Try the 5-4-3-2-1 grounding technique")

        if emotion == "sad":
            recommendations.append("💝 Connect with someone close to you")

        if emotion == "bored":
            recommendations.append("🎯 Explore a new hobby or skill")

        return recommendations[:3]

# ========================= MAIN SYSTEM =========================
class UltimateHealthCoach:
    """Complete mental health coaching system with 15+ features"""

    def __init__(self):
        print("\n" + "="*70)
        print("🚀 ULTIMATE MENTAL HEALTH COACH - INITIALIZING 🚀")
        print("="*70 + "\n")

        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.crisis_detector = CrisisDetectionSystem()
        self.coping_generator = CopingStrategyGenerator()
        self.progress_tracker = ProgressTracker(Config.DB_PATH)
        self.personalization = PersonalizationEngine()
        self.session_replay = SessionReplay()
        self.wellness_score = WellnessScore()
        self.report_exporter = ReportExporter()
        self.breathing_guide = BreathingGuide()
        self.mood_calendar = MoodCalendar()
        self.ai_recommendations = AIRecommendations()

        self.lm_studio = LMStudioMode()
        self.gemini = GeminiMode()
        self.auto = AutoIntelligentMode()

    def comprehensive_analysis(self, scenario_name: str, user_input: str):
        """Full analysis with all 15+ features"""

        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE MENTAL HEALTH ANALYSIS")
        print(f"{'='*70}")

        # 1. Sentiment
        sentiment = self.sentiment_analyzer.analyze(user_input)
        print(f"\n📈 SENTIMENT: {sentiment['label']} (Polarity: {sentiment['polarity']:.2f})")

        # 2. Emotion Wheel (FEATURE 8)
        EmotionWheel.display(sentiment['emotion'], sentiment['intensity'])

        # 3. Crisis Detection
        crisis = self.crisis_detector.detect(user_input, sentiment)
        print(f"\n🚨 SAFETY: {'🚨 CRISIS' if crisis['is_crisis'] else '✅ SAFE'}")

        # 4. Coping Strategies
        strategies = self.coping_generator.generate(sentiment['emotion'])
        print(f"\n💡 STRATEGIES:")
        for s in strategies:
            print(f"   • {s}")

        # 5-7. AI Responses
        print(f"\n🤖 AI RESPONSES:")
        if self.lm_studio.available:
            resp, elapsed = self.lm_studio.generate(user_input)
            print(f"   🔵 LM Studio ({elapsed:.0f}ms): {resp[:150]}...")
        if self.auto.available:
            resp, elapsed, model = self.auto.generate(user_input)
            print(f"   🟠 Auto ({model}, {elapsed:.0f}ms): {resp[:150]}...")

        # 8. Log and personalize
        self.progress_tracker.log_session(scenario_name, sentiment, crisis['is_crisis'], 
                                         "LM Studio", 0)
        self.personalization.update_profile(sentiment['emotion'])

        # 9. Recommendations (FEATURE 15)
        recs = self.ai_recommendations.get_recommendations(
            sentiment['emotion'], 
            sentiment['intensity'],
            self.personalization.user_profile["sessions_count"]
        )
        if recs:
            print(f"\n💼 RECOMMENDATIONS:")
            for rec in recs:
                print(f"   {rec}")

        # 10. Wellness Score (FEATURE 11)
        trend = self.progress_tracker.get_mood_trend(7)
        wellness = self.wellness_score.calculate(trend)
        print(f"\n📊 WELLNESS SCORE: {wellness['score']}/100 ({wellness['level']})")

        # 11. Affirmation
        affirmation = random.choice(Config.AFFIRMATIONS)
        print(f"\n✨ TODAY'S AFFIRMATION:\n  \"{affirmation}\"")

    def interactive_menu(self):
        """Main interactive menu"""
        print("\n" + "="*70)
        print("💬 ULTIMATE MENTAL HEALTH COACH")
        print("="*70)
        print("""
🌟 15+ UNIQUE FEATURES:
  1. Advanced Sentiment Analysis
  2. Crisis Detection & Escalation
  3. Smart Coping Strategies
  4. Progress Tracking & Analytics
  5. LM Studio (Local AI)
  6. Gemini (Cloud AI)
  7. Auto Routing
  8. Emotion Wheel 🎯
  9. Personalization Engine 🎯
  10. Session Replay 🎯
  11. Wellness Score 🎯
  12. Export Reports 🎯
  13. Breathing Guide 🎯
  14. Mood Calendar 🎯
  15. AI Recommendations 🎯

COMMANDS:
  stress/anxiety/custom - Full analysis
  breathing             - Guided breathing
  wellness              - Check wellness score
  calendar              - View mood calendar
  profile               - See your profile
  replay                - View recent sessions
  export                - Export report
  all                   - Test all scenarios
  help                  - Show commands
  exit                  - Quit
""")

        while True:
            command = input("\n> ").strip().lower()

            if command == "exit":
                print("\n💙 Take care! 🌸\n")
                break

            elif command == "breathing":
                self.breathing_guide.guided_breathing()

            elif command == "wellness":
                trend = self.progress_tracker.get_mood_trend(7)
                wellness = self.wellness_score.calculate(trend)
                print(f"\n📊 WELLNESS SCORE: {wellness['score']}/100")
                print(f"Level: {wellness['level']}")

            elif command == "calendar":
                print(self.mood_calendar.get_mood_calendar(Config.DB_PATH))

            elif command == "profile":
                print(self.personalization.get_profile_summary())

            elif command == "replay":
                sessions = self.session_replay.get_recent_sessions(Config.DB_PATH)
                if sessions:
                    print("\n📹 RECENT SESSIONS:")
                    for s in sessions[:5]:
                        print(f"  • {s['timestamp'][:10]}: {s['emotion']} ({s['scenario']})")
                else:
                    print("No sessions yet")

            elif command == "export":
                result = self.report_exporter.export_sessions(Config.DB_PATH)
                print(f"\n{result}")

            elif command in Config.TEST_SCENARIOS or command == "custom":
                if command == "custom":
                    user_text = input("Your text: ").strip()
                else:
                    user_text = Config.TEST_SCENARIOS[command]

                if user_text:
                    self.comprehensive_analysis(command, user_text)

            elif command == "all":
                for scenario_name, user_input in Config.TEST_SCENARIOS.items():
                    self.comprehensive_analysis(scenario_name, user_input)

            else:
                print("❓ Unknown command. Try: stress, anxiety, breathing, wellness, profile, export, help, exit")

# ========================= MAIN =========================
if __name__ == "__main__":
    try:
        coach = UltimateHealthCoach()
        coach.interactive_menu()
    except KeyboardInterrupt:
        print("\n\n💙 Goodbye! 🌸\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
