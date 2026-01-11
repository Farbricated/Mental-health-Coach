# 🧠 Ultimate Mental Health AI Chatbot

[![GitHub Repository](https://img.shields.io/badge/GitHub-Mental--health--Coach-blue?logo=github)](https://github.com/Farbricated/Mental-health-Coach)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**🔒 Privacy-First Mental Health Support** - An advanced AI-powered mental health support system with 15+ unique features including sentiment analysis, crisis detection, personalized coping strategies, and local AI processing via LM Studio.

**🔗 Repository:** https://github.com/Farbricated/Mental-health-Coach

## 🎯 Core Philosophy

**Privacy is paramount.** This chatbot is designed with a privacy-first approach:
- 🔒 **100% Local Processing** - All AI inference runs on your machine via LM Studio
- 💾 **Local Data Storage** - Your conversations stay on your device (SQLite database)
- 🚫 **No Cloud Dependency** - No data sent to external servers (Gemini is optional backup only)
- 🛡️ **Complete Control** - You own and control all your mental health data

## 📅 Project Updates

This is a **PCL group project** with continuous development:
- **Current Version:** v1.0
- **Branch Strategy:** Check repository branches for different versions and updates
- **Roadmap:** Check [Issues](https://github.com/Farbricated/Mental-health-Coach/issues) for upcoming features

**📖 See different branches in the repository for version history**

## ✨ Features

### Core Features
1. **Advanced Sentiment Analysis** - Real-time emotion detection and polarity scoring
2. **Crisis Detection System** - Multi-level severity detection with immediate support resources
3. **Smart Coping Strategies** - Personalized recommendations based on emotional state
4. **Progress Tracking** - SQLite-based analytics and mood trend analysis
5. **LM Studio Integration (Primary)** - Privacy-first local AI processing
6. **Auto-Intelligent Routing** - Automatic failover between AI models
7. **Gemini Integration (Optional Backup)** - Cloud AI as fallback only
8. **Emotion Wheel** - Visual representation of emotional states
9. **Personalization Engine** - Learns from user interactions
10. **Session Replay** - Review past conversations and insights
11. **Wellness Score Calculator** - Track overall mental health trends
12. **Report Exporter** - Export sessions to CSV/JSON for analysis
13. **Guided Breathing Exercises** - Interactive mindfulness tools
14. **Mood Calendar** - 7-day mood visualization
15. **AI Recommendations** - Context-aware suggestions
16. **Crisis Resources** - Immediate access to helplines (India & US)

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- **LM Studio** (Required) - Download from [lmstudio.ai](https://lmstudio.ai)
  - Must be running locally on port 1234
  - Recommended model: **SmolLM2-1.7B-Instruct** or similar small models
  - Other compatible models: Llama 2 7B, Mistral 7B, Phi-2, TinyLlama
- (Optional) Google Gemini API key for cloud backup

### LM Studio Setup Guide

1. **Download and Install LM Studio:**
   - Visit [lmstudio.ai](https://lmstudio.ai)
   - Download for your OS (Windows/Mac/Linux)
   - Install and launch LM Studio

2. **Download a Model:**
   - Click on "Search" (🔍) in LM Studio
   - Search for: **"SmolLM2-1.7B-Instruct"** (Recommended for mental health support)
   - Alternative models (if SmolLM not available):
     - `TinyLlama-1.1B-Chat` (lightweight)
     - `Phi-2-2.7B` (balanced)
     - `Mistral-7B-Instruct` (more capable, needs more RAM)
   - Click Download (choose GGUF format if multiple options)

3. **Load the Model:**
   - Go to "Local Server" tab (💬) in LM Studio
   - Select your downloaded model from dropdown
   - Click "Start Server"
   - Server will start on `http://localhost:1234`

4. **Verify Server is Running:**
   - Open browser and go to: `http://localhost:1234`
   - You should see LM Studio API page
   - Or test with: `curl http://localhost:1234/v1/models`

5. **Configure Model Settings (Optional):**
   - In LM Studio Server settings:
     - **Temperature:** 0.7 (default, good for empathetic responses)
     - **Max Tokens:** 512 (sufficient for mental health advice)
     - **Context Length:** 2048+ (for conversation history)
     - **GPU Offloading:** Enable if you have GPU (faster inference)

### Installation

1. **Set up LM Studio** (see Prerequisites above for detailed guide):
   - Install LM Studio
   - Download SmolLM2-1.7B-Instruct (or alternative model)
   - Start the local server on port 1234

2. Clone the repository:
```bash
git clone https://github.com/Farbricated/Mental-health-Coach.git
cd Mental-health-Coach
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Set up Gemini API as backup:
```bash
# Linux/Mac
export GEMINI_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"
```

5. Run the chatbot:
```bash
python mental_health.py
```

**Note:** To explore different versions, check out different branches in the repository.

## 💬 Commands

- `stress` / `anxiety` / `loneliness` etc. - Analyze predefined scenarios
- `custom` - Enter your own text for analysis
- `breathing` - Start guided breathing exercise
- `wellness` - Check your wellness score
- `calendar` - View 7-day mood calendar
- `profile` - See your personalization profile
- `replay` - Review recent sessions
- `export` - Export your data (CSV/JSON)
- `all` - Run all test scenarios
- `help` - Show available commands
- `exit` - Quit the application

## 🏗️ Architecture

```
mental_health.py
├── Configuration (Config class)
├── Sentiment Analysis Engine
├── Crisis Detection System
├── Coping Strategy Generator
├── Progress Tracker (SQLite - Local)
├── Personalization Engine
├── AI Integration Layer (Privacy-First)
│   ├── LM Studio Mode (PRIMARY - Local AI)
│   ├── Auto-Intelligent Mode (Local first, cloud fallback)
│   └── Gemini Mode (OPTIONAL - Cloud backup)
├── Wellness & Analytics (Local)
│   ├── Wellness Score Calculator
│   ├── Mood Calendar
│   ├── Session Replay
│   └── Report Exporter
└── Interactive Features
    ├── Emotion Wheel
    ├── Breathing Guide
    └── AI Recommendations
```

### Privacy Flow:
1. User input → Local sentiment analysis
2. LM Studio (local AI) → Response generation
3. All data stored locally in SQLite
4. No external API calls (unless Gemini backup enabled)

### LM Studio Inference Configuration

The chatbot uses the following inference settings for optimal mental health support:

**API Endpoint:**
```
POST http://localhost:1234/v1/chat/completions
```

**Request Parameters:**
```json
{
  "model": "llm",
  "messages": [
    {"role": "user", "content": "user's mental health concern"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9
}
```

**Why These Settings?**
- **Temperature 0.7:** Balanced creativity and consistency for empathetic responses
- **Max Tokens 512:** Sufficient for detailed mental health advice without overwhelming
- **Top P 0.9:** High quality, coherent responses

**System Prompt (Implicit):**
The chatbot relies on the model's training for empathetic mental health support. User input is directly sent with context about their emotional state (from sentiment analysis) to generate appropriate responses.

**Model Recommendation:**
- **SmolLM2-1.7B-Instruct:** Lightweight (1.7B parameters), fast inference, optimized for instruction-following
- **Why SmolLM?** Privacy-focused, runs on consumer hardware, low latency responses
- **RAM Requirement:** 4-6GB for model + 2GB for system = ~8GB total

## 📊 Database Schema

The application uses SQLite with the following structure:
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    scenario TEXT,
    mood TEXT,
    polarity REAL,
    intensity REAL,
    emotion TEXT,
    is_crisis INTEGER,
    model_used TEXT,
    response_time_ms REAL
);
```

## 🔐 Privacy & Safety

### Privacy-First Design
- **100% Local Processing**: All AI inference happens on your machine via LM Studio
- **Local-Only Storage**: All conversations stored locally in SQLite database on your device
- **Zero Cloud Dependency**: Works completely offline (Gemini is optional backup only)
- **No Data Sharing**: Your mental health data NEVER leaves your computer
- **Full Control**: You can delete, export, or backup your data anytime
- **Open Source**: Inspect the code to verify privacy claims

### Safety Features
- **Crisis Detection**: Immediate helpline resources for critical situations
- **Indian Helplines**: AASRA (+91-9820466726), iCall (+91-96540 22000), NIMHANS (+91-80-26995000)
- **US Helplines**: 988 Suicide & Crisis Lifeline, Crisis Text Line (741741)

### Why Local AI?
- **Privacy**: No one can access your conversations, not even us
- **Speed**: Faster responses with no network latency
- **Reliability**: Works without internet connection
- **Cost**: No API costs or subscription fees
- **Security**: No risk of data breaches from cloud services

## 🛠️ Configuration

Edit the `Config` class to customize:
- LM Studio URL and model (default: localhost:1234)
- Gemini API settings (optional backup)
- Temperature and generation parameters
- Database path
- Crisis keywords
- Affirmations library

## 🗓️ Development Roadmap

### Current Version: v1.0
**Core Features:**
- ✅ Local AI processing via LM Studio
- ✅ Advanced sentiment analysis
- ✅ Crisis detection system
- ✅ 15+ unique features
- ✅ Complete privacy-first architecture

### Planned Features (Future Versions)
- 🔄 Enhanced emotion detection algorithms
- 🔄 Multi-language support (Hindi, regional languages)
- 🔄 Voice input/output capabilities
- 🔄 Advanced analytics dashboard
- 🔄 Therapy technique modules (CBT, DBT)
- 🔄 Journal integration
- 🔄 Goal tracking system
- Mobile app version
- Secure peer support community
- Professional therapist integration tools
- Advanced pattern recognition
- Predictive wellness insights

**Contribute:** Have ideas for future versions? [Open an issue](https://github.com/Farbricated/Mental-health-Coach/issues)!

**Check different branches for development progress on new features.**

## 📈 Export Formats

The chatbot supports exporting your data in:
- **CSV**: For spreadsheet analysis
- **JSON**: For programmatic processing
- Exports include: timestamp, scenario, mood, polarity, emotion, and crisis flags

## 🔧 Troubleshooting

### LM Studio Not Connecting
1. Verify LM Studio is running and server is started
2. Check if port 1234 is available: `curl http://localhost:1234/v1/models`
3. Ensure a model is loaded in LM Studio (should show in Server tab)
4. Check firewall settings aren't blocking localhost:1234
5. Try restarting LM Studio server

### Model Loading Issues
- **Use lightweight models** for faster response:
  - ✅ SmolLM2-1.7B-Instruct (1.7B params) - **Recommended**
  - ✅ TinyLlama-1.1B-Chat (1.1B params) - Ultra-fast
  - ✅ Phi-2 (2.7B params) - Good balance
  - ⚠️ Mistral-7B (7B params) - Needs 16GB RAM
  - ⚠️ Llama-2-7B (7B params) - Needs 16GB RAM
- **RAM Requirements:**
  - 8GB RAM: SmolLM2-1.7B, TinyLlama-1.1B
  - 16GB RAM: Phi-2, Mistral-7B, Llama-2-7B
  - 32GB+ RAM: Any larger models
- **Use GGUF quantized models** for better performance
- **Enable GPU offloading** in LM Studio if you have a compatible GPU

### Response Quality Issues
- **Responses too generic?** Try these models with better instruction-following:
  - SmolLM2-1.7B-Instruct
  - Phi-2
  - Mistral-7B-Instruct-v0.2
- **Adjust temperature** in Config class (mental_health.py):
  - Lower (0.5-0.6): More consistent, factual
  - Higher (0.8-0.9): More creative, varied
- **Increase max_tokens** if responses are cut off (default: 512)

### Database Errors
- Check write permissions in the project directory
- Delete `mental_health_enterprise.db` to reset (loses history)
- Ensure SQLite3 is installed (usually comes with Python)

### Performance Optimization
- **Close other resource-intensive applications**
- **Use quantized models** (Q4_K_M or Q5_K_M variants in GGUF)
- **Enable GPU acceleration** in LM Studio settings (if available)
- **Reduce context length** if running out of memory
- **Use SSD** for model storage (faster loading)

## 🎯 Use Cases

- Personal mental health tracking
- Emotional awareness development
- Stress management
- Crisis prevention
- Pattern recognition in mood changes
- Progress monitoring over time

## ⚠️ Disclaimer

**This chatbot is NOT a replacement for professional mental health care.** It is designed as a supportive tool for self-reflection and emotional awareness. If you're experiencing a mental health crisis, please contact:

- **India**: AASRA (+91-9820466726), iCall (+91-96540 22000)
- **US**: 988 Suicide & Crisis Lifeline
- **Emergency**: Always call local emergency services (911/112)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built as a **PCL group project** with continuous development
- **LM Studio** for enabling privacy-first local AI processing
- Inspired by the need for **accessible and private** mental health support tools
- Special thanks to the open-source AI community
- Dedicated to everyone prioritizing mental health awareness

## 👥 Team

**Farbricated** - PCL Group Project Team
- Continuous development and updates
- Focus on privacy, accessibility, and innovation
- Open to collaboration and contributions

## 📧 Contact

For questions or feedback, please open an issue on [GitHub](https://github.com/Farbricated/Mental-health-Coach/issues).

**Repository:** https://github.com/Farbricated/Mental-health-Coach

---

**Made with 💙 for mental health awareness**
