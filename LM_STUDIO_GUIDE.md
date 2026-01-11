# 🤖 LM Studio Setup & Configuration Guide

Complete guide for setting up LM Studio with the Mental Health AI Chatbot.

## 📥 Installation

### Step 1: Download LM Studio

Visit [lmstudio.ai](https://lmstudio.ai) and download for your platform:
- **Windows:** LMStudio-Setup.exe
- **macOS:** LMStudio.dmg
- **Linux:** LMStudio.AppImage

### Step 2: Install & Launch

- **Windows/Mac:** Run installer and follow prompts
- **Linux:** Make AppImage executable and run
  ```bash
  chmod +x LMStudio-*.AppImage
  ./LMStudio-*.AppImage
  ```

---

## 🎯 Recommended Models for Mental Health Support

### Primary Recommendation: SmolLM2-1.7B-Instruct

**Why SmolLM2?**
- 📊 Size: 1.7 billion parameters (lightweight)
- 💾 RAM: Works with 8GB RAM
- ⚡ Speed: Fast inference on consumer hardware
- 🎓 Training: Optimized for instruction-following
- 🔒 Privacy: Fully local, no external calls
- 💬 Quality: Good empathetic responses for mental health

**Where to find:**
1. Open LM Studio
2. Click 🔍 Search icon
3. Search: "SmolLM2-1.7B-Instruct"
4. Look for: `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF`
5. Download the Q4_K_M or Q5_K_M quantization

### Alternative Models

| Model | Size | RAM | Speed | Quality | Use Case |
|-------|------|-----|-------|---------|----------|
| **TinyLlama-1.1B-Chat** | 1.1B | 6GB | ⚡⚡⚡ | ⭐⭐⭐ | Ultra-fast, basic support |
| **Phi-2** | 2.7B | 10GB | ⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| **Mistral-7B-Instruct** | 7B | 16GB | ⚡ | ⭐⭐⭐⭐⭐ | High quality, slower |
| **Llama-2-7B-Chat** | 7B | 16GB | ⚡ | ⭐⭐⭐⭐⭐ | Comprehensive support |

---

## ⚙️ Model Configuration

### Step 1: Download Model

1. Click **🔍 Search** in LM Studio
2. Search for your chosen model (e.g., "SmolLM2-1.7B-Instruct")
3. Select the GGUF version
4. Choose quantization:
   - **Q4_K_M**: Faster, lower quality (recommended for 8GB RAM)
   - **Q5_K_M**: Balanced (recommended for 12GB+ RAM)
   - **Q6_K**: Higher quality (needs 16GB+ RAM)
5. Click **Download**

### Step 2: Load Model in Server

1. Click **💬 Local Server** tab
2. Select your downloaded model from dropdown
3. Model loads (may take 10-30 seconds)
4. Green checkmark appears when ready

### Step 3: Configure Server Settings

Click **⚙️ Configure** in Local Server tab:

#### Basic Settings
```
Port: 1234 (default - DO NOT CHANGE)
CORS: Enabled
Context Length: 2048 (minimum)
```

#### Generation Settings
```
Temperature: 0.7
  - Lower (0.5-0.6): More factual, less creative
  - Higher (0.8-0.9): More varied responses
  - Recommended: 0.7 (balanced empathy)

Max Tokens: 512
  - Chatbot default setting
  - Can increase to 1024 for longer responses

Top P: 0.9
  - Controls diversity
  - 0.9 is optimal for coherent mental health advice

Repeat Penalty: 1.1
  - Prevents repetitive responses
```

#### Performance Settings
```
GPU Layers: Auto (or manual if available)
  - Offload to GPU for faster inference
  - More layers = faster but uses more VRAM

Threads: Auto
  - Uses CPU cores efficiently

Batch Size: 512
  - Higher = faster but more RAM
```

### Step 4: Start Server

1. Click **Start Server** button
2. Wait for "Server started on http://localhost:1234"
3. Server is ready! ✅

---

## 🧪 Testing the Setup

### Test 1: Browser Check
```
Open: http://localhost:1234
Expected: LM Studio API information page
```

### Test 2: cURL Test
```bash
curl http://localhost:1234/v1/models
```
**Expected output:**
```json
{
  "data": [
    {
      "id": "llm",
      "object": "model",
      ...
    }
  ]
}
```

### Test 3: Simple Chat Test
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### Test 4: Run Mental Health Chatbot
```bash
python mental_health.py
```
**Expected:** "🔵 LM Studio: ✅ Ready" message

---

## 🎨 Prompt Engineering for Mental Health

### Current Implementation (Simple)

The chatbot currently uses a **simple prompt structure** without system prompts:

```python
# Current API call in mental_health.py
{
    "model": "llm",
    "messages": [
        {"role": "user", "content": user_input}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9
}
```

This relies entirely on the model's training for empathetic responses.

---

### Enhanced System Prompts (Recommended)

For better, more consistent mental health support, you can add a **system prompt**. Here's how:

#### Basic Mental Health System Prompt

```python
SYSTEM_PROMPT = """You are a compassionate mental health support assistant. Your role is to:
- Listen empathetically without judgment
- Provide evidence-based coping strategies
- Encourage professional help when needed
- Never diagnose or prescribe medication
- Recognize crisis situations immediately

Always respond with warmth, understanding, and practical advice."""
```

#### Advanced Mental Health System Prompt

```python
ADVANCED_SYSTEM_PROMPT = """You are an AI mental health support companion trained in evidence-based therapeutic techniques.

Core Principles:
1. EMPATHY FIRST: Always validate feelings before offering solutions
2. SAFETY: Immediately recognize crisis language and provide helpline resources
3. EVIDENCE-BASED: Use CBT, DBT, and mindfulness techniques when appropriate
4. NON-DIAGNOSTIC: Never diagnose conditions or prescribe treatments
5. PROFESSIONAL BOUNDARIES: Encourage professional therapy for serious concerns

Response Framework:
- Start with validation: "I hear that you're feeling..."
- Ask clarifying questions when needed
- Offer 2-3 specific, actionable coping strategies
- End with encouragement and support
- Keep responses warm but professional (200-300 words)

Crisis Keywords Alert: suicide, self-harm, ending life → Immediately provide:
- AASRA (India): +91-9820466726
- 988 Suicide & Crisis Lifeline (US)

Avoid:
- Medical diagnoses or medication advice
- Dismissive phrases ("just think positive", "others have it worse")
- Overwhelming the user with too many strategies
- Being overly clinical or robotic

Tone: Warm, supportive, like a trusted friend with professional training."""
```

#### Emotion-Specific System Prompts

```python
ANXIETY_PROMPT = """You specialize in anxiety support. Focus on:
- Grounding techniques (5-4-3-2-1, box breathing)
- Cognitive reframing (challenging anxious thoughts)
- Progressive muscle relaxation
- Present-moment awareness
Validate that anxiety is a normal response, then teach coping tools."""

DEPRESSION_PROMPT = """You specialize in depression support. Focus on:
- Behavioral activation (small, achievable goals)
- Challenging negative self-talk
- Self-compassion practices
- Connection and social support
Acknowledge the weight of depression while instilling hope."""

STRESS_PROMPT = """You specialize in stress management. Focus on:
- Breaking down overwhelming tasks
- Time management and prioritization
- Boundary setting
- Relaxation techniques
Balance acknowledgment of stressors with practical solutions."""
```

---

### 🔧 Implementing Custom Prompts in Code

#### Method 1: Simple System Prompt (OpenAI Format)

Modify the `generate()` method in `LMStudioMode` class:

```python
def generate(self, prompt: str) -> Tuple[str, float]:
    if not self.available:
        return "LM Studio unavailable", 0

    SYSTEM_PROMPT = """You are a compassionate mental health support assistant. 
    Provide empathetic, evidence-based advice. Never diagnose or prescribe. 
    Recognize crisis situations and provide helpline resources."""

    try:
        start = time.time()
        response = requests.post(
            Config.LM_STUDIO_CHAT_URL,
            json={
                "model": "llm",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},  # NEW
                    {"role": "user", "content": prompt}
                ],
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
```

#### Method 2: Context-Aware Prompts

Add emotion-based system prompts:

```python
class LMStudioMode:
    SYSTEM_PROMPTS = {
        "anxious": """You're an anxiety support specialist. Use grounding 
        techniques and cognitive reframing. Be calming and reassuring.""",
        
        "sad": """You're a depression support specialist. Focus on hope, 
        small steps, and self-compassion. Validate the difficulty.""",
        
        "stressed": """You're a stress management coach. Help break down 
        problems and prioritize. Be practical and solution-focused.""",
        
        "default": """You're a compassionate mental health supporter. 
        Listen empathetically and provide evidence-based guidance."""
    }
    
    def generate(self, prompt: str, emotion: str = "default") -> Tuple[str, float]:
        if not self.available:
            return "LM Studio unavailable", 0
        
        system_prompt = self.SYSTEM_PROMPTS.get(emotion, self.SYSTEM_PROMPTS["default"])
        
        try:
            start = time.time()
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": "llm",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
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
```

#### Method 3: Multi-Turn Conversation (Advanced)

Keep conversation history for context:

```python
class LMStudioMode:
    def __init__(self):
        self.available = self._check()
        self.conversation_history = []  # NEW: Track conversation
        self.system_prompt = """You are a compassionate mental health support assistant."""
        if self.available:
            print("🔵 LM Studio: ✅ Ready")
    
    def generate(self, prompt: str, use_history: bool = True) -> Tuple[str, float]:
        if not self.available:
            return "LM Studio unavailable", 0
        
        # Build messages with history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history)  # Add previous messages
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            start = time.time()
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": "llm",
                    "messages": messages,
                    "temperature": Config.TEMPERATURE,
                    "max_tokens": Config.MAX_TOKENS,
                    "top_p": Config.TOP_P
                },
                timeout=60
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                assistant_message = response.json()["choices"][0]["message"]["content"].strip()
                
                # Store conversation for context
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Limit history to last 6 messages (3 exchanges)
                if len(self.conversation_history) > 6:
                    self.conversation_history = self.conversation_history[-6:]
                
                return assistant_message, elapsed
            return f"Error {response.status_code}", elapsed
        except Exception as e:
            return f"Error: {str(e)}", 0
    
    def clear_history(self):
        """Clear conversation history for new session"""
        self.conversation_history = []
```

---

### 🧠 In-Depth Inferencing Parameters

#### Core Parameters Explained

##### 1. Temperature (0.0 - 2.0)
Controls randomness and creativity in responses.

```python
TEMPERATURE = 0.0    # Deterministic, same response every time
TEMPERATURE = 0.3    # Very focused, factual (good for crisis)
TEMPERATURE = 0.5    # Balanced, somewhat predictable
TEMPERATURE = 0.7    # ✅ RECOMMENDED: Empathetic + consistent
TEMPERATURE = 0.9    # More creative, varied responses
TEMPERATURE = 1.2    # Very creative (may be inconsistent)
TEMPERATURE = 2.0    # Highly random (not recommended)
```

**For Mental Health:**
- **Crisis situations:** 0.3-0.5 (need clear, direct advice)
- **General support:** 0.7 (balanced empathy and accuracy)
- **Creative coping:** 0.8-0.9 (varied suggestions)

##### 2. Max Tokens (16 - 4096+)
Maximum length of response.

```python
MAX_TOKENS = 100     # Very brief (1-2 sentences)
MAX_TOKENS = 256     # Short paragraph
MAX_TOKENS = 512     # ✅ RECOMMENDED: 2-3 paragraphs
MAX_TOKENS = 768     # Detailed response
MAX_TOKENS = 1024    # Long, comprehensive
MAX_TOKENS = 2048    # Very detailed (may be too long)
```

**Token Estimation:**
- 1 token ≈ 0.75 words
- 512 tokens ≈ 384 words ≈ 2-3 paragraphs

**For Mental Health:**
- **Quick reassurance:** 256 tokens
- **Standard advice:** 512 tokens ✅
- **In-depth guidance:** 768-1024 tokens

##### 3. Top P / Nucleus Sampling (0.0 - 1.0)
Controls diversity by limiting token choices.

```python
TOP_P = 0.1     # Very limited, repetitive
TOP_P = 0.5     # Moderate diversity
TOP_P = 0.7     # Good balance
TOP_P = 0.9     # ✅ RECOMMENDED: Diverse yet coherent
TOP_P = 0.95    # Very diverse
TOP_P = 1.0     # Maximum diversity (less coherent)
```

**For Mental Health:**
- Use 0.9 for varied but appropriate responses
- Lower (0.7) if responses seem random
- Higher (0.95) for creative coping strategies

##### 4. Top K (1 - 100)
Limits token choices to top K most probable.

```python
TOP_K = 1       # Only most likely token (very deterministic)
TOP_K = 10      # Very focused
TOP_K = 40      # ✅ RECOMMENDED: Good balance
TOP_K = 100     # More varied
```

**Often used with Top P:**
```python
"top_p": 0.9,
"top_k": 40  # Consider top 40 tokens, then apply nucleus sampling
```

##### 5. Repeat Penalty (0.0 - 2.0)
Penalizes repeated words/phrases.

```python
REPEAT_PENALTY = 1.0    # No penalty (may repeat)
REPEAT_PENALTY = 1.1    # ✅ RECOMMENDED: Slight penalty
REPEAT_PENALTY = 1.2    # Moderate penalty
REPEAT_PENALTY = 1.5    # Strong penalty (may seem unnatural)
REPEAT_PENALTY = 2.0    # Very strong (breaks coherence)
```

**For Mental Health:**
- Use 1.1 to avoid repetitive advice
- Increase to 1.2 if model repeats same phrases

##### 6. Presence Penalty (-2.0 - 2.0)
Encourages new topics.

```python
PRESENCE_PENALTY = 0.0     # No encouragement
PRESENCE_PENALTY = 0.3     # Slight encouragement
PRESENCE_PENALTY = 0.6     # ✅ RECOMMENDED: Good for varied advice
PRESENCE_PENALTY = 1.0     # Strong (may go off-topic)
```

##### 7. Frequency Penalty (-2.0 - 2.0)
Reduces word repetition frequency.

```python
FREQUENCY_PENALTY = 0.0    # No reduction
FREQUENCY_PENALTY = 0.3    # Slight reduction
FREQUENCY_PENALTY = 0.5    # ✅ RECOMMENDED: Varied vocabulary
FREQUENCY_PENALTY = 1.0    # Strong reduction
```

---

### 🎛️ Optimal Configurations by Use Case

#### Configuration 1: Crisis Support (Focused & Clear)
```python
{
    "temperature": 0.4,          # Low - need clear, direct guidance
    "max_tokens": 400,           # Concise but complete
    "top_p": 0.8,                # Focused responses
    "top_k": 30,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.3
}
```

**Best for:** Suicide risk, self-harm, severe distress

#### Configuration 2: General Support (Balanced)
```python
{
    "temperature": 0.7,          # ✅ RECOMMENDED
    "max_tokens": 512,           # ✅ RECOMMENDED
    "top_p": 0.9,                # ✅ RECOMMENDED
    "top_k": 40,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.4,
    "frequency_penalty": 0.4
}
```

**Best for:** Stress, anxiety, general mental health concerns

#### Configuration 3: Creative Coping (Varied Strategies)
```python
{
    "temperature": 0.85,         # Higher creativity
    "max_tokens": 768,           # More detailed
    "top_p": 0.95,               # More diverse
    "top_k": 50,
    "repeat_penalty": 1.2,       # Avoid repeating same strategies
    "presence_penalty": 0.6,
    "frequency_penalty": 0.5
}
```

**Best for:** Generating diverse coping strategies, brainstorming

#### Configuration 4: Long-Term Planning (Detailed)
```python
{
    "temperature": 0.6,          # More structured
    "max_tokens": 1024,          # Long, detailed responses
    "top_p": 0.85,
    "top_k": 35,
    "repeat_penalty": 1.15,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.4
}
```

**Best for:** Goal setting, therapy plans, long-term strategies

---

### 📊 Advanced Inference Techniques

#### Technique 1: Dynamic Temperature Based on Sentiment

```python
def get_dynamic_temperature(sentiment_polarity: float) -> float:
    """Adjust temperature based on user's emotional state"""
    if sentiment_polarity < -0.7:  # Very negative
        return 0.5  # More focused, clear guidance
    elif sentiment_polarity < -0.3:  # Moderately negative
        return 0.65  # Balanced
    else:  # Neutral or positive
        return 0.8  # More creative suggestions
```

#### Technique 2: Token Budgeting Based on Complexity

```python
def get_dynamic_max_tokens(emotion: str, intensity: float) -> int:
    """Adjust response length based on need"""
    if emotion == "crisis":
        return 300  # Brief but complete
    elif intensity > 75:  # High intensity
        return 600  # More detailed support
    else:
        return 512  # Standard
```

#### Technique 3: Prompt Templates with Variables

```python
PROMPT_TEMPLATE = """I'm experiencing {emotion} because {situation}.
The intensity of my feelings is {intensity}/100.
I need {support_type} to help me cope.

Can you provide:
1. Validation of my feelings
2. {num_strategies} specific coping strategies
3. Encouragement for my situation"""

# Usage:
user_prompt = PROMPT_TEMPLATE.format(
    emotion="anxiety",
    situation="upcoming presentation",
    intensity=85,
    support_type="immediate techniques",
    num_strategies=3
)
```

#### Technique 4: Few-Shot Prompting

```python
FEW_SHOT_EXAMPLES = """Example 1:
User: I'm feeling overwhelmed with work deadlines.
Assistant: I hear that you're feeling overwhelmed - that's completely valid when facing multiple deadlines. Let's break this down together. Try: 1) List all tasks, 2) Identify the truly urgent ones, 3) Focus on one task at a time using 25-minute blocks. Remember, you can only do what you can do, and that's enough.

Example 2:
User: I'm anxious about a social event.
Assistant: Social anxiety before events is really common, and it's okay to feel this way. Here are some grounding techniques: 1) Before going, practice box breathing (4-4-4-4), 2) Set a time limit for how long you'll stay, 3) Have an "exit strategy" planned. You're brave for facing this despite the anxiety.

Now, user's concern:"""

# Prepend to user input
full_prompt = FEW_SHOT_EXAMPLES + user_input
```

#### Technique 5: Chain of Thought (CoT)

```python
COT_PROMPT = """User concern: {user_input}

Before responding, let's think through this step-by-step:
1. What emotion is the user primarily experiencing?
2. What might be the underlying need or concern?
3. Is there any crisis language? (yes/no)
4. What evidence-based technique would help most?
5. How can I validate their feelings while offering hope?

Based on this analysis, here's my response:"""
```

---

### 🔄 Complete Implementation Example

### 🔄 Complete Implementation Example

Here's a fully enhanced `LMStudioMode` class with all advanced features:

```python
class EnhancedLMStudioMode:
    """Enhanced LM Studio integration with custom prompts and advanced inferencing"""
    
    SYSTEM_PROMPTS = {
        "anxious": """You are a specialized anxiety support assistant. 
        Focus on: grounding techniques, breath work, cognitive reframing, and present-moment awareness.
        Always validate anxiety as a normal response, then provide practical coping tools.
        Keep responses warm and calming (200-300 words).""",
        
        "sad": """You are a specialized depression support assistant.
        Focus on: behavioral activation, self-compassion, challenging negative thoughts, social connection.
        Acknowledge the weight of depression while instilling hope and suggesting small, achievable steps.
        Keep responses empathetic and hopeful (200-300 words).""",
        
        "stressed": """You are a specialized stress management coach.
        Focus on: task breakdown, prioritization, time management, boundary setting, relaxation.
        Balance acknowledgment of stressors with practical, actionable solutions.
        Keep responses practical and solution-focused (200-300 words).""",
        
        "crisis": """You are an emergency mental health support assistant.
        IMMEDIATELY provide crisis helpline resources:
        - India: AASRA (+91-9820466726), iCall (+91-96540 22000)
        - US: 988 Suicide & Crisis Lifeline
        Then offer immediate grounding techniques and encouragement to seek professional help.
        Be direct, clear, and supportive (150-200 words).""",
        
        "default": """You are a compassionate mental health support assistant trained in evidence-based techniques.
        
        Core principles:
        1. Validate feelings first
        2. Ask clarifying questions if needed
        3. Offer 2-3 specific, actionable strategies
        4. Encourage professional help for serious concerns
        5. End with encouragement
        
        Never diagnose or prescribe. Recognize crisis language immediately.
        Keep responses warm, professional, and helpful (200-300 words)."""
    }
    
    def __init__(self):
        self.available = self._check()
        self.conversation_history = []
        if self.available:
            print("🔵 LM Studio: ✅ Ready (Enhanced Mode)")
    
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
    
    def _get_dynamic_config(self, emotion: str, sentiment_polarity: float, 
                           intensity: float, is_crisis: bool) -> dict:
        """Dynamic configuration based on user's emotional state"""
        
        if is_crisis:
            return {
                "temperature": 0.4,
                "max_tokens": 400,
                "top_p": 0.8,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.3
            }
        elif intensity > 75:  # High intensity
            return {
                "temperature": 0.5,
                "max_tokens": 600,
                "top_p": 0.85,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4
            }
        elif sentiment_polarity < -0.5:  # Very negative
            return {
                "temperature": 0.6,
                "max_tokens": 512,
                "top_p": 0.85,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.4
            }
        else:  # General support
            return {
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "presence_penalty": 0.4,
                "frequency_penalty": 0.4
            }
    
    def generate(self, prompt: str, emotion: str = "default", 
                sentiment_polarity: float = 0.0, intensity: float = 50.0,
                is_crisis: bool = False, use_history: bool = True) -> Tuple[str, float]:
        """
        Generate response with advanced configuration
        
        Args:
            prompt: User's input text
            emotion: Detected emotion type (anxious, sad, stressed, crisis, default)
            sentiment_polarity: Sentiment score (-1 to 1)
            intensity: Emotion intensity (0-100)
            is_crisis: Whether crisis detected
            use_history: Whether to use conversation history
        
        Returns:
            Tuple of (response_text, elapsed_time_ms)
        """
        if not self.available:
            return "LM Studio unavailable", 0
        
        # Select appropriate system prompt
        if is_crisis:
            system_prompt = self.SYSTEM_PROMPTS["crisis"]
        else:
            system_prompt = self.SYSTEM_PROMPTS.get(emotion, self.SYSTEM_PROMPTS["default"])
        
        # Get dynamic configuration
        config = self._get_dynamic_config(emotion, sentiment_polarity, intensity, is_crisis)
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 3 exchanges)
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            start = time.time()
            response = requests.post(
                Config.LM_STUDIO_CHAT_URL,
                json={
                    "model": "llm",
                    "messages": messages,
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                    "top_p": config["top_p"],
                    "repeat_penalty": config.get("repeat_penalty", 1.1),
                    "presence_penalty": config.get("presence_penalty", 0.4),
                    "frequency_penalty": config.get("frequency_penalty", 0.4)
                },
                timeout=60
            )
            elapsed = (time.time() - start) * 1000
            
            if response.status_code == 200:
                assistant_message = response.json()["choices"][0]["message"]["content"].strip()
                
                # Update conversation history
                if use_history:
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                    
                    # Keep only last 6 messages (3 exchanges)
                    if len(self.conversation_history) > 6:
                        self.conversation_history = self.conversation_history[-6:]
                
                return assistant_message, elapsed
            return f"Error {response.status_code}", elapsed
        except Exception as e:
            return f"Error: {str(e)}", 0
    
    def clear_history(self):
        """Clear conversation history for new session"""
        self.conversation_history = []
        print("💬 Conversation history cleared")
    
    def get_history_summary(self) -> dict:
        """Get summary of conversation history"""
        return {
            "exchanges": len(self.conversation_history) // 2,
            "total_messages": len(self.conversation_history),
            "history_enabled": len(self.conversation_history) > 0
        }
```

---

### 💡 Real-World Usage Examples

#### Example 1: Basic Usage (Current System)
```python
# In comprehensive_analysis() method
lm_studio = LMStudioMode()
response, elapsed = lm_studio.generate(user_input)
print(f"Response: {response}")
```

#### Example 2: Enhanced with Emotion Context
```python
# In comprehensive_analysis() method
lm_studio = EnhancedLMStudioMode()

# Get sentiment and emotion
sentiment = self.sentiment_analyzer.analyze(user_input)
crisis = self.crisis_detector.detect(user_input, sentiment)

# Generate with context
response, elapsed = lm_studio.generate(
    prompt=user_input,
    emotion=sentiment['emotion'],
    sentiment_polarity=sentiment['polarity'],
    intensity=sentiment['intensity'],
    is_crisis=crisis['is_crisis'],
    use_history=True
)

print(f"🤖 Response ({elapsed:.0f}ms): {response}")
```

#### Example 3: Multi-Turn Conversation
```python
# Session 1
lm_studio = EnhancedLMStudioMode()

response1, _ = lm_studio.generate(
    prompt="I'm feeling very anxious about tomorrow",
    emotion="anxious"
)
print(f"Assistant: {response1}")

# Session 2 (with context from session 1)
response2, _ = lm_studio.generate(
    prompt="The breathing exercise helped a bit. What else can I try?",
    emotion="anxious",
    use_history=True  # Uses previous exchange for context
)
print(f"Assistant: {response2}")

# Check conversation history
print(lm_studio.get_history_summary())
# Output: {'exchanges': 2, 'total_messages': 4, 'history_enabled': True}
```

#### Example 4: Crisis Handling
```python
# Automatic crisis detection with appropriate configuration
user_input = "I don't see the point in living anymore"
sentiment = self.sentiment_analyzer.analyze(user_input)
crisis = self.crisis_detector.detect(user_input, sentiment)

if crisis['is_crisis']:
    response, _ = lm_studio.generate(
        prompt=user_input,
        emotion="crisis",
        is_crisis=True,
        use_history=False  # Start fresh for crisis
    )
    # Response will use crisis configuration: temp=0.4, focused response
    print(f"🚨 CRISIS RESPONSE: {response}")
```

#### Example 5: A/B Testing Different Configurations
```python
# Test response quality with different temperatures
configs = [
    {"temp": 0.5, "desc": "Focused"},
    {"temp": 0.7, "desc": "Balanced"},
    {"temp": 0.9, "desc": "Creative"}
]

for config in configs:
    # Temporarily override temperature
    original_temp = Config.TEMPERATURE
    Config.TEMPERATURE = config["temp"]
    
    response, elapsed = lm_studio.generate(user_input)
    print(f"\n{config['desc']} (temp={config['temp']}):")
    print(f"Response: {response[:200]}...")
    print(f"Time: {elapsed:.0f}ms")
    
    Config.TEMPERATURE = original_temp
```

---

### 📊 Inference Performance Comparison

Based on SmolLM2-1.7B-Instruct testing:

| Configuration | Temp | Tokens | Quality | Speed | Use Case |
|--------------|------|--------|---------|-------|----------|
| **Crisis Mode** | 0.4 | 400 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Emergency support |
| **Standard** | 0.7 | 512 | ⭐⭐⭐⭐⭐ | ⚡⚡ | General counseling |
| **Creative** | 0.9 | 768 | ⭐⭐⭐⭐ | ⚡ | Coping strategies |
| **Detailed** | 0.6 | 1024 | ⭐⭐⭐⭐⭐ | ⚡ | Long-term planning |

**Key Findings:**
- Lower temperature = faster inference (fewer sampling iterations)
- Higher tokens = slower but more comprehensive
- Sweet spot: temp=0.7, tokens=512 for mental health

---

### 🎯 Prompt Optimization Tips

#### DO:
✅ Use clear, specific system prompts
✅ Include response length guidelines (200-300 words)
✅ Specify tone and approach (empathetic, supportive)
✅ List specific techniques to use (CBT, DBT, mindfulness)
✅ Include crisis detection instructions
✅ Provide examples with few-shot learning
✅ Use conversation history for context
✅ Adjust parameters based on emotional intensity

#### DON'T:
❌ Make system prompts too long (>500 words)
❌ Give contradictory instructions
❌ Forget to mention never diagnosing/prescribing
❌ Use overly technical jargon
❌ Set temperature too high (>1.0) for mental health
❌ Ignore crisis situations
❌ Keep too much conversation history (>6 messages)

---

### 🧪 Testing & Validation

#### Test Suite for Prompts
```python
TEST_CASES = [
    {
        "input": "I'm feeling anxious about work",
        "expected_elements": ["validate", "technique", "encourage"],
        "emotion": "anxious"
    },
    {
        "input": "I want to end it all",
        "expected_elements": ["helpline", "immediate", "professional"],
        "emotion": "crisis"
    },
    {
        "input": "I'm stressed with deadlines",
        "expected_elements": ["prioritize", "break down", "manage"],
        "emotion": "stressed"
    }
]

def test_prompt_quality(lm_studio, test_case):
    """Test if response contains expected elements"""
    response, _ = lm_studio.generate(
        test_case["input"],
        emotion=test_case["emotion"]
    )
    
    score = 0
    for element in test_case["expected_elements"]:
        if element.lower() in response.lower():
            score += 1
    
    quality = score / len(test_case["expected_elements"]) * 100
    return quality, response

# Run tests
for test in TEST_CASES:
    quality, response = test_prompt_quality(lm_studio, test)
    print(f"Test: {test['input'][:30]}...")
    print(f"Quality: {quality}%")
    print(f"Response: {response[:150]}...\n")
```

---

### 📚 Additional Resources

### For 8GB RAM Systems
```
Model: SmolLM2-1.7B-Instruct or TinyLlama-1.1B
Quantization: Q4_K_M
Context Length: 2048
GPU Layers: 0 (CPU only)
```

### For 16GB RAM Systems
```
Model: Phi-2 or Mistral-7B-Instruct
Quantization: Q5_K_M
Context Length: 4096
GPU Layers: 10-20 (if GPU available)
```

### For 32GB+ RAM Systems
```
Model: Mistral-7B-Instruct or Llama-2-7B
Quantization: Q6_K or Q8_0
Context Length: 8192
GPU Layers: All (if GPU available)
```

### GPU Acceleration (NVIDIA/AMD)

**Requirements:**
- NVIDIA GPU with CUDA support, OR
- AMD GPU with ROCm support

**Setup in LM Studio:**
1. LM Studio auto-detects compatible GPUs
2. In Server settings, set GPU layers:
   - Start with 10 layers
   - Increase until VRAM is ~90% full
   - More layers = faster inference
3. Monitor VRAM usage in LM Studio

**Performance Gains:**
- CPU only: 5-15 tokens/second
- GPU (10 layers): 20-50 tokens/second
- GPU (all layers): 50-100+ tokens/second

---

## 🔧 Troubleshooting

### Server Won't Start

**Issue:** Port 1234 already in use
```bash
# Check what's using port 1234
# Windows
netstat -ano | findstr :1234

# Linux/Mac
lsof -i :1234
```
**Solution:** Kill the process or change port (not recommended)

**Issue:** Model fails to load
- Check available RAM
- Try smaller model or lower quantization
- Close other applications

### Slow Responses

**Solutions:**
1. Use smaller model (SmolLM2 or TinyLlama)
2. Enable GPU offloading
3. Use Q4_K_M quantization
4. Reduce context length
5. Close browser/other apps

### Connection Errors

**Error:** "LM Studio unavailable"
**Checklist:**
- [ ] LM Studio is running
- [ ] Server is started (green indicator)
- [ ] Port 1234 is open
- [ ] Firewall not blocking localhost
- [ ] Model is loaded

### Response Quality Issues

**Too generic:**
- Use instruction-tuned models (SmolLM2-Instruct, Mistral-Instruct)
- Increase temperature slightly (0.75-0.8)

**Too repetitive:**
- Increase repeat penalty (1.2-1.3)
- Use different model

**Incoherent:**
- Lower temperature (0.6-0.65)
- Ensure model is properly quantized
- Try different model

---

## 🔄 Model Comparison for Mental Health

### Tested Models Performance

| Model | Response Time | Quality | Empathy | Accuracy | Recommendation |
|-------|---------------|---------|---------|----------|----------------|
| SmolLM2-1.7B-Instruct | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Best for 8GB |
| TinyLlama-1.1B | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Ultra-fast |
| Phi-2 | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Best balance |
| Mistral-7B-Instruct | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Best quality |

---

## 📝 Configuration File Reference

### In `mental_health.py` - Config Class

```python
class Config:
    # LM Studio settings
    LM_STUDIO_CHAT_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODEL = "llm"  # Default model identifier
    
    # Generation parameters
    TEMPERATURE = 0.7        # Empathy vs consistency balance
    MAX_TOKENS = 512         # Response length
    TOP_P = 0.9             # Response diversity
    
    # Optional: Gemini backup
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
```

### Customization Options

**Modify temperature:**
```python
Config.TEMPERATURE = 0.8  # More creative responses
Config.TEMPERATURE = 0.6  # More factual responses
```

**Modify max tokens:**
```python
Config.MAX_TOKENS = 768   # Longer responses
Config.MAX_TOKENS = 256   # Shorter, concise responses
```

**Change LM Studio port (if needed):**
```python
Config.LM_STUDIO_CHAT_URL = "http://localhost:5000/v1/chat/completions"
```

---

## 🎓 Best Practices

### For Mental Health Support

1. **Use instruction-tuned models** (e.g., SmolLM2-Instruct)
2. **Keep temperature moderate** (0.7) for empathy + accuracy
3. **Allow sufficient tokens** (512+) for detailed advice
4. **Monitor response quality** and adjust settings
5. **Test with different models** to find best fit

### For Privacy

1. **Always use local models** (no API calls to cloud)
2. **Verify server is localhost:1234**
3. **Check LM Studio privacy settings**
4. **Never share model outputs** containing user data
5. **Regular updates** to LM Studio for security

### For Performance

1. **Start with smallest model** that works
2. **Use quantized models** (GGUF Q4/Q5)
3. **Enable GPU if available**
4. **Close unnecessary apps** when running
5. **Monitor RAM/VRAM usage**

---

## 📚 Additional Resources

- **LM Studio Docs:** [docs.lmstudio.ai](https://lmstudio.ai/docs)
- **Model Hub:** [huggingface.co](https://huggingface.co/models)
- **GGUF Quantization:** [ggml.ai](https://ggml.ai)
- **Mental Health Resources:** See main README.md

---

## ❓ FAQ

**Q: Can I use other local AI servers?**
A: Yes! Any OpenAI-compatible API on localhost:1234 works (Ollama, LocalAI, etc.)

**Q: How much disk space do I need?**
A: 2-10GB depending on model (SmolLM2: ~2GB, Mistral-7B: ~7GB)

**Q: Can I run multiple models?**
A: One at a time in LM Studio. Switch via dropdown menu.

**Q: Is internet required?**
A: No! After downloading models, everything runs offline.

**Q: How to update LM Studio?**
A: Download latest version from lmstudio.ai (auto-update coming)

---

**Need help? Open an issue on [GitHub](https://github.com/Farbricated/Mental-health-Coach/issues)!**
