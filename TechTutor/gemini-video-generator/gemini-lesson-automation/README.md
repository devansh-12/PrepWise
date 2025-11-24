# TechTutor - AI Lesson Video Generator

An automated system that generates professional educational videos for placement interview preparation using Google's Gemini AI. The system creates complete video lessons with AI-generated content, text-to-speech narration, visual slides, and background music.


## What This Project Does

This project automatically creates educational videos for technical interview preparation. Here's what happens:

1. **You choose a subject** (DBMS, OOP, Operating Systems, or Networking)
2. **AI generates 10 relevant topics** for that subject
3. **You pick a topic** you want to learn
4. **AI creates a complete video** with:
   - 5 educational slides with content
   - Professional voiceover narration
   - Visual backgrounds
   - Background music
   - Intro and outro slides

**Example Output:** A 3-5 minute video explaining "DBMS: Normalization and Keys (1NF–3NF)" ready to watch.

---

## 🔧 Prerequisites

Before you start, make sure you have these installed on your computer:

### 1. Python (Version 3.8 or higher)
- **Check if installed:** Open Command Prompt (CMD) and type:
  ```cmd
  python --version
  ```
### 2. FFmpeg (Required for video processing)
- **What it does:** Combines audio and images into video files
- **Windows Installation:**
  1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
  2. Extract the ZIP file to `C:\ffmpeg`
  3. Add to PATH:
     - Search "Environment Variables" in Windows
     - Edit "Path" under System Variables
     - Add `C:\ffmpeg\bin`
  4. Verify installation:
     ```cmd
     ffmpeg -version
     ```
### 3. Google Gemini API Key
- **What it does:** Allows the AI to generate lesson content
- **How to get it:**
  1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
  2. Sign in with your Google account
  3. Click "Create API Key"
---

## 📥 Installation Guide

Follow these steps carefully:

### Step 1: Download the Project
```cmd
git clone https://https://github.com/devansh-12/PrepWise/.git
cd /gemini-video-generator/gemini-lesson-automation
```
### Step 2: Create a Virtual Environment (Recommended)
This keeps your project dependencies separate from other Python projects.

```cmd
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**On Windows (CMD):**
```cmd
venv\Scripts\activate
```
You'll see `(venv)` appear at the start of your command line.

### Step 4: Install Required Packages
```cmd
pip install -r requirements.txt
```

This installs all necessary libraries:
- `google-generativeai` - For AI content generation
- `gTTS` - For text-to-speech
- `moviepy` - For video creation
- `Pillow` - For image processing
- `pydub` - For audio processing
- `edge-tts` - For high-quality voice synthesis

**Note:** This may take 2-5 minutes depending on your internet speed.
---

## ⚙️ Configuration

### Set Up Your API Key

1. Create the `.env` file in the project folder
2. Add your own:
   ```
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```
3. Save the file

## 🚀 How to Use

### Running the Program

1. **Open Command Prompt** in the project folder
2. **Activate virtual environment** (if not already active):
   ```cmd
   venv\Scripts\activate
   ```
3. **Run the program:**
   ```cmd
   python main.py
   ```

## 📁 Project Structure

```
gemini-lesson-automation/
│
├── main.py                  # Main program - START HERE
├── requirements.txt         # List of required Python packages
├── .env                     # Your API keys (KEEP SECRET!)
├── content_plan.json        # Tracks completed lessons
│
├── src/                     # Source code folder
│   ├── __init__.py          # Makes 'src' a Python package
│   └── generator.py         # Core functions (AI, TTS, video creation)
│
├── assets/                  # Resources for video creation
│   ├── fonts/               # Fonts for slide text
│   │   └── arial.ttf
│   ├── music/               # Background music
│   │   └── bg_music.mp3
│   └── fallback.jpg         # Default background image
│
├── output/                  # Generated videos appear here (organized by lesson)
│   └── 20251123_170222_Topic_Name/
│       ├── 1_content/       # AI-generated content (JSON)
│       ├── 2_audio/         # Individual slide audio files
│       ├── 3_slides/        # Individual slide images
│       └── 4_final/         # Final video
│
└── venv/                    # Virtual environment (created by you)
```

---

## Flow of the Feature

### 1. User Input (`main.py`)
- You select subject and topic
- Program creates a "lesson" object with your choices

### 2. Content Generation (`generator.py` → `generate_lesson_content()`)
- Sends request to Gemini AI
- AI creates 5 slides with:
  - Title
  - Introduction sentence
  - Main explanation (3-5 sentences)

### 3. Text-to-Speech (`generator.py` → `text_to_speech()`)
- Converts each slide's text to natural speech
- Uses Microsoft Edge TTS (Jenny Neural voice or Neerja Neural voice for Indian Dialect)
- Saves as MP3 files

### 4. Visual Creation (`generator.py` → `generate_visuals()`)
- Creates professional slide images (1920x1080)
- Adds title, content, and footer

### 5. Video Assembly (`generator.py` → `create_video()`)
- Combines slides and audio
- Adds background music (lowered volume)
- Adds fade-in/fade-out effects
- Exports as MP4 video

### 6. Output
- Final video saved in organized folder structure
- Each lesson gets its own timestamped folder

---



