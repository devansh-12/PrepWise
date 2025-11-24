# PrepWise - Comprehensive AI-Powered Career Preparation Platform

**PrepWise** is an integrated suite of AI-powered tools designed to help job seekers excel in their career preparation journey. From interview practice to skill gap analysis and personalized coaching, PrepWise provides end-to-end support for landing your dream job.

---

## 🎯 Platform Overview

PrepWise consists of **four main modules**, each addressing a critical aspect of career preparation:

| Module | Purpose | Key Technology |
|--------|---------|----------------|
| **🎤 Main Platform** | AI-powered mock interviews with real-time feedback | Next.js, VAPI AI, Deepgram |
| **📊 Skill Gap Analyser** | Resume analysis and skill gap identification | CareerBERT, FAISS, ESCO Taxonomy |
| **🎥 Interview Coach** | Real-time performance monitoring and coaching | MediaPipe, OpenCV, Librosa |
| **📚 TechTutor** | Personalized learning content generation | Gemini AI, Video Generation |

---

## 📦 Modules

### 1. 🎤 **Main Platform - AI Interview Preparation**

**Location**: [`/frontend`](./frontend) & [`/backend`](./backend)

The core PrepWise platform provides a complete interview preparation experience with AI-powered mock interviews, voice interaction, and intelligent feedback.

#### **Key Features**
- ✅ **AI Mock Interviews**: Customized questions based on job role, description, and experience
- 🎙️ **Voice Interaction**: Natural conversation with VAPI AI agents
- 📹 **Video Recording**: Record and review your interview performance
- 🤖 **Intelligent Feedback**: AI-generated ratings, strengths, and improvement suggestions
- 🔐 **Secure Authentication**: Clerk-based user management
- 💾 **Cloud Storage**: Firebase and UploadThing for media management

#### **Tech Stack**
- **Frontend**: Next.js 15.5, React 18.3, TypeScript, Tailwind CSS
- **Backend**: FastAPI (Python), PostgreSQL (Neon)
- **AI Services**: VAPI AI, Deepgram, Google Gemini, Groq
- **Database**: Drizzle ORM, Neon Serverless PostgreSQL

#### **Quick Start**
```bash
# Install dependencies
npm install
cd frontend && npm install

# Setup environment variables (see frontend/.env.example)
# Configure database
cd frontend && npm run db:push

# Run the platform
npm run dev  # Runs both frontend (3000) and backend (8000)
```

📖 **[View Detailed Documentation →](./frontend/README.md)**

---

### 2. 📊 **Skill Gap Analyser**

**Location**: [`/skillgap-analyser`](./skillgap-analyser)

An intelligent system that analyzes resumes to extract skills, map them to the ESCO taxonomy, identify suitable occupations, and perform comprehensive skill gap analysis.

#### **Key Features**
- 🔍 **Hybrid Skill Detection**: PhraseMatcher + Fuzzy Matching + Semantic Search
- 🗂️ **ESCO Integration**: 13,000+ standardized skills, 3,000+ occupations
- 🎯 **Occupation Matching**: Find best-fit roles based on your skills
- 📈 **Gap Analysis**: Identify strengths and missing skills for target roles
- 🧠 **CareerBERT**: Domain-specific AI model for career/skill embeddings
- ⚡ **Fast Search**: FAISS vector similarity search

#### **Tech Stack**
- **ML/AI**: CareerBERT (sentence-transformers), FAISS
- **NLP**: spaCy, RapidFuzz
- **Data**: ESCO Taxonomy, pandas, NumPy
- **Processing**: PyPDF2, python-docx

#### **Quick Start**
```bash
cd skillgap-analyser

# Install dependencies
pip install sentence-transformers faiss-cpu pandas numpy spacy rapidfuzz PyPDF2 python-docx
python -m spacy download en_core_web_md

# Download ESCO data (see README for details)
# Build index (one-time setup, ~10-15 min)
python app/build_esco_index.py

# Analyze a resume
python app/parse_resume.py data/resumes/your_resume.pdf
```

📖 **[View Detailed Documentation →](./skillgap-analyser/README.md)**  

---

### 3. 🎥 **Interview Coach - Real-Time Performance Monitoring**

**Location**: [`/InterviewCoach`](./InterviewCoach)

A real-time AI coaching system that monitors audio, video, posture, speech patterns, and behavioral indicators during mock interviews, providing instant feedback and comprehensive performance reports.

#### **Key Features**
- 👁️ **Face Detection**: MediaPipe FaceMesh for eye contact tracking
- 🧍 **Posture Analysis**: Shoulder alignment assessment
- 🎤 **Audio Analysis**: Volume, speech rate, pause detection with auto-calibration
- 💡 **Live Feedback**: Real-time on-screen advisories (11 feedback types)
- 📊 **Performance Reports**: Automated PDF/HTML reports with charts
- 🔒 **Security**: Multi-face detection to prevent cheating

#### **Tech Stack**
- **Computer Vision**: OpenCV, MediaPipe (FaceMesh, Pose)
- **Audio Processing**: SoundDevice, Librosa, NumPy
- **Reporting**: Matplotlib, ReportLab
- **Concurrency**: Python threading

#### **Quick Start**
```bash
cd InterviewCoach

# Install dependencies
pip install -r requirements.txt

# Run interview coach
python main.py

# Follow prompts:
# 1. Press Enter to begin
# 2. Speak during mic calibration (2-3 sec)
# 3. Answer questions (30 sec each)
# 4. Press Q to quit early

# Outputs: performance_summary.png, system_accuracy.png, report.html, report.pdf
```

📖 **[View Detailed Documentation →](./InterviewCoach/README.md)**

---

### 4. 📚 **TechTutor - Personalized Learning Content**

**Location**: [`/TechTutor`](./TechTutor)

An AI-powered educational content generation system that creates personalized learning materials, video lessons, and tutorials using Gemini AI.

#### **Key Features**
- 🎬 **Video Generation**: Automated lesson video creation
- 📝 **Content Personalization**: Tailored to individual learning needs
- 🤖 **Gemini AI**: Advanced content generation
- 📚 **Curriculum Design**: Structured learning paths

#### **Tech Stack**
- **AI**: Google Gemini API
- **Video**: Automated video generation tools
- **Content**: Markdown, HTML generation

#### **Quick Start**
```bash
cd TechTutor/gemini-video-generator/gemini-lesson-automation

# Install dependencies and configure
# (See module documentation for details)
```

📖 **[View Module Directory →](./TechTutor)**

---

## 🚀 Getting Started

### **Prerequisites**

- **Node.js** 18+ and npm
- **Python** 3.8+
- **PostgreSQL** (or Neon account)
- **Git**

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PrepWise
   ```

2. **Install root dependencies**
   ```bash
   npm install
   ```

3. **Choose your module(s)**

   - **Main Platform**: Follow [frontend/README.md](./frontend/README.md)
   - **Skill Gap Analyser**: Follow [skillgap-analyser/README.md](./skillgap-analyser/README.md)
   - **Interview Coach**: Follow [InterviewCoach/README.md](./InterviewCoach/README.md)
   - **TechTutor**: Follow [TechTutor/README.md](./TechTutor/README.md)

---

## 🎯 Use Cases

### **For Job Seekers**
1. **Practice Interviews** → Use Main Platform for realistic mock interviews
2. **Analyze Skills** → Use Skill Gap Analyser to identify missing skills
3. **Improve Performance** → Use Interview Coach for real-time feedback
4. **Learn New Skills** → Use TechTutor for personalized learning

### **For Recruiters/HR**
1. **Candidate Assessment** → Evaluate interview performance objectively
2. **Skill Verification** → Validate candidate skills against job requirements
3. **Training Programs** → Use Interview Coach for employee development

### **For Educational Institutions**
1. **Career Preparation** → Comprehensive interview training
2. **Skill Development** → Identify and bridge skill gaps
3. **Performance Tracking** → Monitor student progress

---

## 🛠️ Technology Stack Summary

### **Frontend Technologies**
- Next.js 15.5, React 18.3, TypeScript 5.8
- Tailwind CSS, Radix UI, Framer Motion
- Clerk (Auth), Drizzle ORM, Neon PostgreSQL

### **Backend Technologies**
- FastAPI (Python), Uvicorn
- PostgreSQL, Drizzle ORM

### **AI & ML**
- **Voice AI**: VAPI AI, Deepgram, Groq
- **LLM**: Google Gemini
- **ML Models**: CareerBERT (sentence-transformers)
- **Computer Vision**: MediaPipe, OpenCV
- **Audio**: Librosa, SoundDevice
- **Search**: FAISS vector similarity

### **Data & Processing**
- ESCO Taxonomy (13,000+ skills, 3,000+ occupations)
- spaCy, RapidFuzz, pandas, NumPy
- PyPDF2, python-docx

### **Storage & Media**
- Firebase Storage, UploadThing
- Video: react-webcam, MediaRecorder API

---


## 🔑 Environment Variables

Each module requires specific environment variables. See individual README files:

- **Main Platform**: [frontend/README.md](./frontend/README.md#configuration)
- **Skill Gap Analyser**: No API keys required (uses local models)
- **Interview Coach**: No configuration needed
- **TechTutor**: See module documentation

---


## 📚 Documentation Index

- **Main Platform**: [frontend/README.md](./frontend/README.md)
- **Skill Gap Analyser**: [skillgap-analyser/README.md](./skillgap-analyser/README.md)
- **Skill Gap Workflow**: [skillgap-analyser/WORKFLOW.md](./skillgap-analyser/WORKFLOW.md)
- **Interview Coach**: [InterviewCoach/README.md](./InterviewCoach/README.md)
- **TechTutor**: [TechTutor/](./TechTutor)


