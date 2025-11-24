# PrepWise - AI-Powered Interview Preparation Platform

PrepWise is an intelligent interview preparation platform that leverages AI to provide realistic mock interviews, real-time feedback, and comprehensive analytics to help candidates excel in their job interviews.

---

## 🚀 Features

### 🎯 Core Modules

#### 1. **AI Mock Interviews**
- Create customized mock interviews based on job position, description, and experience level
- AI-generated interview questions tailored to specific roles
- Video recording of interview sessions for review

#### 2. **Real-Time Voice Interaction**
- **VAPI Integration**: Natural conversational AI for realistic interview experience
- **Speech-to-Text**: Multiple transcription options:
  - Deepgram API for high-accuracy audio transcription
  - Browser-based speech recognition as fallback
- **Text-to-Speech**: AI agent responds with natural voice synthesis

#### 3. **Intelligent Feedback System**
- AI-powered analysis of interview responses
- Detailed feedback on each answer including:
  - Rating (1-10 scale)
  - Strengths and areas for improvement
  - Suggested improvements
  - Comparison with ideal answers
- Overall interview performance analytics

#### 4. **Video Recording & Playback**
- Record complete interview sessions with webcam
- Question-wise video segmentation
- Upload and store videos using UploadThing
- Review recordings to improve presentation skills



#### 5. **Authentication & Authorization**
- Secure user authentication via Clerk
- Role-based access control
- Allowed users management system
- Email verification and password reset



---

## 🛠️ Tech Stack

### **Frontend**
- **Framework**: Next.js 15.5.0 (React 18.3.1)
- **Language**: TypeScript 5.8.3 & JavaScript
- **Styling**: Tailwind CSS 3.4.1
- **UI Components**: 
  - Radix UI (Dialog, Toast, Collapsible, Slot)
  - Custom components with class-variance-authority
  - Framer Motion for animations
  - GSAP for advanced animations
- **State Management**: React Hooks


### **Database**
- **ORM**: Drizzle ORM 0.33.0
- **Database**: PostgreSQL (Neon Serverless)
- **Schema Management**: Drizzle Kit 0.31.4

### **AI & ML Services**
- **Conversational AI**: VAPI AI (@vapi-ai/web)
- **LLM Integration**: 
  - Google Generative AI (Gemini)
  - AI SDK (@ai-sdk/google)
- **Speech-to-Text**: 
  - Deepgram SDK 4.11.2
  - Groq SDK 0.19.0
  - Browser Speech Recognition (react-hook-speech-to-text)

### **File Storage & Management**
- **File Uploads**: UploadThing 5.7.4
- **Cloud Storage**: Firebase Storage
- **Video Recording**: react-webcam 7.2.0

### **Authentication**
- **Auth Provider**: Clerk (@clerk/nextjs 6.31.3)
- **Session Management**: Clerk React



---

## 📋 Prerequisites

Before setting up PrepWise, ensure you have the following installed:

- **Node.js** (v18 or higher)
- **npm** or **yarn** package manager
- **Python** (v3.8 or higher)
- **pip** (Python package manager)
- **PostgreSQL** database (or Neon account)
- **Git**

---

## 🔧 Installation & Setup

### **Step 1: Clone the Repository**

```bash
git clone <repository-url>
cd PrepWise
```

### **Step 2: Install Root Dependencies**

```bash
npm install
```

This installs `concurrently` which allows running both frontend and backend simultaneously.

### **Step 3: Frontend Setup**

#### 3.1 Navigate to Frontend Directory
```bash
cd frontend
```

#### 3.2 Install Frontend Dependencies
```bash
npm install
```

#### 3.3 Configure Environment Variables

Create a `.env` file in the `frontend` directory with the following structure:

```env
# Clerk Authentication
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up

# Database (Neon PostgreSQL)
NEXT_PUBLIC_DRIZZLE_DB_URL=postgresql://user:password@host/database?sslmode=require&channel_binding=require

# AI Services
NEXT_PUBLIC_GEMINI_API_KEY=your_gemini_api_key
NEXT_PUBLIC_GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key

# VAPI AI
NEXT_PUBLIC_VAPI_WORKFLOW_ID=your_vapi_workflow_id
NEXT_PUBLIC_VAPI_WEB_TOKEN=your_vapi_web_token

# Firebase Configuration
NEXT_PUBLIC_FIREBASE_API_KEY=your_firebase_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your_measurement_id

# UploadThing
UPLOADTHING_TOKEN=your_uploadthing_token

# SMTP Configuration (for email notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
ADMIN_EMAIL=admin@example.com

# Application Configuration
NODE_ENV=development
NEXT_PUBLIC_INFO=Welcome to AI Interview
NEXT_PUBLIC_QUESTION_NOTE=Ensure your mic is working before starting
```

#### 3.4 Setup Database

**Push Database Schema:**
```bash
npm run db:push
```

**Open Drizzle Studio (Optional - for database management):**
```bash
npm run db:studio
```

**Generate Migrations (if needed):**
```bash
npm run db:generate
```

**Run Migrations (if needed):**
```bash
npm run db:migrate
```

### **Step 4: Backend Setup**

#### 4.1 Navigate to Backend Directory
```bash
cd ../backend
```

#### 4.2 Create Python Virtual Environment
```bash
python -m venv venv
```

#### 4.3 Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

#### 4.4 Install Python Dependencies
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

#### 4.5 Configure Backend Environment Variables

Create a `.env` file in the `backend` directory (if needed for future configurations):

```env
# Add backend-specific environment variables here
# Currently, the backend uses minimal configuration
```

---

## 🚀 Running the Application

### **Option 1: Run Both Frontend & Backend Together (Recommended)**

From the **root directory** (`PrepWise`):

```bash
npm run dev
```

This command uses `concurrently` to run:
- **Frontend**: `npm run dev` in the `frontend` directory (runs on `http://localhost:3000`)
- **Backend**: `uvicorn backend/app/main:app --reload` (runs on `http://localhost:8000`)

### **Option 2: Run Frontend & Backend Separately**

#### Run Frontend Only
```bash
cd frontend
npm run dev
```
Access at: `http://localhost:3000`

#### Run Backend Only
```bash
cd backend
uvicorn app.main:app --reload
```
Access at: `http://localhost:8000`

---

## 📁 Project Structure

<!-- Add your project structure image here -->
![Project Structure](/image.png)

---


## 🎬 How to Use PrepWise

### **1. Sign Up / Sign In**
- Create an account using Clerk authentication
- Verify your email address

### **2. Create a Mock Interview**
- Navigate to the Dashboard
- Click "Add New Interview"
- Fill in:
  - Job Position (e.g., "Software Engineer")
  - Job Description
  - Years of Experience
- Click "Generate Questions"
- AI will create customized interview questions

### **3. Start the Interview**
- Click "Start Interview" on your created interview
- Enable webcam and microphone permissions
- Click "Start Recording" to begin
- The AI agent will ask questions via voice
- Respond naturally using your microphone
- Your video and audio are recorded

### **4. Answer Questions**
- Listen to each question from the AI agent
- Speak your answer clearly
- The system transcribes your response using:
  - Deepgram API (primary)
  - Browser speech recognition (fallback)
- Move to the next question when ready

### **5. Review Feedback**
- After completing all questions, click "End Interview"
- Navigate to "Feedback" section
- View detailed AI-generated feedback for each answer:
  - Rating (1-10)
  - Strengths
  - Areas for improvement
  - Suggested better answers


---

## 🧪 Testing & Development

### **Run Frontend in Development Mode**
```bash
cd frontend
npm run dev
```

### **Build Frontend for Production**
```bash
cd frontend
npm run build
npm start
```

### **Lint Frontend Code**
```bash
cd frontend
npm run lint
```

### **Run Backend with Auto-Reload**
```bash
cd backend
uvicorn app.main:app --reload
```

---

## 🐛 Troubleshooting

### **Issue: Database Connection Error**
- Verify `NEXT_PUBLIC_DRIZZLE_DB_URL` is correct
- Ensure Neon database is active
- Check network connectivity
- Run `npm run db:push` to sync schema

### **Issue: Transcription Not Working**
- Verify `DEEPGRAM_API_KEY` is valid
- Check microphone permissions in browser
- Ensure audio is being captured
- Check browser console for errors

### **Issue: VAPI Agent Not Responding**
- Verify `NEXT_PUBLIC_VAPI_WORKFLOW_ID` and `NEXT_PUBLIC_VAPI_WEB_TOKEN`
- Check VAPI dashboard for workflow status
- Ensure internet connection is stable

### **Issue: File Upload Failing**
- Verify `UPLOADTHING_TOKEN` is correct
- Check file size limits
- Ensure proper file format (video/audio)

### **Issue: Backend Not Starting**
- Activate Python virtual environment
- Install all requirements: `pip install -r requirements.txt`
- Check Python version (>=3.8)
- Verify port 8000 is not in use

---

## 📝 API Endpoints

### **Frontend API Routes** (Next.js)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/interview` | POST | Create new mock interview |
| `/api/deepgram-transcribe` | POST | Transcribe audio using Deepgram |
| `/api/transcribe` | POST | Transcribe audio (fallback) |
| `/api/saveVideo` | POST | Save interview video |
| `/api/getResumes` | GET | Fetch user resumes |
| `/api/saveResume` | POST | Upload user resume |

### **Backend API Routes** (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check / Welcome message |
| `/api/*` | Various | Additional backend endpoints |

Access API documentation at: `http://localhost:8000/docs` (Swagger UI)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
