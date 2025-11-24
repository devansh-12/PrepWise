# AI Interview Coach - Real-Time Performance Analysis System

A comprehensive real-time AI-powered interview coaching system that analyzes audio, video, posture, speech patterns, and behavioral indicators to provide immediate feedback during mock interview sessions. The system uses computer vision, audio signal processing, and rule-based evaluation to deliver objective, data-driven performance assessments.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Metrics Explained](#metrics-explained)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## 🎯 Overview

The AI Interview Coach is a **real-time** interview performance evaluation system that:

1. **Monitors** candidate behavior through webcam and microphone
2. **Analyzes** multiple performance indicators simultaneously
3. **Provides** instant visual feedback during the interview
4. **Generates** comprehensive performance reports with charts and metrics

The system runs **3 concurrent threads**:
- **Camera Monitor**: Processes video frames for face detection, eye contact, and posture
- **Audio Monitor**: Analyzes speech rate, volume, and pause detection
- **Main Thread**: Manages interview flow, questions, and timing

---

## ✨ Features

### 1. **Real-Time Camera Analysis**
- **Face Detection**: MediaPipe FaceMesh with support for up to 3 faces
- **Eye Contact Tracking**: Iris deviation analysis to measure gaze direction
- **Posture Evaluation**: Shoulder alignment assessment using MediaPipe Pose
- **Brightness Monitoring**: Detects dim lighting conditions
- **Multi-Face Detection**: Security feature to detect multiple people (terminates after 10 consecutive detections)
- **Confidence Scoring**: Real-time composite score combining audio and visual parameters

### 2. **Real-Time Audio Analysis**
- **Automatic Microphone Calibration**: 2-3 second calibration phase using RMS (Root Mean Square)
- **Volume Tracking**: Continuous RMS-based volume monitoring with smoothing
- **Speech Rate Estimation**: Librosa onset detection to measure speaking pace
- **Pause Detection**: Identifies sustained silent periods (≥3 seconds)
- **Adaptive Thresholds**: Dynamically adjusts based on calibrated baseline

### 3. **Live Feedback System**
Real-time on-screen advisories displayed for 1.5 seconds:
- "Speak louder" - Volume below threshold
- "Lower your volume" - Volume too high
- "You are speaking too fast" - Speech rate > 180 units
- "Try to speak faster" - Speech rate < 90 units
- "Maintain eye contact" - Eye contact score < 0.6
- "Sit upright" - Poor posture detected
- "Long pause detected" - Silence ≥ 3 seconds
- "Please switch on brighter lights" - Brightness < 60
- "Face not visible" - No face detected
- "X faces detected!" - Multiple people detected

### 4. **Performance Metrics Collection**
Continuous recording of:
- Confidence levels per frame
- Eye contact scores (0-1 scale)
- Posture values (0.6 or 1.0)
- Speech rate values (relative units)
- Volume levels (RMS values)
- Pause occurrences
- Time-series data for trend analysis

### 5. **System Reliability Metrics**
- **Detection Stability**: Percentage of frames with successful face, posture, and audio detection
- **Feedback Precision**: Proportion of session with confidence > 0.4
- **Reliability Score**: Weighted composite = (0.6 × Detection Stability) + (0.4 × Feedback Precision)

### 6. **Automated Reporting**
Generates 4 output files:
- `performance_summary.png` - Time-series charts
- `system_accuracy.png` - Reliability metrics bar chart
- `report.html` - Browser-viewable summary
- `report.pdf` - Formal PDF report (auto-opens)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Computer Vision** | OpenCV, MediaPipe (FaceMesh, Pose) |
| **Audio Processing** | SoundDevice, Librosa, NumPy |
| **Plotting** | Matplotlib |
| **PDF Generation** | ReportLab |
| **HTML Generation** | Native Python string formatting |
| **Parallel Execution** | Python `threading` module |
| **UI Overlay** | OpenCV drawing utilities |

---

## 📋 Prerequisites

### **System Requirements**
- **Python**: 3.7 or higher
- **Webcam**: Built-in or external camera
- **Microphone**: Built-in or external microphone
- **RAM**: 2GB+ recommended
- **OS**: Windows, macOS, or Linux

### **Python Packages**

All required packages are listed in `requirements.txt`:
```
opencv-python
mediapipe
sounddevice
numpy
librosa
matplotlib
reportlab
pillow
```

---

## 🔧 Installation & Setup

### **Step 1: Navigate to Project Directory**

```bash
cd /path/to/PrepWise/InterviewCoach
```

### **Step 2: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Or install manually**:
```bash
pip install opencv-python mediapipe sounddevice numpy librosa matplotlib reportlab pillow
```

### **Step 3: Verify Hardware**

**Test Webcam**:
```python
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Webcam detected")
cap.release()
```

**Test Microphone**:
```python
import sounddevice as sd
print(sd.query_devices())  # Should list available audio devices
```

---

## 🚀 Usage

### **Running the Interview Coach**

```bash
python main.py
```

### **Interview Flow**

1. **Welcome Screen**
   ```
   Welcome to the AI Interview Coach
   Press Enter to begin...
   ```

2. **Microphone Calibration** (2-3 seconds)
   ```
   🎤 Speak normally for 2–3 seconds to calibrate microphone volume...
   ✅ Mic calibrated. Base volume: 0.0234
   ```

3. **Interview Questions** (30 seconds each by default)
   ```
   Q1: Tell me about yourself.
   ⏱️ Time's up! Moving to next question...
   
   Q2: What are your strengths and weaknesses?
   ⏱️ Time's up! Moving to next question...
   
   Q3: Why should we hire you?
   ⏱️ Time's up! Moving to next question...
   ```

4. **Real-Time Feedback**
   - Live video feed with overlays
   - Current question displayed at top
   - Confidence bar at bottom (green > 75%, yellow > 50%, red < 50%)
   - Feedback messages in red text (top-left)
   - Press `Q` to quit early

5. **Report Generation**
   ```
   ✅ Saved 'performance_summary.png' with multi-metric trends.
   📊 Saved system accuracy chart as system_accuracy.png
   
   ----- SYSTEM RELIABILITY METRICS -----
   Detection Stability: 94.23%
   Feedback Precision: 87.56%
   Overall Reliability Score: 91.52%
   --------------------------------------
   
   ✅ Saved HTML report: report.html
   ✅ Saved PDF report: report.pdf
   📂 Automatically opened: /path/to/report.pdf
   ```

---

## 🔍 How It Works

### **System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                    Main Thread                          │
│  - Interview flow control                               │
│  - Question timing (30s each)                           │
│  - Report generation                                    │
└─────────────────────────────────────────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│   Camera Thread      │      │   Audio Thread       │
│  (monitor_camera)    │      │  (monitor_audio)     │
├──────────────────────┤      ├──────────────────────┤
│ • Face detection     │      │ • Mic calibration    │
│ • Eye contact        │      │ • Volume tracking    │
│ • Posture analysis   │      │ • Speech rate        │
│ • Brightness check   │      │ • Pause detection    │
│ • Confidence calc    │      │ • Feedback rules     │
└──────────────────────┘      └──────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
              ┌────────────────────┐
              │  Shared State      │
              ├────────────────────┤
              │ • audio_state      │
              │ • last_feedback    │
              │ • confidence_hist  │
              │ • camera_state     │
              └────────────────────┘
```

### **1. Microphone Calibration (`monitor_audio`)**

**Purpose**: Automatically adapt to different microphone sensitivities and environments.

**Process**:
1. Records 3 audio samples (1.5 seconds each)
2. Calculates RMS (Root Mean Square) for each sample
3. Computes average RMS as `mic_base_volume`
4. Sets adaptive thresholds:
   - `speak_thresh = mic_base_volume × 1.5` (minimum to count as speaking)
   - `loud_thresh = mic_base_volume × 6.0` (too loud warning)
   - `silence_thresh = mic_base_volume × 0.8` (below = silence)

**Example**:
```
Calibration samples: [0.0234, 0.0241, 0.0229]
mic_base_volume = 0.0235
speak_thresh = 0.0353
loud_thresh = 0.1410
silence_thresh = 0.0188
```

### **2. Audio Analysis Loop**

**Runs every 1.5 seconds**:

```python
# 1. Record audio window
audio = sd.rec(int(1.5 * 16000), samplerate=16000, channels=1)

# 2. Calculate RMS volume
rms = sqrt(mean(audio^2))
vol_smooth = 0.85 × vol_smooth + 0.15 × rms  # Exponential smoothing

# 3. Detect speaking state
is_speaking = vol_smooth > speak_thresh

# 4. Estimate speech rate (if speaking)
if is_speaking:
    onset_env = librosa.onset.onset_strength(audio, sr=16000)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env)
    rate = (len(onsets) / 1.5) × 40  # Scale to relative units
    rate_smooth = 0.7 × rate_smooth + 0.3 × rate

# 5. Detect long pauses
if vol_smooth < silence_thresh:
    if pause_counter == 0:
        pause_counter = current_time
    long_pause = (current_time - pause_counter) >= 3.0
else:
    pause_counter = 0
    long_pause = False

# 6. Generate feedback
if is_speaking:
    update_feedback("Speak louder", vol_smooth < speak_thresh × 1.1)
    update_feedback("Lower your volume", vol_smooth > loud_thresh)
    update_feedback("You are speaking too fast", rate_smooth > 180)
    update_feedback("Try to speak faster", rate_smooth < 90)
else:
    update_feedback("Long pause detected", long_pause)
```

### **3. Camera Analysis Loop (`monitor_camera`)**

**Runs at camera FPS (typically 30 FPS)**:

```python
# 1. Capture frame
ret, frame = cap.read()

# 2. Check brightness
brightness = mean(grayscale(frame))
if brightness < 60:
    update_feedback("Please switch on brighter lights", True)

# 3. Process with MediaPipe
rgb = cvtColor(frame, BGR2RGB)
face_results = mp_face.process(rgb)
pose_results = mp_pose.process(rgb)

# 4. Analyze posture
if pose_results.pose_landmarks:
    left_shoulder_y = pose_results.landmarks[LEFT_SHOULDER].y
    right_shoulder_y = pose_results.landmarks[RIGHT_SHOULDER].y
    tilt = abs(left_shoulder_y - right_shoulder_y)
    posture_val = 1.0 if tilt < 0.05 else 0.6
    update_feedback("Sit upright", posture_val < 1.0)

# 5. Analyze eye contact
if face_results.multi_face_landmarks:
    face_count = len(face_results.multi_face_landmarks)
    face = face_results.multi_face_landmarks[0]
    
    # Eye contact from iris position
    left_iris_x = face.landmark[33].x
    right_iris_x = face.landmark[263].x
    center_deviation = abs(0.5 - (left_iris_x + right_iris_x) / 2)
    eye_contact_val = max(0, 1 - center_deviation × 5)
    
    update_feedback("Maintain eye contact", eye_contact_val < 0.6)
    
    # Multi-face detection
    if face_count > 1:
        multi_face_count += 1
        update_feedback(f"{face_count} faces detected!", True)
        if multi_face_count > 10:
            print("Multiple people detected — interview terminated.")
            running = False

# 6. Calculate confidence score
vol_conf = 1.0 - abs(volume - 0.05) / 0.05  # Optimal volume = 0.05
rate_conf = 1.0 - abs(speech_rate - 120) / 120  # Optimal rate = 120
voice_conf = (vol_conf + rate_conf) / 2
confidence = clip(0.5×voice_conf + 0.3×eye_contact + 0.2×posture, 0, 1)

# 7. Draw UI overlay
# - Header bar with title
# - Current question
# - Confidence bar (green/yellow/red)
# - Feedback messages
```

### **4. Confidence Score Calculation**

**Formula**:
```
vol_conf = 1.0 - |volume - 0.05| / 0.05
rate_conf = 1.0 - |speech_rate - 120| / 120
voice_conf = (vol_conf + rate_conf) / 2

confidence = clip(
    0.5 × voice_conf + 
    0.3 × eye_contact + 
    0.2 × posture,
    0, 1
)
```

**Weights**:
- Voice (volume + rate): 50%
- Eye contact: 30%
- Posture: 20%

**Optimal Values**:
- Volume: 0.05 RMS
- Speech rate: 120 units
- Eye contact: 1.0 (centered gaze)
- Posture: 1.0 (aligned shoulders)

### **5. Feedback Display Logic**

```python
def update_feedback(tag, condition):
    now = time.time()
    if condition:
        last_feedback[tag] = now  # Add/update timestamp
    elif tag in last_feedback and now - last_feedback[tag] > 1.5:
        del last_feedback[tag]  # Remove if expired

def draw_feedbacks(frame):
    now = time.time()
    y = 80
    for tag, timestamp in last_feedback.items():
        if now - timestamp < 1.5:  # Show for 1.5 seconds
            cv2.putText(frame, tag, (30, y), FONT, 0.8, RED, 2)
            y += 35
```

---

## 📊 Output Files

### **1. performance_summary.png**

**Two-panel chart**:

**Panel 1: Confidence Over Time**
- X-axis: Time (seconds)
- Y-axis: Confidence (0-100%)
- Line plot showing confidence trend throughout interview

**Panel 2: Multi-Metric Trends**
- Speech Rate (blue line)
- Eye Contact % (orange line)
- Posture % (purple line)
- X-axis: Time steps
- Y-axis: Metric values

### **2. system_accuracy.png**

**Bar chart** with 3 metrics:
- Detection Stability (green)
- Feedback Precision (blue)
- Reliability Score (yellow)
- Y-axis: Percentage (0-100%)

### **3. report.html**

**Browser-viewable report** containing:
- Summary metrics table:
  - Average Confidence
  - Eye Contact maintained
  - Posture stability
  - Pauses Detected
  - Average Speech Rate
  - Average Volume Level
  - Detection Stability
  - Feedback Precision
  - Reliability Score
- Embedded PNG charts

**To view**: Open `report.html` in any web browser

### **4. report.pdf**

**Formal PDF report** with:
- Title: "Interview Performance Report"
- All metrics listed
- Embedded performance charts
- Professional formatting using ReportLab

**Auto-opens** after interview completion (platform-specific):
- Windows: `os.startfile()`
- macOS: `open` command
- Linux: `xdg-open` command

---

## ⚙️ Configuration

### **Interview Questions**

Edit the `QUESTIONS` list in `main.py`:

```python
QUESTIONS = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Why should we hire you?"
]
```

Add or remove questions as needed.

### **Question Duration**

Change the time per question (in seconds):

```python
QUESTION_DURATION = 30  # Default: 30 seconds
```

### **Audio Thresholds**

Thresholds are **automatically calibrated**, but you can adjust multipliers:

```python
# In monitor_audio() function:
speak_thresh = mic_base_volume * 1.5     # Speaking threshold
loud_thresh = mic_base_volume * 6.0      # Loud warning
silence_thresh = mic_base_volume * 0.8   # Silence threshold
```

### **Speech Rate Thresholds**

Adjust feedback triggers:

```python
# In monitor_audio() function:
update_feedback("You are speaking too fast", rate_smooth > 180)  # Default: 180
update_feedback("Try to speak faster", rate_smooth < 90)         # Default: 90
```

### **Eye Contact Threshold**

```python
# In monitor_camera() function:
update_feedback("Maintain eye contact", eye_val < 0.6)  # Default: 0.6
```

### **Brightness Threshold**

```python
# In monitor_camera() function:
if brightness < 60:  # Default: 60
    update_feedback("Please switch on brighter lights", True)
```

### **Multi-Face Termination**

```python
# In monitor_camera() function:
if multi_face_count > 10:  # Default: 10 consecutive detections
    print("Multiple people detected — interview terminated.")
    running = False
```

---

## 📈 Metrics Explained

### **Performance Metrics**

| Metric | Description | Range | Optimal |
|--------|-------------|-------|---------|
| **Average Confidence** | Overall performance score | 0-100% | >75% |
| **Eye Contact** | Percentage of time maintaining gaze | 0-100% | >80% |
| **Posture Stability** | Percentage of time with good posture | 0-100% | >90% |
| **Pauses Detected** | Percentage of time in long pauses | 0-100% | <10% |
| **Speech Rate** | Speaking pace (relative units) | 0-300 | 90-150 |
| **Volume Level** | Average RMS volume | 0.0-0.2 | 0.03-0.07 |

### **Reliability Metrics**

| Metric | Formula | Description |
|--------|---------|-------------|
| **Detection Stability** | `mean([eye_valid%, posture_valid%, audio_valid%])` | Percentage of frames with successful detection |
| **Feedback Precision** | `(frames_with_confidence>0.4 / total_frames) × 100` | Proportion of session with stable confidence |
| **Reliability Score** | `0.6 × Detection_Stability + 0.4 × Feedback_Precision` | Overall system reliability |

**Interpretation**:
- **>90%**: Excellent system performance
- **75-90%**: Good performance
- **<75%**: May indicate hardware/environment issues

---

## 🐛 Troubleshooting

### **Issue: Webcam not detected**

**Error**: `❌ Webcam not found.`

**Solutions**:
1. Check if webcam is connected
2. Test with: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
3. Try different camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
4. Grant camera permissions (macOS/Linux)

### **Issue: Microphone calibration fails**

**Error**: `⚠️ Mic calibration failed, using default thresholds.`

**Solutions**:
1. Check microphone connection
2. List devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
3. Set default device: `sd.default.device = 'Your Mic Name'`
4. Speak during calibration phase
5. Reduce background noise

### **Issue: No face detected**

**Feedback**: "Face not visible"

**Solutions**:
1. Ensure adequate lighting (brightness >60)
2. Position face in center of frame
3. Remove obstructions (masks, hands)
4. Adjust camera angle
5. Check MediaPipe installation: `pip install --upgrade mediapipe`

### **Issue: Poor eye contact scores**

**Feedback**: "Maintain eye contact"

**Solutions**:
1. Look directly at camera (not screen)
2. Position camera at eye level
3. Reduce head movement
4. Ensure good lighting on face

### **Issue: Posture warnings**

**Feedback**: "Sit upright"

**Solutions**:
1. Sit with shoulders level
2. Ensure full upper body visible
3. Avoid leaning to one side
4. Position camera to capture shoulders

### **Issue: Speech rate too fast/slow**

**Feedback**: "You are speaking too fast" / "Try to speak faster"

**Solutions**:
1. Practice speaking at moderate pace
2. Adjust thresholds in code if needed
3. Ensure clear microphone input
4. Reduce background noise

### **Issue: Volume warnings**

**Feedback**: "Speak louder" / "Lower your volume"

**Solutions**:
1. Adjust microphone distance (6-12 inches)
2. Speak during calibration phase
3. Check system volume settings
4. Use external microphone for better quality

### **Issue: Multiple faces detected**

**Feedback**: "X faces detected!"

**Solutions**:
1. Ensure only candidate is visible
2. Remove photos/posters with faces in background
3. Close video calls on other screens
4. Adjust camera angle to exclude others

### **Issue: Dim light warnings**

**Feedback**: "Please switch on brighter lights"

**Solutions**:
1. Increase room lighting
2. Use desk lamp facing candidate
3. Avoid backlighting (window behind)
4. Adjust brightness threshold in code if needed

### **Issue: PDF doesn't auto-open**

**Error**: `⚠️ Could not auto-open PDF: ...`

**Solutions**:
1. Manually open `report.pdf` from folder
2. Install PDF viewer (Windows: Edge/Adobe, macOS: Preview, Linux: Evince)
3. Check file permissions

### **Issue: Charts not generated**

**Error**: `No time-series data to plot.`

**Solutions**:
1. Ensure interview runs for at least 5 seconds
2. Check if data is being collected (print statements)
3. Verify matplotlib installation: `pip install --upgrade matplotlib`

---

## ⚠️ Limitations

### **1. Eye Contact Accuracy**
- **Depends on**: Lighting quality, camera positioning, head angle
- **Best practices**: Direct lighting on face, camera at eye level
- **Limitation**: Cannot distinguish between looking at camera vs. screen

### **2. Speech Rate Estimation**
- **Depends on**: Microphone quality, background noise
- **Best practices**: Use external mic, quiet environment
- **Limitation**: May spike with sudden loud sounds or music

### **3. Posture Detection**
- **Depends on**: Full upper body visibility, lighting
- **Best practices**: Sit 2-3 feet from camera, good lighting
- **Limitation**: Only detects shoulder tilt, not overall posture

### **4. Low-Light Performance**
- **Depends on**: Ambient lighting, camera sensor quality
- **Best practices**: Brightness >60 (grayscale mean)
- **Limitation**: Face/pose landmarks may fail in dim conditions

### **5. Background Noise**
- **Depends on**: Microphone quality, environment
- **Best practices**: Quiet room, close windows, mute notifications
- **Limitation**: High ambient noise affects pause detection and volume thresholds

### **6. Multi-Face Detection**
- **Depends on**: MediaPipe face detection accuracy
- **Best practices**: Clear background, no face photos/posters
- **Limitation**: May trigger false positives with face-like patterns

### **7. Microphone Calibration**
- **Depends on**: Speaking during calibration phase
- **Best practices**: Speak normally for 2-3 seconds
- **Limitation**: If silent during calibration, uses default thresholds

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the interview
python main.py

# 3. Follow prompts:
#    - Press Enter to begin
#    - Speak during calibration
#    - Answer questions (30s each)
#    - Press Q to quit early

# 4. Review outputs:
#    - performance_summary.png
#    - system_accuracy.png
#    - report.html
#    - report.pdf (auto-opens)
```

---

