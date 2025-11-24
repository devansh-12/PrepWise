
import cv2, mediapipe as mp, sounddevice as sd, numpy as np, threading, time, librosa, warnings, matplotlib.pyplot as plt, os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
QUESTIONS = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Why should we hire you?"
]
QUESTION_DURATION = 30
running = True
multi_face_count = 0
current_question = ""

# Global tracking
audio_state = {"volume": 0.0, "speech_rate": 120.0, "long_pause": False}
last_feedback = {}
confidence_history, time_history = [], []
camera_state = {
    "dim_light": False,
    "last_light_warning": 0
}

# Mic calibration globals
mic_calibrated = False
mic_base_volume = 0.02  # default fallback



# Stats tracking
eye_contact_values, posture_values, pause_flags, speech_rates, volumes = [], [], [], [], []

mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=3)
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

def get_frame_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean()

# ---------------- AUDIO MONITOR ----------------
def monitor_audio():
    global mic_calibrated, mic_base_volume
    sr = 16000
    window = 1.5
    pause_counter = 0
    vol_smooth, rate_smooth = 0.03, 120.0
    calibration_samples = []

    print("\n🎤 Speak normally for 2–3 seconds to calibrate microphone volume...")

    # --- Calibration phase (2 sec) ---
    for _ in range(3):
        audio = sd.rec(int(window * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        y = np.squeeze(audio)
        if len(y) == 0:
            continue
        calibration_samples.append(float(np.sqrt(np.mean(y ** 2))))

    if calibration_samples:
        mic_base_volume = np.mean(calibration_samples)
        mic_calibrated = True
        print(f"✅ Mic calibrated. Base volume: {mic_base_volume:.4f}")
    else:
        print("⚠️ Mic calibration failed, using default thresholds.")

    # Adaptive thresholds
    speak_thresh = mic_base_volume * 1.5     # must exceed this to count as speaking
    loud_thresh = mic_base_volume * 6.0      # dynamic loud warning
    silence_thresh = mic_base_volume * 0.8   # below = silence

    # --- Main loop ---
    while running:
        audio = sd.rec(int(window * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        y = np.squeeze(audio)
        if len(y) == 0:
            continue

        rms = float(np.sqrt(np.mean(y ** 2)))
        vol_smooth = 0.85 * vol_smooth + 0.15 * rms
        volumes.append(vol_smooth)
        audio_state["volume"] = vol_smooth

        # Determine speaking state
        is_speaking = vol_smooth > speak_thresh

        # Speech rate logic
        if is_speaking:
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                rate = (len(onsets) / window) * 40
                if rate > 350:  # ignore unrealistic spikes
                    rate = rate_smooth
            except:
                rate = rate_smooth
        else:
            rate = 0  # not speaking → no speech rate
        rate_smooth = 0.7 * rate_smooth + 0.3 * rate
        speech_rates.append(rate_smooth)
        audio_state["speech_rate"] = rate_smooth

        # Pause detection
        # Time-based pause detection
        if vol_smooth < silence_thresh:
            if pause_counter == 0:
                pause_counter = time.time()
            long_pause = (time.time() - pause_counter) >= 3.0
        else:
            pause_counter = 0
            long_pause = False


        # --- FEEDBACK (safety rules) ---
        if is_speaking:  # only trigger if user is actually speaking
            update_feedback("Speak louder", vol_smooth < speak_thresh * 1.1)
            update_feedback("Lower your volume", vol_smooth > loud_thresh)
            update_feedback("You are speaking too fast", rate_smooth > 180)
            update_feedback("Try to speak faster", rate_smooth < 90 and not long_pause)

        else:  # user is silent
            update_feedback("Long pause detected", long_pause)

        time.sleep(0.5)



# ---------------- HELPERS ----------------
def update_feedback(tag, condition):
    now = time.time()
    if condition:
        last_feedback[tag] = now
    elif tag in last_feedback and now - last_feedback[tag] > 1.5:
        del last_feedback[tag]

def draw_feedbacks(frame):
    now = time.time()
    y0 = 80
    for tag, ts in list(last_feedback.items()):
        if now - ts < 1.5:
            cv2.putText(frame, tag, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y0 += 35
        else:
            del last_feedback[tag]

def analyze_posture(pose_res):
    if not pose_res.pose_landmarks:
        return 0.6, "Sit upright"
    lms = pose_res.pose_landmarks.landmark
    l, r = lms[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y, \
           lms[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
    tilt = abs(l - r)
    val = 1.0 if tilt < 0.05 else 0.6
    posture_values.append(val)
    return val, "" if val == 1.0 else "Sit upright"

def get_eye_contact(face):
    lx, rx = face[33].x, face[263].x
    dev = abs(0.5 - (lx + rx) / 2)
    val = max(0, 1 - dev * 5)
    eye_contact_values.append(val)
    return val


# ---------------- CAMERA MONITOR ----------------
def monitor_camera():
    global running, multi_face_count, current_question,camera_state
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found.")
        running = False
        return

    start_time = time.time()
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        brightness = get_frame_brightness(frame) or 0  # fallback

        # threshold depends on your lighting setup
        if brightness < 60:
            if time.time() - camera_state.get("last_light_warning", 0) > 5:
                update_feedback("Please switch on brighter lights", True)
                camera_state["last_light_warning"] = time.time()
            camera_state["dim_light"] = True
        else:
            camera_state["dim_light"] = False


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_res = mp_face.process(rgb)
        pose_res = mp_pose.process(rgb)
        h, w, _ = frame.shape


        # Posture
        posture_val, posture_msg = analyze_posture(pose_res)
        update_feedback(posture_msg, bool(posture_msg))

        # Eye contact + face count
        face_count, eye_val = 0, 0.7
        if face_res.multi_face_landmarks:
            face_count = len(face_res.multi_face_landmarks)
            f0 = face_res.multi_face_landmarks[0]
            eye_val = get_eye_contact(f0.landmark)

        if face_count > 1:
            multi_face_count += 1
            update_feedback(f"{face_count} faces detected!", True)
            if multi_face_count > 10:
                print("\nMultiple people detected repeatedly — interview terminated.")
                running = False
                break
        else:
            multi_face_count = 0

        update_feedback("Maintain eye contact", eye_val < 0.6)
        update_feedback("Face not visible", face_count == 0)

        # Audio feedback
        # Audio feedback
        vol, rate, pause = audio_state["volume"], audio_state["speech_rate"], audio_state["long_pause"]

        # Speaking checks (only if volume indicates speech)
        if vol > 0.015:
            update_feedback("Speak louder", vol < 0.02)
            update_feedback("Lower your volume", vol > 0.09)
            update_feedback("You are speaking too fast", rate > 180)
            update_feedback("Try to speak faster", rate < 90)
        else:
            # Long pause check (ONLY when silent)
            if pause:
                update_feedback("Long pause detected", True)

        # Confidence (voice + posture + eye contact)
        vol_conf = 1.0 - abs(vol - 0.05) / 0.05
        vol_conf = np.clip(vol_conf, 0, 1)
        rate_conf = 1.0 - abs(rate - 120) / 120
        rate_conf = np.clip(rate_conf, 0, 1)
        voice_conf = (vol_conf + rate_conf) / 2
        conf = np.clip((0.5 * voice_conf + 0.3 * eye_val + 0.2 * posture_val), 0, 1)

        confidence_history.append(conf)
        time_history.append(time.time() - start_time)

        # -------- UI Overlay --------
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, "AI Interview Coach - Press Q to quit", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display current question
        if current_question:
            cv2.putText(frame, f"Q: {current_question}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Confidence bar
        bar_color = (0, 255, 0) if conf > 0.75 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
        bar_len = int(conf * w * 0.4)
        cv2.rectangle(frame, (30, h - 40), (30 + int(w * 0.4), h - 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (30, h - 40), (30 + bar_len, h - 20), bar_color, -1)
        cv2.putText(frame, f"Confidence {int(conf * 100)}%", (30, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

        draw_feedbacks(frame)
        cv2.imshow("Interview Coach", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- PERFORMANCE + RELIABILITY CHARTS ----------------
def plot_summary_charts():
    # Confidence chart
    if len(time_history) == 0:
        print("No time-series data to plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_history, np.array(confidence_history) * 100, color='green', linewidth=2)
    plt.title("Confidence Over Time")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)

    # Multi-metric trends
    plt.subplot(2, 1, 2)
    # Make arrays same length for plotting: use min length
    minlen = min(len(speech_rates), len(eye_contact_values), len(posture_values))
    if minlen > 0:
        plt.plot(speech_rates[:minlen], label="Speech Rate", color='blue')
        plt.plot(np.array(eye_contact_values[:minlen]) * 100, label="Eye Contact (%)", color='orange')
        plt.plot(np.array(posture_values[:minlen]) * 100, label="Posture (%)", color='purple')
        plt.legend()
        plt.xlabel("Time steps")
        plt.ylabel("Metric value")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("performance_summary.png")
    print("✅ Saved 'performance_summary.png' with multi-metric trends.")


def compute_reliability_metrics():
    total_frames = max(1, len(confidence_history))
    valid_eyes = np.count_nonzero(np.array(eye_contact_values) > 0) if eye_contact_values else 0
    valid_posture = np.count_nonzero(np.array(posture_values) > 0) if posture_values else 0
    valid_audio = np.count_nonzero(np.array(volumes) > 0) if volumes else 0
    stable_feedback = np.count_nonzero(np.array(confidence_history) > 0.4)

    detection_stability = np.mean([
        (valid_eyes / total_frames) if total_frames else 0,
        (valid_posture / total_frames) if total_frames else 0,
        (valid_audio / total_frames) if total_frames else 0
    ]) * 100
    feedback_precision = (stable_feedback / total_frames) * 100
    reliability_score = (0.6 * detection_stability + 0.4 * feedback_precision)

    # Plot reliability chart
    plt.figure(figsize=(6, 4))
    labels = ["Detection Stability", "Feedback Precision", "Reliability Score"]
    values = [detection_stability, feedback_precision, reliability_score]
    colors = ["#4CAF50", "#2196F3", "#FFC107"]
    plt.bar(labels, values, color=colors)
    plt.ylim(0, 100)
    plt.title("System Accuracy & Reliability Metrics")
    plt.ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig("system_accuracy.png")
    print("📊 Saved system accuracy chart as system_accuracy.png")

    print("\n----- SYSTEM RELIABILITY METRICS -----")
    print(f"Detection Stability: {detection_stability:.2f}%")
    print(f"Feedback Precision: {feedback_precision:.2f}%")
    print(f"Overall Reliability Score: {reliability_score:.2f}%")
    print("--------------------------------------")

    # Return summary numbers for report
    return {
        "detection_stability": detection_stability,
        "feedback_precision": feedback_precision,
        "reliability_score": reliability_score
    }


# ---------------- REPORT GENERATION ----------------
def generate_html_report(summary_nums, output_html="report.html"):
    avg_conf = np.mean(confidence_history) * 100 if confidence_history else 0
    eye_contact_pct = np.mean(eye_contact_values) * 100 if eye_contact_values else 0
    posture_stability = np.mean(posture_values) * 100 if posture_values else 0
    pause_pct = np.mean(pause_flags) * 100 if pause_flags else 0
    avg_rate = np.mean(speech_rates) if speech_rates else 0
    avg_vol = np.mean(volumes) if volumes else 0

    html = f"""
    <html>
    <head>
      <title>Interview Performance Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 60%; }}
        td, th {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background: #f2f2f2; text-align:left; }}
        .charts {{ margin-top: 20px; }}
      </style>
    </head>
    <body>
      <h1>Interview Performance Report</h1>
      <h3>Summary Metrics</h3>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Average Confidence</td><td>{avg_conf:.1f}%</td></tr>
        <tr><td>Eye Contact maintained</td><td>{eye_contact_pct:.1f}%</td></tr>
        <tr><td>Posture stability</td><td>{posture_stability:.1f}%</td></tr>
        <tr><td>Pauses Detected</td><td>{pause_pct:.1f}%</td></tr>
        <tr><td>Average Speech Rate</td><td>{avg_rate:.1f}</td></tr>
        <tr><td>Average Volume Level</td><td>{avg_vol:.3f}</td></tr>
        <tr><td>Detection Stability</td><td>{summary_nums['detection_stability']:.2f}%</td></tr>
        <tr><td>Feedback Precision</td><td>{summary_nums['feedback_precision']:.2f}%</td></tr>
        <tr><td>Reliability Score</td><td>{summary_nums['reliability_score']:.2f}%</td></tr>
      </table>

      <div class="charts">
        <h3>Charts</h3>
        <img src="performance_summary.png" alt="Performance Summary" style="max-width:800px;">
        <br><br>
        <img src="system_accuracy.png" alt="System Accuracy" style="max-width:600px;">
      </div>
    </body>
    </html>
    """
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Saved HTML report: {output_html}")


def generate_pdf_report(summary_nums, output_pdf="report.pdf"):
    # Create PDF with reportlab
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Interview Performance Report")
    y -= 30

    c.setFont("Helvetica", 10)
    lines = [
        f"Average Confidence: {np.mean(confidence_history)*100:.1f}%" if confidence_history else "Average Confidence: 0.0%",
        f"Eye Contact maintained: {np.mean(eye_contact_values)*100:.1f}%" if eye_contact_values else "Eye Contact maintained: 0.0%",
        f"Posture stability: {np.mean(posture_values)*100:.1f}%" if posture_values else "Posture stability: 0.0%",
        f"Pauses Detected: {np.mean(pause_flags)*100:.1f}%" if pause_flags else "Pauses Detected: 0.0%",
        f"Average Speech Rate: {np.mean(speech_rates):.1f}" if speech_rates else "Average Speech Rate: 0.0",
        f"Average Volume Level: {np.mean(volumes):.3f}" if volumes else "Average Volume Level: 0.000",
        f"Detection Stability: {summary_nums['detection_stability']:.2f}%",
        f"Feedback Precision: {summary_nums['feedback_precision']:.2f}%",
        f"Reliability Score: {summary_nums['reliability_score']:.2f}%"
    ]

    for line in lines:
        c.drawString(margin, y, line)
        y -= 14

    # Add images (if exist) below
    y -= 10
    for img_path in ("performance_summary.png", "system_accuracy.png"):
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                # scale to page width - margins
                max_w = width - 2*margin
                ratio = min(max_w / img.width, 300 / img.height)
                draw_w = img.width * ratio
                draw_h = img.height * ratio
                y -= draw_h
                c.drawImage(ImageReader(img), margin, y, width=draw_w, height=draw_h)
                y -= 20
                if y < 120:
                    c.showPage()
                    y = height - margin
            except Exception as e:
                print("Could not add image to PDF:", e)

    c.save()
    print(f"✅ Saved PDF report: {output_pdf}")


# ---------------- MAIN REPORTING & PRINT ----------------
def print_report():
    if not confidence_history:
        print("No data recorded.")
        return
    avg_conf = np.mean(confidence_history) * 100
    eye_contact_pct = np.mean(eye_contact_values) * 100 if eye_contact_values else 0
    posture_stability = np.mean(posture_values) * 100 if posture_values else 0
    pause_pct = np.mean(pause_flags) * 100 if pause_flags else 0
    avg_rate = np.mean(speech_rates) if speech_rates else 0
    avg_vol = np.mean(volumes) if volumes else 0

    print("\n----- INTERVIEW PERFORMANCE REPORT -----")
    print(f"Average Confidence: {avg_conf:.1f}%")
    print(f"Eye Contact maintained: {eye_contact_pct:.1f}% of time")
    print(f"Posture stability: {posture_stability:.1f}%")
    print(f"Pauses Detected: {pause_pct:.1f}% of time")
    print(f"Average Speech Rate: {avg_rate:.1f} (relative units)")
    print(f"Average Volume Level: {avg_vol:.3f}")
    print("----------------------------------------")
    print("Charts saved: performance_summary.png, system_accuracy.png")


# ---------------- MAIN ----------------
def run_interview():
    global running, current_question
    print("\nWelcome to the AI Interview Coach")
    input("Press Enter to begin...\n")

    threading.Thread(target=monitor_audio, daemon=True).start()
    threading.Thread(target=monitor_camera, daemon=True).start()

    for i, q in enumerate(QUESTIONS, 1):
        if not running:
            print("⚠️ Interview stopped early.")
            break

        current_question = q
        print(f"\nQ{i}: {q}")
        start_q = time.time()
        while time.time() - start_q < QUESTION_DURATION and running:
            time.sleep(1)

        current_question = ""
        if not running:
            print("⚠️ Interview stopped before completing all questions.")
            break

        print("⏱️ Time’s up! Moving to next question...")

    running = False
    time.sleep(1)
    plot_summary_charts()
    summary = compute_reliability_metrics()
    generate_pdf_report(summary)

    print("\nSession complete — PDF report opened automatically.")
    running = False
    time.sleep(1)

    # Generate plots & metrics
    plot_summary_charts()
    summary_nums = compute_reliability_metrics()
    print_report()

    # Create HTML + PDF reports
    generate_html_report(summary_nums, output_html="report.html")
    generate_pdf_report(summary_nums, output_pdf="report.pdf")

    print("\nSession complete. Performance and reliability report generated.")
    print("Files: performance_summary.png, system_accuracy.png, report.html, report.pdf")

    # ---------------- Auto-open PDF Report ----------------
    try:
        import platform, subprocess, os
        pdf_path = os.path.abspath("report.pdf")

        if platform.system() == "Windows":
            os.startfile(pdf_path)  # Opens with default app (e.g., Edge/Adobe)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", pdf_path])
        else:  # Linux
            subprocess.Popen(["xdg-open", pdf_path])

        print(f"📂 Automatically opened: {pdf_path}")
    except Exception as e:
        print(f"⚠️ Could not auto-open PDF: {e}")



if __name__ == "__main__":
    run_interview()










