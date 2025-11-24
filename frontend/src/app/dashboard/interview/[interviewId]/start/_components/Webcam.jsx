"use client";

import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { chatSession } from "@/utils/GeminiAPIModel";
import { db } from "@/utils/db";
import { UserAnswer } from "@/utils/schema";
import { useUser } from "@clerk/nextjs";
import moment from "moment";
import { toast } from "sonner";
import { Mic, CircleStopIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useUploadThing } from "@/utils/uploadthing";

// Cleans LLM output into valid JSON
const safeClean = (raw) => {
  return raw
    .trim()
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .replace(/^\s+|\s+$/g, "")
    .replace(/^[^{\[]*/, "") // remove junk before JSON starts
    .replace(/[^}\]]*$/, ""); // remove junk after JSON ends
};

function WebcamComponent({
  data,
  interviewQues,
  activeIndex,
  setActiveIndex,
  userAnswers,
  onAnswerUpdate,
}) {
  const [userAns, setUserAns] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [recordingDuration, setRecordingDuration] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const { user } = useUser();
  const router = useRouter();

  // UploadThing uploader
  const { startUpload } = useUploadThing("videoUploader");

  // Maximum recording duration in seconds (2 minutes to keep file size small)
  const MAX_DURATION = 120;

  // Start recording - AUDIO ONLY to reduce file size
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000, // Lower sample rate for smaller files
        },
        video: false,
      });

      // Use lower bitrate for smaller files
      const options = {
        mimeType: "audio/webm;codecs=opus",
        audioBitsPerSecond: 32000, // 32kbps - good quality, small size
      };

      mediaRecorderRef.current = new MediaRecorder(stream, options);
      chunksRef.current = [];
      setRecordingDuration(0);
      setUserAns(""); // Clear previous transcription when starting new recording

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        chunksRef.current = [];
        if (timerRef.current) clearInterval(timerRef.current);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingDuration((prev) => {
          const newDuration = prev + 1;
          // Auto-stop at max duration
          if (newDuration >= MAX_DURATION) {
            stopRecording();
            toast.info("Maximum recording duration reached (2 minutes)");
          }
          return newDuration;
        });
      }, 1000);
    } catch (e) {
      console.error("Microphone access error:", e);
      toast.error("Unable to access microphone.");
    }
  };

  // Stop recording
  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return;

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    await new Promise((resolve) => {
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        chunksRef.current = [];
        resolve();
      };

      mediaRecorderRef.current.stop();
    });

    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    setIsRecording(false);
  };

  // Save answer → transcribe with Deepgram (browser speech as fallback)
  const handleSaveAnswer = async () => {
    if (!audioBlob) {
      toast.error("Please record first.");
      return;
    }

    const audioFile = new File(
      [audioBlob],
      `interview-${data?.mockId}-q${activeIndex + 1}.webm`,
      { type: "audio/webm" }
    );

    try {
      // Try Deepgram transcription first
      const fd = new FormData();
      fd.append("file", audioFile);

      toast.info("Transcribing your answer...");

      const res = await fetch("/api/deepgram-transcribe", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "Deepgram transcription failed");
      }

      const { text } = await res.json();

      if (!text || text.trim() === "") {
        throw new Error("No speech detected in recording");
      }

      setUserAns(text);
      onAnswerUpdate(text);
      toast.success("Answer saved!");

      // Upload audio in background (non-blocking, optional)
      startUpload([audioFile])
        .then((uploaded) => {
          if (uploaded) {
            console.log("Audio uploaded:", uploaded[0].url);
          }
        })
        .catch((err) => {
          console.warn("Audio upload failed (non-critical):", err);
        });
    } catch (err) {
      console.error("Deepgram transcription error:", err);

      // Fallback to browser Speech Recognition
      if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        toast.info("Trying browser speech recognition...");

        try {
          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          const recognition = new SpeechRecognition();

          // Create audio element to play the recorded audio
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);

          recognition.lang = 'en-US';
          recognition.continuous = true;
          recognition.interimResults = false;

          let transcript = '';

          recognition.onresult = (event) => {
            for (let i = 0; i < event.results.length; i++) {
              if (event.results[i].isFinal) {
                transcript += event.results[i][0].transcript + ' ';
              }
            }
          };

          recognition.onend = () => {
            if (transcript.trim()) {
              setUserAns(transcript.trim());
              onAnswerUpdate(transcript.trim());
              toast.success("Answer saved using browser speech recognition!");
            } else {
              toast.error("Could not transcribe audio. Please try recording again.");
            }
            URL.revokeObjectURL(audioUrl);
          };

          recognition.onerror = (event) => {
            console.error('Browser speech recognition error:', event.error);
            toast.error("Transcription failed. Please try recording again or type your answer manually.");
            URL.revokeObjectURL(audioUrl);
          };

          // Note: Browser speech recognition works with live audio, not recorded files
          // So we'll just show an error message
          toast.error("Browser speech recognition requires live audio. Please use the record button and speak your answer.");

        } catch (fallbackErr) {
          console.error("Fallback error:", fallbackErr);
          toast.error("All transcription methods failed. Please try again.");
        }
      } else {
        toast.error("Transcription failed: " + err.message);
      }
    }
  };

  // End interview → generate feedback → save in DB
  const handleEndInterview = async () => {
    try {
      for (let i = 0; i < interviewQues.length; i++) {
        try {
          const prompt = `Question: ${interviewQues[i]?.question}, User answer: ${userAnswers[i]}, give JSON {rating,feedback,answer}`;

          const res = await chatSession.sendMessage(prompt);
          const raw = res.response.text();
          console.log(`Raw AI response for Q${i + 1}:`, raw);

          const cleaned = safeClean(raw);
          console.log(`Cleaned JSON for Q${i + 1}:`, cleaned);

          let json;
          try {
            json = JSON.parse(cleaned);
          } catch (parseErr) {
            console.error(`JSON parse error for Q${i + 1}:`, parseErr);
            console.error(`Failed to parse:`, cleaned);
            // Use fallback values if parsing fails
            json = {
              rating: "5",
              feedback: "Unable to generate feedback",
              answer: "N/A"
            };
          }

          await db.insert(UserAnswer).values({
            mockIdRef: data?.mockId,
            question: interviewQues[i]?.question,
            correctAns: json.answer || "N/A",
            userAns: userAnswers[i] || "",
            feedback: json.feedback || "No feedback available",
            rating: Number(json.rating) || 5,
            userEmail: user?.primaryEmailAddress?.emailAddress,
            createdAt: moment().format("DD-MM-yyyy"),
          });
        } catch (questionErr) {
          console.error(`Error processing question ${i + 1}:`, questionErr);
          // Continue with next question even if this one fails
        }
      }

      router.push(`/dashboard/interview/${data?.mockId}/feedback`);
    } catch (e) {
      console.error("Overall error in handleEndInterview:", e);
      toast.error("Failed saving interview: " + e.message);
    }
  };

  return (
    <div className="min-h-screen p-4 text-foreground">
      <div className="flex justify-end space-x-2 mb-4">
        {activeIndex !== interviewQues.length - 1 ? (
          <Button onClick={() => setActiveIndex(activeIndex + 1)}>
            Next →
          </Button>
        ) : (
          <Button variant="destructive" onClick={handleEndInterview}>
            End Interview
          </Button>
        )}
      </div>

      <div className="flex flex-col justify-center items-center bg-black my-5 rounded-lg p-10">
        <Image
          src="/cam.jpg"
          width={150}
          height={200}
          alt="camera overlay"
          className="absolute"
        />
        <Webcam mirrored style={{ zIndex: 100, height: 250, width: "100%" }} />
      </div>

      <div className="flex gap-2">
        {!isRecording ? (
          <Button variant="outline" className="flex-1" onClick={startRecording}>
            <Mic /> Record
          </Button>
        ) : (
          <Button variant="outline" className="flex-1" onClick={stopRecording}>
            <CircleStopIcon className="text-red-500" /> Stop
          </Button>
        )}

        <Button variant="secondary" onClick={handleSaveAnswer}>
          Save
        </Button>
      </div>

      {isRecording && (
        <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-center">
          <p className="text-red-400 text-sm">
            🔴 Recording: {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')} / 2:00
          </p>
        </div>
      )}

      {userAns && (
        <div className="mt-4 p-4 bg-gradient-to-br from-purple-900/30 to-blue-900/30 border-2 border-purple-500/50 rounded-lg shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-purple-400 font-semibold text-sm">Your Transcription:</span>
            <span className="text-green-400 text-xs">✓ Saved</span>
          </div>
          <p className="text-gray-100 text-base leading-relaxed">{userAns}</p>
        </div>
      )}
    </div>
  );
}

export default WebcamComponent;

