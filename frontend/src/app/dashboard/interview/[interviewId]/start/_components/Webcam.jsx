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
  const [videoBlob, setVideoBlob] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const { user } = useUser();
  const router = useRouter();

  // UploadThing uploader
  const { startUpload } = useUploadThing("videoUploader");

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: true,
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: "video/webm",
      });

      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoBlob(blob);
        chunksRef.current = [];
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (e) {
      toast.error("Unable to access webcam.");
    }
  };

  // Stop recording
  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return;

    await new Promise((resolve) => {
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        setVideoBlob(blob);
        chunksRef.current = [];
        resolve();
      };

      mediaRecorderRef.current.stop();
    });

    mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
    setIsRecording(false);
  };

  // Save answer → upload video → transcribe → update UI
  const handleSaveAnswer = async () => {
    if (!videoBlob) {
      toast.error("Please record first.");
      return;
    }

    const videoFile = new File(
      [videoBlob],
      `interview-${data?.mockId}-q${activeIndex + 1}.webm`,
      { type: "video/webm" }
    );

    try {
      // Upload to UploadThing
      const uploaded = await startUpload([videoFile]);
      if (!uploaded) throw new Error("Upload failed");

      const videoUrl = uploaded[0].url;

      // Transcribe using Groq Whisper
      const fd = new FormData();
      fd.append("file", videoFile);

      const res = await fetch("/api/transcribe", {
        method: "POST",
        body: fd,
      });

      const { text } = await res.json();
      setUserAns(text);

      onAnswerUpdate(text);
    } catch (err) {
      console.error(err);
      toast.error("Error saving answer.");
    }
  };

  // End interview → generate feedback → save in DB
  const handleEndInterview = async () => {
    try {
      for (let i = 0; i < interviewQues.length; i++) {
        const prompt = `Question: ${interviewQues[i]?.question}, User answer: ${userAnswers[i]}, give JSON {rating,feedback,answer}`;

        const res = await chatSession.sendMessage(prompt);

        const json = JSON.parse(
          res.response.text().replace("```json", "").replace("```", "")
        );

        await db.insert(UserAnswer).values({
          mockIdRef: data?.mockId,
          question: interviewQues[i]?.question,
          correctAns: json.answer,
          userAns: userAnswers[i],
          feedback: json.feedback,
          rating: Number(json.rating), // important fix
          userEmail: user?.primaryEmailAddress?.emailAddress,
          createdAt: moment().format("DD-MM-yyyy"),
        });
      }

      router.push(`/dashboard/interview/${data?.mockId}/feedback`);
    } catch (e) {
      console.error(e);
      toast.error("Failed saving interview.");
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

      {userAns && (
        <p className="mt-4 p-3 bg-gray-800 rounded">{userAns}</p>
      )}
    </div>
  );
}

export default WebcamComponent;
