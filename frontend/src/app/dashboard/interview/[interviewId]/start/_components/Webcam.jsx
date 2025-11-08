"use client";
import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { chatSession } from "@/utils/GeminiAPIModel";
import { db } from "@/utils/db";
import { UserAnswer } from "@/utils/schema";
import { useUser } from "@clerk/nextjs";
import moment from "moment";
import { toast } from "sonner";
import { LoaderCircle, Mic, CircleStopIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import Groq from "groq-sdk";

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
  const [videoChunks, setVideoChunks] = useState([]);
  const mediaRecorderRef = useRef(null);
  const [loader, setLoader] = useState(false);
  const { user } = useUser();
  const router = useRouter();

  useEffect(() => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.error("Audio/Video Recording not supported in this browser.");
      toast.error("Browser doesn't support webcam recording.");
      return;
    }

    return () => {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  /** 🎥 Start recording video + audio */
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: true,
      });
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: "video/webm",
      });
      setVideoChunks([]);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setVideoChunks((prev) => [...prev, event.data]);
        }
      };

      mediaRecorderRef.current.start(1000);
      setIsRecording(true);
      toast.info("Recording started!");
    } catch (error) {
      console.error("Error accessing media devices:", error);
      toast.error("Failed to access webcam or microphone.");
    }
  };

  /** ⏹ Stop recording + process transcription + upload video */
  const stopRecording = async () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      mediaRecorderRef.current.onstop = async () => {
        try {
          const videoBlob = new Blob(videoChunks, { type: "video/webm" });
          if (videoBlob.size === 0) throw new Error("No video data recorded.");

          // 🧠 Step 1: Upload video to UploadThing
          const videoFile = new File(
            [videoBlob],
            `interview-${data?.mockId}-q${activeIndex + 1}.webm`,
            { type: "video/webm" }
          );

          const formData = new FormData();
          formData.append("file", videoFile);

          const uploadRes = await fetch("/api/uploadthing/videoUploader", {
            method: "POST",
            body: formData,
          });

          if (!uploadRes.ok) throw new Error("Upload failed");
          const uploadData = await uploadRes.json();

          // 🗃️ Step 2: Save metadata to DB
          await fetch("/api/saveVideo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              mockIdRef: data?.mockId,
              questionNo: activeIndex + 1,
              videoUrl: uploadData.url,
              uploadKey: uploadData.key,
            }),
          });

          toast.success("Video uploaded and saved!");

          // 🧩 Step 3: Transcribe audio from same blob via Groq
          const groq = new Groq({
            apiKey: process.env.NEXT_PUBLIC_GROQ_API_KEY,
            dangerouslyAllowBrowser: true,
          });

          const transcription = await groq.audio.transcriptions.create({
            file: videoFile,
            model: "whisper-large-v3",
            response_format: "verbose_json",
          });

          setUserAns(transcription.text);
        } catch (err) {
          console.error("Error processing video:", err);
          toast.error(err.message || "Failed to upload/transcribe video");
        } finally {
          mediaRecorderRef.current.stream
            .getTracks()
            .forEach((track) => track.stop());
          setVideoChunks([]);
        }
      };
    }
  };

  const handleButtonClick = async () => {
    if (isRecording) await stopRecording();
    else await startRecording();
  };

  const handleSaveAnswer = () => {
    onAnswerUpdate(userAns);
    setUserAns("");
    toast.success("Answer saved for this question!");
  };

  /** 💾 End Interview - save all answers + feedback */
  const handleEndInterview = async () => {
    setLoader(true);
    try {
      for (let i = 0; i < interviewQues.length; i++) {
        if (userAnswers[i]) {
          const feedbackPrompt = `Question: ${interviewQues[i]?.question}, User's answer: ${userAnswers[i]}, depending on given question and user's answer please give a rating (out of 10) for the answer and feedback as area of improvement if any, and also a recommended answer for the user if you were in his place, in max 3-4 lines in JSON format with rating,feedback and answer as fields in JSON.`;

          const res = await chatSession.sendMessage(feedbackPrompt);
          const mockJsonResp = res.response
            .text()
            .replace("```json", "")
            .replace("```", "");
          const JsonFeedback = JSON.parse(mockJsonResp);

          await db.insert(UserAnswer).values({
            mockIdRef: data?.mockId,
            question: interviewQues[i]?.question,
            correctAns: JsonFeedback?.answer,
            userAns: userAnswers[i],
            feedback: JsonFeedback?.feedback,
            rating: JsonFeedback?.rating,
            userEmail: user?.primaryEmailAddress?.emailAddress,
            createdAt: moment().format("DD-MM-yyyy"),
          });
        }
      }

      toast.success("Interview completed successfully!");
      router.push(`/dashboard/interview/${data?.mockId}/feedback`);
    } catch (error) {
      console.error("Error saving interview:", error);
      toast.error("Error saving interview responses");
    } finally {
      setLoader(false);
    }
  };

  return (
    <div className="min-h-screen p-4 text-foreground">
      {/* 🔘 Controls */}
      <div className="flex justify-end space-x-2 mb-4">
        {activeIndex !== interviewQues?.length - 1 && (
          <Button onClick={() => setActiveIndex(activeIndex + 1)}>
            Next {" >"}
          </Button>
        )}
        {activeIndex === interviewQues?.length - 1 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleEndInterview}
            disabled={loader}
          >
            {loader ? (
              <LoaderCircle className="h-4 w-4 animate-spin" />
            ) : (
              "End Interview"
            )}
          </Button>
        )}
      </div>

      {/* 🎥 Video preview */}
      <div className="flex flex-col justify-center items-center bg-black my-5 rounded-lg p-10">
        <Image src={"/cam.jpg"} width={150} height={200} className="absolute" />
        <Webcam mirrored={true} style={{ zIndex: 100, height: 250, width: "100%" }} />
      </div>

      {/* 🎙 Controls + transcript */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            className="flex-1 my-1"
            onClick={handleButtonClick}
          >
            {isRecording ? (
              <span className="text-red-500 flex gap-1">
                <CircleStopIcon size={18} />
                Recording..
              </span>
            ) : (
              <span className="flex gap-1">
                <Mic size={18} />
                Record Answer
              </span>
            )}
          </Button>
          <Button variant="secondary" onClick={handleSaveAnswer}>
            Save
          </Button>
        </div>

        {userAns && (
          <div className="bg-gray-900 rounded-lg border p-4 transition-all duration-300">
            <p className="text-sm text-gray-100">{userAns}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default WebcamComponent;
