"use client";

import { db } from "@/utils/db";
import { UserAnswer } from "@/utils/schema";
import { eq } from "drizzle-orm";
import React, { useEffect, useState } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import Link from "next/link";

function FeedbackPage() {
  const [feedbackList, setFeedbackList] = useState([]);
  const router = useRouter();
  const params = useParams();

  // Safely extract interviewId from params (string or string[])
  const interviewId = React.useMemo(() => {
    if (!params) return undefined;
    const raw = params.interviewId;
    return Array.isArray(raw) ? raw[0] : raw;
  }, [params]);

  useEffect(() => {
    if (!interviewId) return;

    const getFeedback = async () => {
      try {
        console.log("params id", interviewId);
        const result = await db
          .select()
          .from(UserAnswer)
          .where(eq(UserAnswer.mockIdRef, interviewId));
        console.log("db data", result);
        setFeedbackList(result);
      } catch (err) {
        console.error("Error loading feedback:", err);
      }
    };

    getFeedback();
  }, [interviewId]);

  return (
    <div className="text-white min-h-screen p-6 rounded-lg">
      {feedbackList.length > 0 ? (
        <>
          <h2 className="my-4 text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-300">
            Congratulations, Here is your interview feedback!
          </h2>
          <h2 className="text-lg my-2 mx-1 text-gray-200">
            Overall interview rating:{" "}
            <span className="font-semibold text-white">7/10</span>
            {/* you can compute actual avg later */}
          </h2>
          <h2 className="my-6 mx-1 text-gray-300">
            Find below your questions, answers and feedbacks related to the interview
          </h2>
        </>
      ) : (
        <>
          <h2 className="mt-10 mb-2 text-2xl font-bold text-gray-200">
            Oops, looks like you haven&apos;t completed the interview for this post yet!
          </h2>
          {interviewId && (
            <Link href={`/dashboard/interview/${interviewId}`}>
              <span className="text-blue-500 hover:text-blue-400 transition-colors duration-200">
                Give Interview now?
              </span>
            </Link>
          )}
        </>
      )}

      {feedbackList &&
        feedbackList.map((item, index) => (
          <div className="my-4" key={index}>
            <Collapsible>
              <CollapsibleTrigger className="w-full p-3 rounded-lg bg-gradient-to-r from-gray-800 to-gray-700 flex justify-between items-center gap-2 hover:from-gray-700 hover:to-gray-600 transition-all duration-200 text-gray-200">
                {item?.question}
                <ChevronDown className="text-gray-400" />
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="my-2 bg-gradient-to-br from-gray-800/90 to-gray-700/90 border border-gray-600 rounded-lg flex flex-col gap-3 p-5 text-gray-200">
                  <h2
                    className={`${
                      Number(item?.rating) < 5 ? "text-red-400" : "text-blue-400"
                    } font-semibold`}
                  >
                    <span>Score</span> : {item?.rating}/10
                  </h2>
                  <h2>
                    <span className="font-semibold text-gray-300">Your answer</span> :{" "}
                    {item?.userAns}
                  </h2>
                  <h2>
                    <span className="font-semibold text-gray-300">Feedback</span> :{" "}
                    {item?.feedback}
                  </h2>
                  <h2>
                    <span className="font-semibold text-gray-300">
                      Recommended answer
                    </span>{" "}
                    : {item?.correctAns}
                  </h2>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        ))}

      <div className="flex justify-end">
        {feedbackList.length > 0 && (
          <Button
            className="mt-6 bg-purple-300 hover:bg-purple-200 text-gray-800 transition-colors duration-200"
            onClick={() => router.replace("/dashboard")}
          >
            Go to home
          </Button>
        )}
      </div>
    </div>
  );
}

export default FeedbackPage;
