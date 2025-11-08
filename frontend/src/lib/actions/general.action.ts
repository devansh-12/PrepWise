// lib/actions/general.action.ts
"use server";

import { generateObject } from "ai";
import { google } from "@ai-sdk/google";
import { db } from "@/firebase/admin";
import { 
  feedbackSchema, 
  CreateFeedbackParams, 
  GetFeedbackByInterviewIdParams,
  GetLatestInterviewsParams,
  Interview,
  Feedback,
  FeedbackType,
  CompleteFeedback
} from "@/lib/types";

export async function createFeedback(params: CreateFeedbackParams) {
  const { interviewId, userId, transcript, feedbackId } = params;
  let feedback: Partial<CompleteFeedback> | null = null;

  try {
    const formattedTranscript = transcript
      .map(
        (sentence: { role: string; content: string }) =>
          `- ${sentence.role}: ${sentence.content}\n`
      )
      .join("");

    const { object } = await generateObject({
      model: google("gemini-2.0-flash", {
        structuredOutputs: true,
      }) as any,
      schema: feedbackSchema,
      prompt: `
        You are an AI interviewer analyzing a mock interview. Your task is to evaluate the candidate based on structured categories. Be thorough and detailed in your analysis. Don't be lenient with the candidate.
        
        Transcript:
        ${formattedTranscript}

        Please score the candidate from 0 to 100 in these areas only:
        - Communication Skills: Clarity, articulation, structured responses
        - Technical Knowledge: Understanding of key concepts for the role
        - Problem-Solving: Ability to analyze problems and propose solutions
        - Cultural & Role Fit: Alignment with company values and job role
        - Confidence & Clarity: Confidence in responses, engagement, and clarity
      `,
      system:
        "You are a professional interviewer analyzing a mock interview. Provide scores 0-100 and detailed feedback.",
    });

    // Type-safe object destructuring
    const typedObject = object as FeedbackType;

    // Construct complete feedback with database metadata
    const completeFeedback: CompleteFeedback = {
      interviewId,
      userId,
      totalScore: typedObject.totalScore,
      categoryScores: typedObject.categoryScores,
      strengths: typedObject.strengths,
      areasForImprovement: typedObject.areasForImprovement,
      finalAssessment: typedObject.finalAssessment,
      createdAt: new Date().toISOString(),
    };

    feedback = completeFeedback;
    console.log("Generated feedback:", feedback);

    const feedbackRef = feedbackId
      ? db.collection("feedback").doc(feedbackId)
      : db.collection("feedback").doc();

    // Type-safe set with proper type
    await feedbackRef.set(completeFeedback, { merge: true });

    return { success: true, feedbackId: feedbackRef.id };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    console.error("Error creating feedback:", errorMessage);
    if (feedback) console.error("Partial feedback data:", feedback);
    return { success: false, error: errorMessage };
  }
}

export async function getInterviewById(id: string): Promise<Interview | null> {
  try {
    const interview = await db.collection("interviews").doc(id).get();
    return interview.data() as Interview | null;
  } catch (error) {
    console.error("Error fetching interview:", error);
    return null;
  }
}

export async function getFeedbackByInterviewId(
  params: GetFeedbackByInterviewIdParams
): Promise<Feedback | null> {
  const { interviewId, userId } = params;

  try {
    const querySnapshot = await db
      .collection("feedback")
      .where("interviewId", "==", interviewId)
      .where("userId", "==", userId)
      .limit(1)
      .get();

    if (querySnapshot.empty) return null;

    const feedbackDoc = querySnapshot.docs[0];
    return { id: feedbackDoc.id, ...feedbackDoc.data() } as Feedback;
  } catch (error) {
    console.error("Error fetching feedback:", error);
    return null;
  }
}

export async function getLatestInterviews(
  params: GetLatestInterviewsParams
): Promise<Interview[]> {
  const { userId, limit = 20 } = params;

  try {
    const interviews = await db
      .collection("interviews")
      .where("finalized", "==", true)
      .where("userId", "!=", userId)
      .orderBy("userId")
      .orderBy("createdAt", "desc")
      .limit(limit)
      .get();

    return interviews.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    })) as Interview[];
  } catch (error) {
    console.error("Error fetching latest interviews:", error);
    return [];
  }
}

export async function getInterviewsByUserId(
  userId: string
): Promise<Interview[]> {
  try {
    const interviews = await db
      .collection("interviews")
      .where("userId", "==", userId)
      .orderBy("createdAt", "desc")
      .get();

    return interviews.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    })) as Interview[];
  } catch (error) {
    console.error("Error fetching user interviews:", error);
    throw new Error("Failed to fetch interviews");
  }
}
