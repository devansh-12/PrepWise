// lib/types/index.ts
import { z } from 'zod';

export interface CreateFeedbackParams {
  interviewId: string;
  userId: string;
  transcript: Array<{ role: string; content: string }>;
  feedbackId?: string;
}

export interface GetFeedbackByInterviewIdParams {
  interviewId: string;
  userId: string;
}

export interface GetLatestInterviewsParams {
  userId: string;
  limit?: number;
}

export interface Interview {
  id: string;
  userId: string;
  role: string;
  type: string;
  techstack: string[];
  level: string;
  questions: string[];
  finalized: boolean;
  createdAt: string;
}

export interface Feedback {
  id: string;
  interviewId: string;
  userId: string;
  totalScore: number;
  categoryScores: Array<{ name: string; score: number; comment: string }>;
  strengths: string[];
  areasForImprovement: string[];
  finalAssessment: string;
  createdAt: string;
}

// Schema for AI-generated feedback (what the model returns)
export const feedbackSchema = z.object({
  totalScore: z.number().min(0).max(100).describe("Overall interview score"),
  categoryScores: z.array(
    z.object({
      name: z.enum([
        "Communication Skills",
        "Technical Knowledge",
        "Problem-Solving",
        "Cultural & Role Fit",
        "Confidence & Clarity"
      ]),
      score: z.number().min(0).max(100),
      comment: z.string(),
    })
  ).min(5).max(5),
  strengths: z.array(z.string()).min(1),
  areasForImprovement: z.array(z.string()).min(1),
  finalAssessment: z.string(),
});

// Inferred type from Zod schema (AI model output only)
export type FeedbackType = z.infer<typeof feedbackSchema>;

// Complete feedback type including database metadata
export type CompleteFeedback = FeedbackType & {
  interviewId: string;
  userId: string;
  createdAt: string;
};
