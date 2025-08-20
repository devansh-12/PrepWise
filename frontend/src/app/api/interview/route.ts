// src/app/api/interview/route.ts
import { NextResponse } from "next/server";
import { db } from "@/utils/db";
import { v4 as uuidv4 } from "uuid";

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const newInterview = await db.aiInterview.create({
      data: {
        interview_id: uuidv4(),
        candidate_id: body.candidate_id ?? 1, // temp fallback
        job_position: body.job_position,
        job_description: body.job_description,
        experience_level: parseInt(body.experience_level, 10),
        feedback: body.feedback,
        created_at: new Date(),
      },
    });

    return NextResponse.json(newInterview);
  } catch (err: any) {
    console.error("Interview create error:", err);
    return NextResponse.json({ error: "Failed to create interview" }, { status: 500 });
  }
}
