import { NextResponse } from "next/server";
import { db } from "@/utils/db";
import { InterviewVideo } from "@/utils/schema";
import moment from "moment";

export async function POST(req: Request) {
  try {
    const { mockIdRef, questionNo, videoUrl, uploadKey } = await req.json();

    if (!mockIdRef || !questionNo || !videoUrl) {
      return NextResponse.json({ error: "Missing parameters" }, { status: 400 });
    }

    await db.insert(InterviewVideo).values({
      mockIdRef,
      questionNo,
      videoUrl,
      uploadKey: uploadKey || null,
      uploadedAt: moment().format("DD-MM-YYYY HH:mm:ss"),
    });

    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("[saveVideo] Error:", err);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
