import { NextResponse } from "next/server";
import Groq from "groq-sdk";

export const runtime = "nodejs"; // ensures fs & buffers work

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file received" }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());

    const groq = new Groq({ apiKey: process.env.GROQ_KEY });

    const transcript = await groq.audio.transcriptions.create({
    file: {
    name: file.name,
    buffer,
    } as any,                 // 👈 FIX TS ERROR
    model: "whisper-large-v3",
   response_format: "verbose_json",
  });


    return NextResponse.json({ text: transcript.text });
  } catch (err: any) {
    console.error("Transcription error:", err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
