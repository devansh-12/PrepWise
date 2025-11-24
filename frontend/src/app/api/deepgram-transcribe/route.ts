import { createClient } from "@deepgram/sdk";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: Request) {
    try {
        const form = await req.formData();
        const file = form.get("file") as File;

        if (!file) {
            return NextResponse.json({ error: "No file received" }, { status: 400 });
        }

        const buffer = Buffer.from(await file.arrayBuffer());

        // Initialize Deepgram client
        const deepgram = createClient(process.env.DEEPGRAM_API_KEY!);

        // Transcribe the audio
        const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
            buffer,
            {
                model: "nova-2",
                smart_format: true,
                punctuate: true,
                paragraphs: false,
                utterances: false,
            }
        );

        if (error) {
            console.error("Deepgram error:", error);
            return NextResponse.json({ error: error.message }, { status: 500 });
        }

        const transcript = result.results.channels[0].alternatives[0].transcript;

        return NextResponse.json({ text: transcript });
    } catch (err: any) {
        console.error("Transcription error:", err);
        return NextResponse.json({ error: err.message }, { status: 500 });
    }
}
