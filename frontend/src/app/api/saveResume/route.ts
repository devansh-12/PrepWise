// src/app/api/saveResume/route.ts
import { NextResponse } from "next/server"
import { db } from "@/utils/db"
import { UserResume } from "@/utils/schema"
import moment from "moment"

export async function POST(req: Request) {
  try {
    const { userEmail, resumeUrl, uploadKey } = await req.json()

    if (!userEmail || !resumeUrl) {
      return NextResponse.json({ error: "Missing parameters" }, { status: 400 })
    }

    await db.insert(UserResume).values({
      userEmail,
      resumeUrl,
      uploadKey: uploadKey || null,
      uploadedAt: moment().format("DD-MM-YYYY HH:mm:ss"),
    })

    return NextResponse.json({ success: true })
  } catch (err: any) {
    console.error("[saveResume] Error:", err)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
