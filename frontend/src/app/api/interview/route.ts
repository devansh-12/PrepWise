import { NextResponse } from "next/server"
import { db } from "@/utils/db"
import { MockInterview } from "@/utils/schema"
import { v4 as uuidv4 } from "uuid"
import moment from "moment"

export async function POST(req: Request) {
  try {
    const body = await req.json()

    // ✅ Insert interview using Drizzle ORM
    const inserted = await db
      .insert(MockInterview)
      .values({
        mockId: body.mockId ?? uuidv4(),
        jsonMockResp: JSON.stringify(body.jsonMockResp),
        jobPosition: body.jobPosition,
        jobDesc: body.jobDesc,
        jobExperience: parseInt(body.jobExperience, 10),
        createdBy: body.createdBy,
        createdAt: body.createdAt ?? moment().format("DD-MM-YYYY"),
      })
      .returning({ mockId: MockInterview.mockId })

    return NextResponse.json({ success: true, mockId: inserted[0].mockId })
  } catch (err: any) {
    console.error("❌ Error creating interview:", err)
    return NextResponse.json(
      { success: false, error: "Failed to create interview" },
      { status: 500 }
    )
  }
}
