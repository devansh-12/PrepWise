import { NextResponse } from 'next/server';
import nodemailer from 'nodemailer';
import { db } from '@/utils/db';
import { allowedUsers } from '@/utils/schema';
import { eq } from 'drizzle-orm';

export async function POST(req) {
  try {
    const body = await req.json();
    const { userEmail } = body;

    console.log("user email", userEmail);

    if (!userEmail) {
      return NextResponse.json({ error: 'Missing userEmail' }, { status: 400 });
    }

    // 1. Check if already exists
    const existing = await db
      .select()
      .from(allowedUsers)
      .where(eq(allowedUsers.email, userEmail));

    console.log("Existing user:", existing);

    if (existing.length > 0) {
      return NextResponse.json(
        { message: "Request already submitted." },
        { status: 200 }
      );
    }

    // 2. Insert new request
    await db.insert(allowedUsers).values({
      email: userEmail,
      status: "pending",
    });

    console.log("Inserted user:", userEmail);

    // 3. SMTP transport
    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT || 587),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });

    // 4. Email URLs
    const base = "https://ai-prep-gem.vercel.app";

    const approveUrl = `${base}/api/handle-request?action=approve&email=${encodeURIComponent(userEmail)}`;
    const rejectUrl = `${base}/api/handle-request?action=reject&email=${encodeURIComponent(userEmail)}`;
    const tempUrl = `${base}/api/handle-request?action=temporary&email=${encodeURIComponent(userEmail)}`;

    // 5. Send email to admin
    await transporter.sendMail({
      from: `"1-1 Access Request" <${process.env.SMTP_USER}>`,
      to: process.env.ADMIN_EMAIL,
      subject: "Access Request for 1-1 Feature",
      html: `
        <h2>Access Request</h2>
        <p>User <strong>${userEmail}</strong> requested access.</p>
        <p>
          <a href="${approveUrl}">Approve</a> |
          <a href="${rejectUrl}">Reject</a> |
          <a href="${tempUrl}">Temporary Approve (20 min)</a>
        </p>
      `,
    });

    return NextResponse.json(
      { message: "Request submitted and email sent." },
      { status: 200 }
    );
  } catch (error) {
    console.error("[send-request] Error:", error);
    return NextResponse.json(
      { error: "Internal Server Error", details: error.message },
      { status: 500 }
    );
  }
}
