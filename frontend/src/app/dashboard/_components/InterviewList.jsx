"use client"

import { db } from "@/utils/db"
import { useUser } from "@clerk/nextjs"
import { MockInterview } from "@/utils/schema"
import { eq } from "drizzle-orm"
import React, { useEffect, useState, useRef } from "react"
import InterviewCard from "./InterviewCard"
import gsap from "gsap"

function InterviewList() {
  const { user } = useUser()
  const [list, setList] = useState([])
  const interviewListRef = useRef([])

  const getInterviewListforUser = async () => {
    if (!user?.primaryEmailAddress?.emailAddress) return
    const res = await db
      .select()
      .from(MockInterview)
      .where(eq(MockInterview.createdBy, user.primaryEmailAddress.emailAddress))
    console.log("response from db", res)
    setList(res)
  }

  useEffect(() => {
    if (user) getInterviewListforUser()
  }, [user])

  useEffect(() => {
    if (interviewListRef.current.length > 0) {
      gsap.fromTo(
        interviewListRef.current,
        { opacity: 0, y: 20 },
        {
          opacity: 1,
          y: 0,
          duration: 0.5,
          stagger: 0.15,
          ease: "power3.out",
        }
      )
    }
  }, [list])

  return (
    <div>
      <h2 className="text-2xl font-semibold text-purple-400 py-4">
        Previous Interviews
      </h2>
      <div className="space-y-4">
        {list && list.length > 0 ? (
          list.map((interview, index) => (
            <div
              key={interview.mockId || index}
              ref={(el) => (interviewListRef.current[index] = el)}
            >
              <InterviewCard interview={interview} />
            </div>
          ))
        ) : (
          <p className="text-gray-500 text-center py-4 pb-80">
            No interviews available
          </p>
        )}
      </div>
    </div>
  )
}

export default InterviewList
