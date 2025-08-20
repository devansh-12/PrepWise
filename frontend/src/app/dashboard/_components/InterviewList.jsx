'use client'
import { db } from '@/utils/db';
import { useUser } from '@clerk/nextjs';
import React, { useEffect, useState, useRef } from 'react';
import InterviewCard from './InterviewCard';
import gsap from 'gsap';

function InterviewList() {
    const { user } = useUser();
    const [list, setList] = useState([]);
    const interviewListRef = useRef([]);

    const getInterviewListforUser = async () => {
        if (!user?.primaryEmailAddress?.emailAddress) return;

        try {
            // Fetch CandidateProfile for this user
            const candidateProfile = await db.candidateProfile.findUnique({
                where: {
                    candidate_id: user.id ? parseInt(user.id) : -1, // assuming Clerk's user.id = User.user_id
                },
                include: {
                    interviews: true, // includes AiInterview[]
                },
            });

            const interviews = candidateProfile?.interviews || [];
            console.log("response from db", interviews);
            setList(interviews);
        } catch (err) {
            console.error("Error fetching interviews:", err);
        }
    };

    useEffect(() => {
        if (user) {
            getInterviewListforUser();
        }
    }, [user]);

    // GSAP Animation
    useEffect(() => {
        if (interviewListRef.current.length > 0) {
            interviewListRef.current.forEach((el, index) => {
                gsap.fromTo(
                    el,
                    { opacity: 0, y: 20 },
                    { opacity: 1, y: 0, duration: 0.5, delay: index * 0.2, ease: 'power3.out' }
                );
            });
        }
    }, [list]);

    return (
        <div>
            <h2 className="text-2xl font-semibold text-purple-400 py-4">Previous Interviews</h2>
            <div className="space-y-4">
                {list && list.length > 0 ? (
                    list.map((interview, index) => (
                        <div 
                            key={index}
                            ref={(el) => (interviewListRef.current[index] = el)}
                        >
                            <InterviewCard interview={interview} />
                        </div>
                    ))
                ) : (
                    <p className="text-gray-500 text-center py-4 pb-80">No interviews available</p>
                )}
            </div>
        </div>
    );
}

export default InterviewList;
