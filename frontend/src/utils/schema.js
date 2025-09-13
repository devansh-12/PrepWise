import { varchar, text } from "drizzle-orm/pg-core";

import { pgTable, serial, varchar, text, timestamp, integer } from "drizzle-orm/pg-core";


export const MockInterview = pgTable('mockInterview', {
    id: serial('id').primaryKey(),
    jsonMockResp: text('jsonMockResp').notNull(),
    jobPosition: varchar('jobPosition').notNull(),
    jobDesc: varchar('jobDesc').notNull(),
    jobExperience: varchar('jobExperience').notNull(),
    createdBy: varchar('createdBy').notNull(),
    createdAt: varchar('createdAt').notNull(),
    mockId: varchar('mockId').notNull()
});

export const UserAnswer = pgTable('userAnswer', {
    id: serial('id').primaryKey(),
    mockIdRef: varchar('mockId').notNull(), 
    question: varchar('question').notNull(),
    correctAns: text('correctAns'),
    userAns: text('userAns'),
    feedback: text('feedback'),
    rating: varchar('rating'),
    userEmail: varchar('userEmail'),
    createdAt: varchar('createdAt')
});

export const allowedUsers = pgTable('allowedUsers', {
    email: varchar('email').primaryKey(),
    status: varchar('status')
});


// -------------------- Users --------------------
export const users = pgTable("users", {
  user_id: serial("user_id").primaryKey(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  name: varchar("name", { length: 255 }),
  created_at: timestamp("created_at").defaultNow().notNull(),
});

// -------------------- Candidate Profiles --------------------
export const candidateProfiles = pgTable("candidate_profiles", {
  id: serial("id").primaryKey(),
  candidate_id: integer("candidate_id")
    .notNull()
    .references(() => users.user_id), // foreign key to users
  created_at: timestamp("created_at").defaultNow().notNull(),
});

// -------------------- Resumes --------------------
export const resumes = pgTable("resumes", {
  id: serial("id").primaryKey(),
  userId: varchar("userId").notNull(), // Clerk's user.id
  filePath: text("filePath").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
})
