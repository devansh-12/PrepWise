import { createUploadthing, type FileRouter } from "uploadthing/next";

const f = createUploadthing();

export const ourFileRouter = {
  resumeUploader: f({
    pdf: { maxFileSize: "8MB" },
    blob: { maxFileSize: "8MB" },
  })
    .onUploadComplete(async ({ file }) => {
      console.log("Resume uploaded:", file.url);
      // ❗ v5: do NOT return anything
    }),

  videoUploader: f({
    blob: { maxFileSize: "64MB" },
  })
    .onUploadComplete(async ({ file }) => {
      console.log("Video uploaded:", file.url);
      // ❗ v5: do NOT return anything
    }),
} satisfies FileRouter;

export type OurFileRouter = typeof ourFileRouter;
