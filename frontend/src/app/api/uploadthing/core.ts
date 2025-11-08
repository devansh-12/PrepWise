// src/app/api/uploadthing/core.ts
import { createUploadthing, type FileRouter } from "uploadthing/next";

const f = createUploadthing();

export const ourFileRouter = {
  resumeUploader: f({
    pdf: { maxFileSize: "8MB" }, // PDFs are directly supported
    blob: { maxFileSize: "8MB" }, // use blob for doc/docx
  })
    .onUploadComplete(async ({ file }) => {
      console.log("File uploaded:", file.ufsUrl);
      return { url: file.ufsUrl };
    }),
  videoUploader: f({
  video: { maxFileSize: "128MB" }, // add video support
}).onUploadComplete(async ({ file }) => {
  console.log("Video uploaded:", file.ufsUrl);
  return { url: file.ufsUrl };
}),
} satisfies FileRouter;



export type OurFileRouter = typeof ourFileRouter;
