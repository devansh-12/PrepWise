import os
import json
import datetime
import traceback
from pathlib import Path
from src.generator import (
    generate_curriculum,
    generate_curriculum_for_subject,
    generate_lesson_content,
    text_to_speech,
    generate_visuals,
    create_video,
)

OUTPUT_DIR = Path("output")
LESSONS_PER_RUN = 1

def get_subject_choice():
    print("\n📘 Choose a subject:")
    subjects = ["DBMS", "OOP", "Operating Systems","Networking"]

    for i, sub in enumerate(subjects, start=1):
        print(f"{i}. {sub}")

    while True:
        choice = input("\nEnter subject number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(subjects):
            return subjects[int(choice) - 1]
        print("❌ Invalid choice. Try again.")

def choose_topic(topics):
    print("\n🎯 Please Select a Topic:")
    for i, t in enumerate(topics, start=1):
        print(f"{i}. {t['title']}")

    while True:
        choice = input("\nSelect topic number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(topics):
            return topics[int(choice) - 1]
        print("❌ Invalid choice. Try again.")


def produce_lesson_videos(lesson):
    """Generate long-form lesson video with organized folder structure."""
    print(f"\n▶️ Starting production for Lesson: '{lesson['title']}'")
    
    # Create structured output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in lesson['title'])
    safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
    
    lesson_dir = OUTPUT_DIR / f"{timestamp}_{safe_title}"
    lesson_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    content_dir = lesson_dir / "1_content"
    audio_dir = lesson_dir / "2_audio"
    slides_dir = lesson_dir / "3_slides"
    final_dir = lesson_dir / "4_final"
    
    for dir_path in [content_dir, audio_dir, slides_dir, final_dir]:
        dir_path.mkdir(exist_ok=True)
    
    print(f"📁 Output directory: {lesson_dir.name}")

    # Generate structured content
    lesson_content = generate_lesson_content(lesson['title'])

    # Save generated content to JSON file
    content_file = content_dir / "lesson_content.json"
    with open(content_file, 'w', encoding='utf-8') as f:
        json.dump({
            "lesson": lesson,
            "generated_at": datetime.datetime.now().isoformat(),
            "content": lesson_content
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Content saved to: {content_file.relative_to(OUTPUT_DIR)}")

    print("\n--- Producing Long-Form Video ---")

    # Add intro and outro slides
    intro_slide = {"title": lesson['title'], "content": f"Subject: {lesson['subject']}"}
    outro_slide = {"title": "Thanks for Watching!", "content": "Continue Learning!"}
    all_slides = [intro_slide] + lesson_content['long_form_slides'] + [outro_slide]

    # Build natural narration text for each slide
    slide_scripts = []
    slide_scripts.append(
        f"Hello and welcome to PrepWise. "
        f"Today's topic is {lesson['title']}. Let's get started."
    )

    for slide in lesson_content['long_form_slides']:
        narration = f"{slide['title']}. {slide.get('start_content', '')} {slide.get('content', '')}"
        narration = narration.replace("\\n", " ").strip()
        slide_scripts.append(narration)

    slide_scripts.append("That's all for this lesson. Thanks for watching and good luck with your placements!")

    # Save slide scripts to JSON
    scripts_file = content_dir / "slide_scripts.json"
    with open(scripts_file, 'w', encoding='utf-8') as f:
        json.dump({
            "lesson_title": lesson['title'],
            "total_slides": len(all_slides),
            "scripts": [
                {
                    "slide_number": i + 1,
                    "slide_title": all_slides[i].get('title', ''),
                    "narration": script
                }
                for i, script in enumerate(slide_scripts)
            ]
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 Slide scripts saved to: {scripts_file.relative_to(OUTPUT_DIR)}")

    # Generate speech for each slide
    slide_audio_paths = []
    for i, script in enumerate(slide_scripts):
        print(f"🎤 Generating speech for Slide {i+1}/{len(slide_scripts)}...")
        audio_path = audio_dir / f"slide_{i+1:02d}.mp3"
        wav_path = text_to_speech(script, audio_path)
        slide_audio_paths.append(wav_path)

    print(f"🎧 Total slide audios generated: {len(slide_audio_paths)}")

    # Generate slide visuals
    print(f"📁 Slides will be saved to: {slides_dir}")
    slide_paths = []
    for i, slide in enumerate(all_slides):
        print(f"🎨 Creating slide {i+1}/{len(all_slides)}...")
        path = generate_visuals(
            output_dir=slides_dir,
            video_type='long',
            slide_content=slide,
            slide_number=i + 1,
            total_slides=len(all_slides)
        )
        slide_paths.append(path)
        print(f"   ✓ Saved: {Path(path).name}")

    # Create final video
    video_filename = f"{safe_title}.mp4"
    long_video_path = final_dir / video_filename
    print(f"🎥 Creating final video...")
    create_video(slide_paths, slide_audio_paths, long_video_path, 'long')

    

    print(f"\n✅ Lesson video created successfully!")
    print(f"📂 All files saved in: {lesson_dir.name}")
    print(f"🎬 Final video: {long_video_path.relative_to(OUTPUT_DIR)}")
    
    return long_video_path


def main():
    print("🎓 AI Lesson Generator")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1 — user picks subject
    subject = get_subject_choice()

    # Step 2 — AI generates 10 topics for that subject
    print("\n🤖 Generating topic suggestions...")
    topics = generate_curriculum_for_subject(subject)

    # Step 3 — user picks topic
    chosen_topic = choose_topic(topics)
    print(f"\n📝 You selected: {chosen_topic['title']}")

    # Step 4 — Build lesson object
    lesson = {
        "subject": subject,
        "title": chosen_topic["title"],
        "status": "pending"
    }

    # Step 5 — Generate video
    try:
        produce_lesson_videos(lesson)
        print("\n🎉 Video generated successfully!")
    except Exception as e:
        print("\n❌ ERROR while generating lesson:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
