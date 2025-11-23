# FILE: src/generator.py

import os
import json
import requests
from io import BytesIO
import google.generativeai as genai
from moviepy.editor import AudioFileClip, ImageClip, CompositeAudioClip, concatenate_videoclips, vfx
from moviepy.config import change_settings
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from pydub import AudioSegment
import re
import json
import edge_tts
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 


def safe_parse_json(text):
    """
    Cleans and parses the LLM JSON safely.
    Handles invalid escapes, Markdown, and extra text.
    """
    # Remove Markdown code fences
    text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Replace bad escape sequences
    text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    # Extract the first valid JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("❌ No valid JSON object found in response.")
    text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed at {e}. Saving raw output to debug_failed_json.txt.")
        with open("debug_failed_json.txt", "w", encoding="utf-8") as f:
            f.write(text)
        raise



# --- Configuration ---
ASSETS_PATH = Path("assets")
FONT_FILE = ASSETS_PATH / "fonts/arial.ttf"
BACKGROUND_MUSIC_PATH = ASSETS_PATH / "music/bg_music.mp3"
FALLBACK_THUMBNAIL_FONT = ImageFont.load_default()


# GitHub Actions compatibility for ImageMagick
if os.name == 'posix':
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})


def get_pexels_image(query, video_type):
    """Searches for a relevant image on Pexels and returns the image object."""
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    if not pexels_api_key:
        # print("⚠️ PEXELS_API_KEY not found. Using solid color background.")
        return None

    orientation = 'landscape' if video_type == 'long' else 'portrait'
    try:
        headers = {"Authorization": pexels_api_key}
        params = {"query": f"abstract {query}", "per_page": 1, "orientation": orientation}
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('photos'):
            image_url = data['photos'][0]['src']['large2x']
            image_response = requests.get(image_url, timeout=15)
            image_response.raise_for_status()
            return Image.open(BytesIO(image_response.content)).convert("RGBA")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching Pexels image for query '{query}': {e}")
    except Exception as e:
        print(f"❌ General error fetching Pexels image for query '{query}': {e}")
    return None


async def voice_agent(text, output_file):
    voice = "en-US-JennyNeural"
    # voice = "en-US-NeerjaNeural"
    

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
    )

    await communicate.save(output_file)

def text_to_speech(text, output_path):
    
    output_path = Path(output_path).with_suffix(".mp3")
    asyncio.run(voice_agent(text, str(output_path)))
    print(f"✅ Generated voice: {output_path}")
    return output_path


def generate_curriculum_for_subject(subject):
    """Generate 10 interview topics ONLY for the selected subject."""
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    prompt = f"""
    Generate a list of exactly 10 high–value placement interview topics
    ONLY for the subject: {subject}

    Keep titles short and focused, for example:
    - DBMS: Indexing & B-Tree Search
    - OS: Deadlocks and Detection
    - OOP: Polymorphism in Depth

    Return valid JSON:
    {{
      "lessons": [
        {{"title": "..." }}
      ]
    }}
    """

    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "")
    data = json.loads(text)
    return data["lessons"]


def generate_curriculum(previous_titles=None):
    """Generates the entire course curriculum using Gemini."""
    print("🤖 No content plan found. Generating a new curriculum from scratch...")
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # model = genai.GenerativeModel('gemini-1.5-flash')
        # model = genai.GenerativeModel('gemini-pro')
        model = genai.GenerativeModel('models/gemini-2.5-pro')

        #Optional: Add prior lesson titles for continuation
        history = ""
        if previous_titles:
            formatted = "\n".join([f"{i+1}. {t}" for i, t in enumerate(previous_titles)])
            history = f"The following lessons have already been created:\n{formatted}\n\nPlease continue from where this series left off.\n"

        prompt = """
        You are an expert computer science educator creating a focused curriculum
        for students preparing for **Placement Technical Round Interviews**.

        Focus strictly on these core subjects:
        - Database Management Systems (DBMS)
        - Object Oriented Programming (OOP)
        - Operating Systems (OS)

        Avoid overlapping topics. Each lesson must target **one high-value interview topic**.
        Make titles short and specific, like “DBMS: Normalization and Keys (1NF–3NF)”.

        Return only a valid JSON object like this:
        {
        "lessons": [
            {
            "subject": "Database Management Systems",
            "title": "DBMS: Normalization and Keys (1NF–3NF)",
            "status": "pending",
            }
        ]
        }

        Generate exactly **15 lessons total**: 5 for DBMS, 5 for OOP, and 5 for OS.
        """


        response = model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        curriculum = json.loads(json_string)
        print("✅ New curriculum generated successfully!")
        return curriculum
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to generate curriculum. {e}")
        raise


def generate_lesson_content(lesson_title):
    """Generates the content for one long-form lesson and its promotional short."""
    print(f"🤖 Generating content for lesson: '{lesson_title}'...")
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        # model = genai.GenerativeModel('gemini-1.5-flash')
        #model = genai.GenerativeModel('gemini-1.5-flash')
        model = genai.GenerativeModel("models/gemini-2.5-flash")

        prompt = f"""
        You are creating a short, structured interview-prep lesson for the YouTube series
        **'Placement Prep'**.

        🎯 Topic: **{lesson_title}**

        Write engaging content suitable for a **5-slide explainer video** that helps students
        understand technical concepts for placement interviews.

        Return only valid JSON in this format:
        {{
        "long_form_slides": [
            {{
                "title": "Short slide title (2–6 words)",
                "start_content": "One short introductory line (like: Let's understand how this works.)",
                "content": "A short, smooth-flowing paragraph (3–5 sentences) explaining the concept in simple words."
            }}
        ]
        }}

        Guidelines:
        - Generate exactly **5 slides**.
        - Each slide’s `content` should be a **natural explanation**, not bullets.
        - Maintain a **spoken tone** as if the instructor is explaining to a student.
        - Add smooth transitions like “next”, “for example”, “this means”, or “in short”.
        - Keep each slide under **80 words** so it fits easily into a voice-over.
        - Include at least **one practical example or question** across the slides.
        - Avoid markdown, lists, or code blocks — use plain sentences only.
        - Focus on topics like DBMS, OOP, and Operating Systems, suitable for placement technical rounds.
        """



        response = model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        # --- Replace this line ---
        # content = json.loads(json_string)
        # --- with this: ---
        content = safe_parse_json(json_string)
        print("✅ Lesson content generated successfully.")
        return content
    except Exception as e:
        print(f"❌ ERROR: Failed to generate lesson content: {e}")
        raise


def generate_visuals(output_dir, video_type, slide_content=None, thumbnail_title=None, slide_number=0, total_slides=0):
    """Generates a single professional, PPT-style slide or a thumbnail with corrected alignment."""
    
    # Ensure output_dir is a Path object and exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    is_thumbnail = thumbnail_title is not None

    width, height = (1920, 1080) if video_type == 'long' else (1080, 1920)
    title = thumbnail_title if is_thumbnail else slide_content.get("title", "")
    bg_image = get_pexels_image(title, video_type)

    if not bg_image:
        bg_image = Image.new('RGBA', (width, height), color=(12, 17, 29))
    bg_image = bg_image.resize((width, height)).filter(ImageFilter.GaussianBlur(5))
    darken_layer = Image.new('RGBA', bg_image.size, (0, 0, 0, 150))
    final_bg = Image.alpha_composite(bg_image, darken_layer).convert("RGB")

    if is_thumbnail and video_type == 'long':
        w, h = final_bg.size
        if h > w:
            print("⚠️ Detected vertical thumbnail for long video. Rotating and resizing to 1920x1080...")
            final_bg = final_bg.transpose(Image.ROTATE_270).resize((1920, 1080))

    draw = ImageDraw.Draw(final_bg)

    try:
        title_font = ImageFont.truetype(str(FONT_FILE), 80 if video_type == 'long' else 90)
        content_font = ImageFont.truetype(str(FONT_FILE), 45 if video_type == 'long' else 55)
        footer_font = ImageFont.truetype(str(FONT_FILE), 25 if video_type == 'long' else 35)
    except IOError:
        title_font = content_font = footer_font = FALLBACK_THUMBNAIL_FONT

    if not is_thumbnail:
        # Header background
        header_height = int(height * 0.18)
        draw.rectangle([0, 0, width, header_height], fill=(25, 40, 65, 200))

        # Wrap title text if needed
        words = title.split()
        title_lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=title_font)
            if bbox[2] - bbox[0] < width * 0.9:
                current_line = test_line
            else:
                title_lines.append(current_line)
                current_line = word
        title_lines.append(current_line)

        # Center vertically in header
        line_height = title_font.getbbox("A")[3] + 10
        total_title_height = len(title_lines) * line_height
        y_text = (header_height - total_title_height) / 2

        for line in title_lines:
            bbox = draw.textbbox((0, 0), line, font=title_font)
            x = (width - (bbox[2] - bbox[0])) / 2
            draw.text((x, y_text), line, font=title_font, fill=(255, 255, 255))
            y_text += line_height
    else:
        # Center title on thumbnail
        bbox = draw.textbbox((0, 0), title, font=title_font)
        x = (width - (bbox[2] - bbox[0])) / 2
        y = (height - (bbox[3] - bbox[1])) / 2
        draw.text((x, y), title, font=title_font, fill=(255, 255, 255), stroke_width=2, stroke_fill="black")

    if not is_thumbnail:
        # Main content block
        content = slide_content.get("content", "")
        is_special_slide = len(content.split()) < 10

        words = content.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if draw.textbbox((0, 0), test_line, font=content_font)[2] < width * 0.85:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        line_height = content_font.getbbox("A")[3] + 15
        total_text_height = len(lines) * line_height
        y_text = (height - total_text_height) / 2 if is_special_slide else header_height + 100

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=content_font)
            x = (width - (bbox[2] - bbox[0])) / 2
            draw.text((x, y_text), line, font=content_font, fill=(230, 230, 230))
            y_text += line_height

        # Footer
        footer_height = int(height * 0.06)
        draw.rectangle([0, height - footer_height, width, height], fill=(25, 40, 65, 200))
        draw.text((40, height - footer_height + 12), f"Placement prep", font=footer_font, fill=(180, 180, 180))

        if total_slides > 0:
            slide_num_text = f"Slide {slide_number} of {total_slides}"
            bbox = draw.textbbox((0, 0), slide_num_text, font=footer_font)
            draw.text((width - bbox[2] - 40, height - footer_height + 12), slide_num_text, font=footer_font, fill=(180, 180, 180))

    file_prefix = "thumbnail" if is_thumbnail else f"slide_{slide_number:02d}"
    path = output_dir / f"{file_prefix}.png"
    final_bg.save(path)
    return str(path)

def create_video(slide_paths, audio_paths, output_path, video_type):
    """Creates a final video from slides and per-slide audio clips with optional background music."""
    print(f"🎬 Creating {video_type} video...")
    try:
        # --- ensure .mp4 extension ---
        output_path = Path(output_path).with_suffix(".mp4")

        if not slide_paths or not audio_paths or len(slide_paths) != len(audio_paths):
            raise ValueError("Mismatch between slides and audio clips, or no slides provided.")

        image_clips = []
        for i, (img_path, audio_path) in enumerate(zip(slide_paths, audio_paths)):
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration + 0.5  # Padding
            img_clip = (
                ImageClip(img_path)
                .set_duration(duration)
                .set_audio(audio_clip)
                .fadein(0.5)
                .fadeout(0.5)
            )
            image_clips.append(img_clip)

        final_video = concatenate_videoclips(image_clips, method="compose")

        if BACKGROUND_MUSIC_PATH.exists():
            print("🎵 Adding background music...")
            bg_music = AudioFileClip(str(BACKGROUND_MUSIC_PATH)).volumex(0.15)
            if bg_music.duration < final_video.duration:
                bg_music = bg_music.fx(vfx.loop, duration=final_video.duration)
            else:
                bg_music = bg_music.subclip(0, final_video.duration)

            composite_audio = CompositeAudioClip([
                final_video.audio.volumex(1.2),
                bg_music
            ])
            final_video = final_video.set_audio(composite_audio)

        final_video.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="medium",
            threads=4
        )
        print(f"✅ {video_type.capitalize()} video created successfully! Saved as {output_path.name}")

    except Exception as e:
        print(f"❌ ERROR during video creation: {e}")
        raise
