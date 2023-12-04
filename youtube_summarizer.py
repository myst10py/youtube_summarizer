import os
import re
import logging

import openai
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# Load the environment variables from the .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_youtube_video_id(url: str) -> str | None:
    logging.info("Extracting YouTube video ID...")
    found = re.search(r"(?:youtu\.be\/|watch\?v=)([\w-]+)", url)
    if found:
        return found.group(1)
    return None

def get_video_transcript(video_id: str) -> str | None:
    logging.info("Fetching video transcript...")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        logging.error("Transcript not available for this video.")
        return None

    text = " ".join([line["text"] for line in transcript])
    return text

def generate_summary(text: str) -> str:
    logging.info("Generating summary using OpenAI API...")
    instructions = "Please summarize the provided text in detail. Extract the recipe and output the ingredients and how much (i.e. 1 onion, 2 boxes of chicken stock this is important) of each is needed and the directions in detail. Also output a summary of how the chef prepares the ingredients including how to cut or chop certain ingredients."
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        n=1,
        max_tokens=4096,
        presence_penalty=0,
        frequency_penalty=0.1,
    )

    return response.choices[0].message.content.strip()

def summarize_youtube_video(video_url: str) -> str:
    logging.info("Starting to summarize YouTube video...")
    video_id = extract_youtube_video_id(video_url)
    transcript = get_video_transcript(video_id)

    if not transcript:
        return f"No English transcript found for this video: {video_url}"

    summary = generate_summary(transcript)
    logging.info("Summary generation completed.")
    return summary

if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=DcC2e-q5_-Y"
    print(summarize_youtube_video(url))