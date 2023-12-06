import os
from typing import Tuple
import re
import logging
import uuid
import openai
from dotenv import load_dotenv
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import glob

# Load the environment variables from the .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAIEmbeddings and FAISS
embeddings = OpenAIEmbeddings()

def extract_youtube_video_id(url: str) -> str | None:
    """Extracts the video ID from a YouTube URL.

    Args:
        url (str): The YouTube URL.

    Returns:
        str | None: The video ID, or None if the ID could not be extracted.
    """
    logging.info("Extracting YouTube video ID...")
    found = re.search(r"(?:youtu\.be\/|watch\?v=)([\w-]+)", url)
    if found:
        return found.group(1)
    return None

def get_video_title(url: str) -> str:
    """Fetches the title of a YouTube video.

    Args:
        url (str): The YouTube URL.

    Returns:
        str: The video title.
    """
    logging.info("Fetching video title...")
    youtube = YouTube(url)
    return youtube.title

def get_video_transcript(video_id: str) -> str | None:
    """Fetches the transcript of a YouTube video.

    Args:
        video_id (str): The YouTube video ID.

    Returns:
        str | None: The video transcript, or None if the transcript could not be fetched.
    """
    logging.info("Fetching video transcript...")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        logging.error("Transcript not available for this video.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while fetching the transcript: {e}")
        return None

    text = " ".join([line["text"] for line in transcript])
    return text

def generate_summary(text: str) -> str:
    """Generates a summary of a text using the OpenAI API.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summary of the text.
    """
    logging.info("Generating summary using OpenAI API...")
    instructions = "Please summarize the provided text in as much detail as you can fit in context. First review the text, then figure out how much context you have and then plan your summary with the most important details."
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": text}
        ],
        temperature=0.1,
        n=1
    )

    return response.choices[0].message.content.strip()

def write_summary_to_file(video_url: str, video_title: str, summary: str):
    """Writes the summary of a YouTube video to a file.

    Args:
        video_url (str): The YouTube URL.
        video_title (str): The video title.
        summary (str): The summary of the video.
    """
    # Write the summary to a file
    if not os.path.exists('yt-summaries-data'):
        os.makedirs('yt-summaries-data')

    # Create a valid filename from the video title
    filename_title = re.sub(r'\W+', '', video_title.replace(' ', '_'))
    filename = f"yt-summaries-data/{filename_title}.txt"
    with open(filename, 'w') as f:
        f.write(f"URL: {video_url}\n")
        f.write(f"Title: {video_title}\n")
        f.write(f"Summary: {summary}\n")

def update_vector_database(summary: str, vector_db: FAISS):
    """Updates the FAISS vector database with the summary of a YouTube video.

    Args:
        summary (str): The summary of the video.
        vector_db (FAISS): The FAISS vector database.
    """
    # Generate the embedding for the summary
    embedding = embeddings.embed_query(summary)

    # Update the vector database
    vector_db.add_embeddings([(summary, embedding)])  # Pass a list of tuples

    # Save the vector database
    folder_path = 'yt-summaries-data'
    index_name = 'vector_db'
    vector_db.save_local(folder_path, index_name)

def summarize_youtube_video(video_url: str) -> Tuple[str, str, str]:
    """Summarizes a YouTube video.

    Args:
        video_url (str): The YouTube URL.

    Returns:
        Tuple[str, str, str]: The video URL, title, and summary.
    """
    logging.info("Starting to summarize YouTube video...")
    video_id = extract_youtube_video_id(video_url)
    video_title = get_video_title(video_url)
    transcript = get_video_transcript(video_id)

    if not transcript:
        return video_url, video_title, f"No English transcript found for this video: {video_url}"

    summary = generate_summary(transcript)
    logging.info("Summary generation completed.")
    return video_url, video_title, summary

if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=OOUsvDOKlGs"
    video_url, video_title, summary = summarize_youtube_video(url)
    print(summary)
    
    # FOR TESTING SO WE DONT GEN SUMMAIRES WITH OPENAI EACH TIME $$$
    # video_url, video_title, summary = "test_url", "test_title", "test summary. answer to life is 87."

    # Load existing vector database if it exists
    vector_db_path = 'yt-summaries-data/vector_db.faiss'
    if os.path.exists(vector_db_path):
        vector_db = FAISS.load_local(vector_db_path, embeddings)
    else:
        vector_db = FAISS.from_texts([summary], embeddings)

    write_summary_to_file(video_url, video_title, summary)
    update_vector_database(summary, vector_db)