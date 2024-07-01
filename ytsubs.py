import logging
import re
from typing import Union

from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

import s3_cache

logger = logging.getLogger(__name__)
CACHE_NAMESPACE = "ytsubs"
s3_cache.set_max_size(CACHE_NAMESPACE, 200)


def is_youtube_video(url):
    return True if extract_video_id(url) else False


def extract_video_id(url):
    """
    Function to extract the video id from a YouTube URL.
    """

    # Standard YouTube URLs (e.g., "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    match = re.search(r"youtube\.com.*v=([^&]*)", url)
    if match:
        return match.group(1)

    # Shortened YouTube URLs (e.g., "https://youtu.be/dQw4w9WgXcQ")
    match = re.search(r"youtu\.be/([^?/]*)", url)
    if match:
        return match.group(1)

    # Embedded YouTube URLs (e.g., "https://www.youtube.com/embed/dQw4w9WgXcQ")
    match = re.search(r"youtube\.com/embed/([^?/]*)", url)
    if match:
        return match.group(1)

    # Live YouTube URLs (e.g., "https://www.youtube.com/live/dQw4w9WgXcQ")
    match = re.search(r"youtube\.com/live/([^?/]*)", url)
    if match:
        return match.group(1)

    return None


def yt_transcript(url: str) -> Union[str, None]:
    """Function to fetch the transcript of a YouTube video, given the URL."""
    cached_transcript = s3_cache.get_cache(CACHE_NAMESPACE, url)
    if cached_transcript:
        return cached_transcript
    try:
        video_id = extract_video_id(url)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if transcript:
                transcript = " ".join(
                    f"[{segment['start']:.2f}] {segment['text']}"
                    for segment in transcript
                )
                s3_cache.set_cache(CACHE_NAMESPACE, url, transcript)
                return transcript
    except Exception as e:
        logger.error(f"Failed to extract transcript for [{url}].")
        return None
    return "<empty>"
