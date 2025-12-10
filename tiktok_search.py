#!/usr/bin/env python3
"""
TikTok Video Search Agent using AutoGLM

Given a keyword, searches TikTok and extracts 3 related videos.
"""

import argparse
import os
import sys
import time

# Add phone_agent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.model import ModelConfig


def search_tiktok_videos(keyword: str, num_videos: int = 3) -> list:
    """
    Search TikTok for videos related to the keyword.

    Args:
        keyword: Search keyword
        num_videos: Number of videos to collect (default: 3)

    Returns:
        List of video information
    """
    # Configure model
    model_config = ModelConfig(
        base_url=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        model_name=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
    )

    # Configure agent
    agent_config = AgentConfig(
        max_steps=50,
        verbose=True,
        lang="en",  # Use English prompts
    )

    # Create agent
    agent = PhoneAgent(
        model_config=model_config,
        agent_config=agent_config,
    )

    # Task: Open TikTok, search for keyword, and collect video info
    task = f"""
    Please perform the following steps:
    1. Open the TikTok app
    2. Tap on the search icon
    3. Enter the search keyword: "{keyword}"
    4. Wait for search results to load
    5. For the first {num_videos} videos shown:
       - Note the video title/caption
       - Note the creator username
       - Note the number of likes if visible
    6. Report back the information for all {num_videos} videos

    Important: Do not tap on any video, just gather the visible information from the search results.
    """

    print(f"\n{'='*60}")
    print(f"TikTok Video Search")
    print(f"{'='*60}")
    print(f"Keyword: {keyword}")
    print(f"Number of videos: {num_videos}")
    print(f"{'='*60}\n")

    # Run the agent
    result = agent.run(task)

    print(f"\n{'='*60}")
    print("Search Results:")
    print(f"{'='*60}")
    print(result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Search TikTok videos by keyword")
    parser.add_argument("keyword", type=str, help="Search keyword")
    parser.add_argument(
        "-n", "--num-videos",
        type=int,
        default=3,
        help="Number of videos to collect (default: 3)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Model API base URL"
    )

    args = parser.parse_args()

    # Set environment variable if provided
    if args.base_url:
        os.environ["PHONE_AGENT_BASE_URL"] = args.base_url

    # Run search
    result = search_tiktok_videos(args.keyword, args.num_videos)

    return result


if __name__ == "__main__":
    main()
