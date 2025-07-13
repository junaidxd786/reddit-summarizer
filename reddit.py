import os
import praw
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for # Changed to render_template
from dotenv import load_dotenv
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import threading
import uuid
from flask_caching import Cache
import random
import subprocess
import json # Added for parsing JSON from Gemini
from praw.models import Comment
import time
from enum import Enum
from datetime import datetime, timedelta
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

load_dotenv()
startup_start = time.time()

# Circuit Breaker Implementation
class CircuitState(Enum):
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if service is back

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()

    def _should_attempt_reset(self):
        if self.last_failure_time is None:
            return False
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed > self.recovery_timeout

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit breaker is OPEN. Service unavailable. Last failure: {self.last_failure_time}")
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def get_status(self):
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }

# Initialize circuit breakers
reddit_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=120,  # 2 minutes
    expected_exception=Exception
)

gemini_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=300,  # 5 minutes (Gemini rate limits are usually longer)
    expected_exception=Exception
)

# Multiple Gemini API keys support
GEMINI_API_KEYS = []
primary_key = os.getenv("GEMINI_API_KEY")
if primary_key:
    GEMINI_API_KEYS.append(primary_key)

# Add additional keys if they exist
for i in range(1, 6):  # Support up to 5 additional keys
    additional_key = os.getenv(f"GEMINI_API_KEY_{i}")
    if additional_key:
        GEMINI_API_KEYS.append(additional_key)

current_key_index = 0

def get_next_gemini_key():
    """Get the next available Gemini API key, cycling through available keys."""
    global current_key_index
    if not GEMINI_API_KEYS:
        return None

    key = GEMINI_API_KEYS[current_key_index]
    logging.info(f"[GEMINI] Using API key index {current_key_index}")
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLI_MODEL = os.getenv("GEMINI_CLI_MODEL", "gemini-2.5-pro")

# Validate environment variables
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
    logging.error("Missing Reddit API credentials. Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your .env file.")
    # Optionally: sys.exit(1)
if not GEMINI_API_KEY:
    logging.error("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
    # Optionally: sys.exit(1)

# Lazy initialization for PRAW

def get_reddit():
    t0 = time.time()
    if not hasattr(get_reddit, "instance"):
        import praw
        get_reddit.instance = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        print("get_reddit() cold init time:", time.time() - t0)
    return get_reddit.instance

# Lazy initialization for SentimentIntensityAnalyzer

def get_sentiment_analyzer():
    t0 = time.time()
    if not hasattr(get_sentiment_analyzer, "instance"):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        get_sentiment_analyzer.instance = SentimentIntensityAnalyzer()
        print("get_sentiment_analyzer() cold init time:", time.time() - t0)
    return get_sentiment_analyzer.instance

# Remove top-level PRAW initialization
# try:
#     reddit = praw.Reddit(
#         client_id=REDDIT_CLIENT_ID,
#         client_secret=REDDIT_CLIENT_SECRET,
#         user_agent=REDDIT_USER_AGENT
#     )
#     logging.info("PRAW initialized successfully.")
# except Exception as e:
#     logging.error(f"Failed to initialize PRAW: {e}. Check your Reddit API credentials and user agent.")
#     reddit = None


app = Flask(__name__)
# Configure cache (SimpleCache for development, consider RedisCache for production)
app.config['CACHE_TYPE'] = 'SimpleCache'
# app.config['CACHE_TYPE'] = 'RedisCache' # Uncomment for Redis
# app.config['CACHE_REDIS_HOST'] = 'localhost' # Configure for Redis
# app.config['CACHE_REDIS_PORT'] = 6379 # Configure for Redis
# app.config['CACHE_REDIS_DB'] = 0 # Configure for Redis
cache = Cache(app) # Initialize cache with app config
logging.info("Flask app and cache initialized.")

# Set up error logging
logging.basicConfig(filename='app_errors.log', level=logging.ERROR, format='%(asctime)s %(levelname)s:%(message)s')

def remove_urls(text):
    """Removes URLs from the given text."""
    return re.sub(r'https?://\S+', '', text)

def get_random_trending_thread_url():
    """Fetches a random trending Reddit thread URL."""
    reddit = get_reddit()
    if not reddit: # Check if PRAW was initialized
        logging.error("Reddit PRAW not initialized, cannot fetch trending threads.")
        return None
    
    def _fetch_trending_thread():
        reddit = get_reddit()
        if not reddit:
            raise Exception("Reddit PRAW not initialized")
        subreddit = reddit.subreddit("popular")
        # Fetch fewer posts for quicker response, and filter as before
        posts = [post for post in subreddit.hot(limit=25) if not post.stickied and post.num_comments > 5]
        if not posts:
            logging.warning("No suitable trending posts found.")
            return None
        post = random.choice(posts)
        return f"https://www.reddit.com{post.permalink}"
    
    try:
        return reddit_circuit_breaker.call(_fetch_trending_thread)
    except Exception as e:
        logging.error(f"Circuit breaker prevented Reddit API call: {e}")
    return None

def fetch_post_and_comments(thread_url, limit=5, sort='top'):
    """Fetches the main post text and the most relevant comments from a Reddit thread.
    sort: 'top' (most upvoted), 'best', or 'controversial'"""
    reddit = get_reddit()
    if not reddit: # Check if PRAW was initialized
        raise ValueError("Reddit PRAW not initialized, cannot fetch post and comments.")

    def _fetch_post_and_comments():
        reddit = get_reddit()
        if not reddit:
            raise Exception("Reddit PRAW not initialized")
        import praw
        from praw.models import Comment
        submission = reddit.submission(url=thread_url)
        # Set the comment sort order
        if sort == 'best':
            submission.comment_sort = 'best'
        elif sort == 'controversial':
            submission.comment_sort = 'controversial'
        else:
            submission.comment_sort = 'top'
        # Only fetch one batch of 'more comments' for speed
        submission.comments.replace_more(limit=1)
        comments = []
        for comment in submission.comments.list():
            if isinstance(comment, Comment):
                comments.append((comment.body, getattr(comment, 'score', 0)))
        # Sort by score for 'top', otherwise keep order as returned by Reddit
        if sort == 'top':
            comments = sorted(comments, key=lambda x: x[1], reverse=True)[:limit]
        else:
            comments = comments[:limit]
        post_text = submission.title + "\n\n" + (submission.selftext or "")
        return post_text, comments
    try:
        return reddit_circuit_breaker.call(_fetch_post_and_comments)
    except Exception as e:
        logging.error(f"Circuit breaker prevented Reddit API call for {thread_url}: {e}")
        raise

def gemini_api_process(text_prompt, model=GEMINI_CLI_MODEL, api_key=None):
    """
    Calls the Gemini API to process a text prompt and returns the output.
    Supports multiple API keys with automatic rotation on rate limits.
    """
    if not api_key:
        api_key = get_next_gemini_key()
    if not api_key:
        return "Error: No Gemini API keys available.", True

    def _call_gemini_api():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": text_prompt}
                    ]
                }
            ]
        }

        response = requests.post(url, headers=headers, json=data, timeout=60)
        # Check for rate limit or quota exceeded
        if response.status_code == 429 or response.status_code == 403:
            error_text = response.text.lower()
            if "quota" in error_text or "rate" in error_text or "limit" in error_text:
                # Try with next API key
                next_key = get_next_gemini_key()
                if next_key and next_key != api_key:
                    logging.info(f"Rate limit hit with key {api_key[:10]}..., trying next key")
                    # Recursive call with next key
                    return gemini_api_process(text_prompt, model, next_key)
            else:
                return f"All Gemini API keys have reached their limits. Please try again later.", True
        if response.status_code != 200:
            return f"Gemini API error: {response.text}", True
        result = response.json()
        # Extract the text from the response
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip(), False

    try:
        return gemini_circuit_breaker.call(_call_gemini_api)
    except Exception as e:
        logging.error(f"Circuit breaker prevented Gemini API call: {e}")
        return f"Gemini API call failed due to circuit breaker: {e}", True

def clean_gemini_output(output):
    """Removes common unwanted output from Gemini CLI."""
    # This might need to be adjusted based on the new JSON output format.
    # For now, it will just clean general CLI chatter.
    return output.replace("Loaded cached credentials.", "").strip()

def extract_json_from_markdown(text):
    """
    Extracts JSON from a Markdown code block if present, otherwise returns the text as is.
    Handles malformed JSON by attempting to fix common issues.
    """
    # First try to extract JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # If no code block, try to find JSON in the text
        json_str = text.strip()
    
    # Try to fix common JSON formatting issues
    try:
        # First attempt: parse as-is
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON parsing failed: {e}")
        
        # Second attempt: try to fix common issues
        try:
            # Remove any trailing commas before closing braces/brackets
            fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Fix missing quotes around keys
            fixed_json = re.sub(r'(\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_json)
            # Fix single quotes to double quotes
            fixed_json = fixed_json.replace("'", '"')
            
            json.loads(fixed_json)
            return fixed_json
        except json.JSONDecodeError as e2:
            logging.warning(f"Fixed JSON parsing also failed: {e2}")
            
            # Third attempt: try to extract just the content we need
            try:
                # Try to extract summary, action_items_notes, and key_facts manually
                summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:"[^"]*")*[^"]*)"', json_str, re.DOTALL)
                action_items_match = re.search(r'"action_items_notes"\s*:\s*"([^"]*(?:"[^"]*")*[^"]*)"', json_str, re.DOTALL)
                key_facts_match = re.search(r'"key_facts"\s*:\s*"([^"]*(?:"[^"]*")*[^"]*)"', json_str, re.DOTALL)
                
                if summary_match or action_items_match or key_facts_match:
                    manual_json = {
                        "summary": summary_match.group(1) if summary_match else "Summary not found.",
                        "action_items_notes": action_items_match.group(1) if action_items_match else "No action items found.",
                        "key_facts": key_facts_match.group(1) if key_facts_match else "No key facts found."
                    }
                    return json.dumps(manual_json)
            except Exception as e3:
                logging.error(f"Manual JSON extraction failed: {e3}")
    
    # If all parsing attempts fail, return a fallback structure
    logging.error(f"All JSON parsing attempts failed. Raw output: {text[:500]}...")
    fallback_json = {
        "summary": "Error: Could not parse Gemini response properly. The summary may be incomplete.",
        "action_items_notes": "No action items could be extracted due to parsing error.",
        "key_facts": "No key facts could be extracted due to parsing error."
    }
    return json.dumps(fallback_json)

def process_thread_with_gemini(combined_text, language="en", summary_length="medium"):
    """
    Processes the entire thread using one Gemini API call to get summary,
    action items, and key facts in a structured format.
    """
    model = os.getenv("GEMINI_CLI_MODEL", "gemini-2.5-pro")
    text_no_urls = remove_urls(combined_text)
    truncated_text = safe_truncate(text_no_urls, 8000)

    summary_directive = ""
    if summary_length == "short":
        summary_directive = "Provide a very brief (1-2 sentences) summary. Focus on the main idea and general consensus."
    elif summary_length == "detailed":
        summary_directive = "Provide a detailed (2-3 paragraphs) summary. Cover the main idea, key points, and range of opinions."
    else: # medium
        summary_directive = "Provide a concise summary (around 150 words). Focus on the main idea of the original post and briefly outline the general consensus or key differing opinions in comments."

    prompt = f"""
As an AI assistant, analyze the following Reddit thread and provide the requested information in a VALID JSON format.
The JSON object must have exactly these keys:
- "summary": A plain English summary based on the length requested.
- "action_items_notes": A bulleted list of 3-5 main action items and important notes. Use * or - for bullet points, one per line.
- "key_facts": A bulleted list of 3-5 key facts. Use * or - for bullet points, one per line.

IMPORTANT: Ensure the JSON is properly formatted with:
- All strings properly quoted
- No trailing commas
- Valid JSON syntax
- Proper escaping of quotes within strings

Here is the Reddit Thread:
---
{truncated_text}
---

Instructions:
1. Summary: {summary_directive}
2. Action Items & Notes: List concise bullet points using * or - format, one per line.
3. Key Facts: List concise bullet points using * or - format, one per line.
4. Language: All output should be in {language} if possible, otherwise default to English.
5. JSON Format: Return ONLY valid JSON, no additional text or markdown formatting.

JSON Output:
"""

    raw_result, is_error = gemini_api_process(prompt, model=model, api_key=GEMINI_API_KEY)

    if is_error:
        return {"summary": raw_result, "action_items_notes": "", "key_facts": ""}, True

    cleaned_result = clean_gemini_output(raw_result)
    json_str = extract_json_from_markdown(cleaned_result)
    try:
        parsed_output = json.loads(json_str)
        summary = parsed_output.get("summary", "Summary not found.")
        action_items_notes = parsed_output.get("action_items_notes", "No action items found.")
        key_facts = parsed_output.get("key_facts", "No key facts found.")

        # Debug logging to see what format is being returned
        logging.info(f"Action items format: {repr(action_items_notes[:200])}")
        logging.info(f"Key facts format: {repr(key_facts[:200])}")
        
        return {"summary": summary, "action_items_notes": action_items_notes, "key_facts": key_facts}, False
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Gemini output: {e}")
        logging.error(f"JSON string length: {len(json_str)}")
        logging.error(f"Raw output preview: {cleaned_result[:1000]}...")
        logging.error(f"Extracted JSON preview: {json_str[:1000]}...")
        
        # Try to provide a more helpful error message
        error_msg = f"Error parsing Gemini response: {e}. The AI response may have been malformed."
        return {"summary": error_msg, "action_items_notes": "", "key_facts": ""}, True
    except Exception as e:
        logging.error(f"Unexpected error processing Gemini output: {e}")
        return {"summary": f"An unexpected error occurred: {e}", "action_items_notes": "", "key_facts": ""}, True

# --- Progress tracking for loading screen ---
PROCESSING_STEPS = [
    {'key': 'fetching', 'label': 'Fetching Thread'},
    {'key': 'analyzing', 'label': 'Analyzing Comments'},
    {'key': 'summarizing', 'label': 'Generating Summary'},
    {'key': 'extracting', 'label': 'Extracting Insights'},
    {'key': 'finalizing', 'label': 'Finalizing'},
]

def set_progress(job_id, step_idx, extra_label=None):
    step = PROCESSING_STEPS[min(step_idx, len(PROCESSING_STEPS)-1)]
    progress = int((step_idx+1) / len(PROCESSING_STEPS) * 100)
    label = step['label']
    if extra_label:
        label = f"{label} ({extra_label})"
    cache.set(f"progress_{job_id}", {
        'step': step_idx+1,
        'total_steps': len(PROCESSING_STEPS),
        'label': label,
        'progress': progress
    }, timeout=600)

def safe_truncate(text, max_chars=4000):
    """Truncates text to a safe character limit."""
    return text[:max_chars]

def analyze_sentiment(comments):
    """Analyzes the sentiment of comments and returns score, breakdown, and verdict."""
    analyzer = get_sentiment_analyzer()
    scores = [analyzer.polarity_scores(text[0])['compound'] for text in comments if text[0].strip()]
    avg_score = sum(scores) / len(scores) if scores else 0
    # Breakdown
    pos = sum(1 for s in scores if s > 0.15)
    neg = sum(1 for s in scores if s < -0.15)
    neu = len(scores) - pos - neg
    total = len(scores) if scores else 1
    breakdown = {
        'positive': round(pos / total * 100, 1),
        'neutral': round(neu / total * 100, 1),
        'negative': round(neg / total * 100, 1)
    }
    if avg_score > 0.15:
        verdict = "Most people like this post. (Positive sentiment)"
    elif avg_score < -0.15:
        verdict = "Most people dislike this post. (Negative sentiment)"
    else:
        verdict = "The sentiment is mixed. (Neutral sentiment)"
    return {
        'score': round(avg_score, 3),
        'breakdown': breakdown,
        'verdict': verdict
    }

def extract_faqs(comments, max_faqs=5):
    """Extracts potential FAQs from comments based on questions and upvotes."""
    questions = [(body, score) for body, score in comments if body.strip().endswith('?')]
    top_questions = sorted(questions, key=lambda x: x[1], reverse=True)[:max_faqs]
    return top_questions

def extract_expert_opinions(thread_url, limit=50):
    """Extracts comments from users with specific 'expert' flairs."""
    reddit = get_reddit()
    if not reddit:
        logging.error("Reddit PRAW not initialized, cannot extract expert opinions.")
        return []
    def _extract_expert_opinions():
        reddit = get_reddit()
        if not reddit:
            raise Exception("Reddit PRAW not initialized")
        submission = reddit.submission(url=thread_url)
        submission.comments.replace_more(limit=0)
        expert_keywords = ["expert", "phd", "dr", "professor", "official", "mod"]
        expert_comments = []
        for comment in submission.comments.list()[:limit]:
            if isinstance(comment, Comment):
                flair = (getattr(comment.author_flair_text, 'lower', lambda: "")() if comment.author_flair_text else "")
                if any(keyword in flair for keyword in expert_keywords):
                    expert_comments.append((comment.body, getattr(comment, 'score', 0), flair))
        expert_comments = sorted(expert_comments, key=lambda x: x[1], reverse=True)
        return expert_comments
    try:
        return reddit_circuit_breaker.call(_extract_expert_opinions)
    except Exception as e:
        logging.error(f"Circuit breaker prevented Reddit API call for expert opinions: {e}")
        return []

def extract_funny_comments(comments, max_funny=3):
    """Extracts comments likely intended to be funny."""
    funny_keywords = ["lol", "funny", "hilarious", "lmao", "rofl", "haha", "xd", "lmfao", "hehe", "humor", "joke"]
    funny_comments = [(body, score) for body, score in comments if any(kw in body.lower() for kw in funny_keywords)]
    top_funny = sorted(funny_comments, key=lambda x: x[1], reverse=True)[:max_funny]
    return top_funny

def extract_links(texts):
    """Extracts all URLs from a list of texts."""
    url_pattern = re.compile(r'https?://\S+')
    links = set()
    for text in texts:
        links.update(url_pattern.findall(text))
    return sorted(list(links))

def background_summarize(job_id, thread_url, language="en", summary_length="medium"):
    """Performs summarization and data extraction in a background thread."""
    try:
        set_progress(job_id, 0)  # Fetching Thread
        # Check cache first
        cache_key = f"summary_{thread_url}_{summary_length}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logging.info(f"Using cached result for {thread_url}")
            cache.set(job_id, cached_result, timeout=600)
            set_progress(job_id, len(PROCESSING_STEPS)-1)  # Finalizing
            return
        post_text, comments = fetch_post_and_comments(thread_url, limit=5)  # Lowered from 20
        set_progress(job_id, 1)  # Analyzing Comments
        # Truncate the combined text BEFORE sending to LLM for all purposes
        combined_for_llm = safe_truncate(post_text + "\n\n" + "\n".join([c[0] for c in comments[:5]]), 4000)  # Use up to 5 comments, 4000 char limit
        set_progress(job_id, 2)  # Generating Summary
        # Only call Gemini, sentiment, and funny comments in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                'gemini': executor.submit(process_thread_with_gemini, combined_for_llm, language, summary_length),
                'sentiment': executor.submit(analyze_sentiment, comments),
                'funny_comments': executor.submit(extract_funny_comments, comments),
            }
            results = {}
            errors = {}
            for key, future in futures.items():
                try:
                    results[key] = future.result()
                except Exception as e:
                    logging.error(f"Error in parallel task {key}: {e}")
                    errors[key] = str(e)
        gemini_outputs, gemini_error = results.get('gemini', (None, True))
        if gemini_error or not gemini_outputs:
            # Check if it's a circuit breaker error
            if gemini_outputs and "circuit breaker" in gemini_outputs.get("summary", "").lower():
                error_msg = "Gemini API is temporarily unavailable due to rate limits or service issues. Please try again in a few minutes."
            else:
                error_msg = gemini_outputs.get("summary", "An unknown error occurred during Gemini processing.") if gemini_outputs else "An unknown error occurred during Gemini processing."
            cache.set(job_id, {'error': error_msg}, timeout=600)
            set_progress(job_id, len(PROCESSING_STEPS)-1)  # Finalizing
            return
        summary = gemini_outputs["summary"]
        action_items_notes = gemini_outputs["action_items_notes"]
        key_facts = gemini_outputs["key_facts"]
        set_progress(job_id, 3)  # Extracting Insights
        top_comments = sorted(comments, key=lambda x: x[1], reverse=True)[:3]
        sentiment_result = results.get('sentiment', {'verdict': '', 'score': 0, 'breakdown': {}})
        funny_comments = results.get('funny_comments', [])
        result = {
            'summary': summary,
            'top_comments': top_comments or [],
            'sentiment_verdict': sentiment_result.get('verdict', ''),
            'sentiment_score': sentiment_result.get('score', 0),
            'sentiment_breakdown': sentiment_result.get('breakdown', {}),
            'funny_comments': funny_comments or [],
            'action_items_notes': action_items_notes,
            'key_facts': key_facts,
            'thread_url': thread_url
        }
        # Cache the result for 1 hour
        cache.set(cache_key, result, timeout=3600)
        cache.set(job_id, result, timeout=600)
        set_progress(job_id, 4)  # Finalizing
        logging.info(f"Summarization job {job_id} completed successfully.")
    except ValueError as ve:
        logging.error(f"Configuration error for job {job_id}: {ve}", exc_info=True)
        cache.set(job_id, {'error': f"Configuration Error: {ve}"}, timeout=600)
        set_progress(job_id, len(PROCESSING_STEPS)-1)  # Finalizing
    except Exception as e:
        # Check if it's a circuit breaker error
        if "circuit breaker" in str(e).lower():
            if "reddit" in str(e).lower():
                error_msg = "Reddit API is temporarily unavailable. Please try again in a few minutes."
            else:
                error_msg = "External service is temporarily unavailable. Please try again in a few minutes."
        else:
            error_msg = f"An error occurred during processing: {e}. Please check the URL or try again later."
        logging.error(f"Error in background summarization for job {job_id} ({thread_url}): {e}", exc_info=True)
        cache.set(job_id, {'error': error_msg}, timeout=600)
        set_progress(job_id, len(PROCESSING_STEPS)-1)  # Finalizing

def get_trending_threads(limit=5):
    reddit = get_reddit()
    if not reddit:
        logging.error("Reddit PRAW not initialized, cannot fetch trending threads.")
        return []
    def _fetch_trending_threads():
        reddit = get_reddit()
        if not reddit:
            raise Exception("Reddit PRAW not initialized")
        subreddit = reddit.subreddit("popular")
        posts = [post for post in subreddit.hot(limit=25) if not post.stickied and post.num_comments > 5]
        threads = [{'title': post.title, 'url': f"https://www.reddit.com{post.permalink}"} for post in posts[:limit]]
        return threads
    try:
        threads = reddit_circuit_breaker.call(_fetch_trending_threads)
        return threads
    except Exception as e:
        logging.error(f"Error fetching trending threads: {e}")
        return []

def get_trending_subreddits(limit=5):
    """Fetches a list of trending subreddits."""
    reddit = get_reddit()
    if not reddit: # Check if PRAW was initialized
        logging.error("Reddit PRAW not initialized, cannot fetch trending subreddits.")
        return []
    
    def _fetch_trending_subreddits():
        reddit = get_reddit()
        if not reddit:
            raise Exception("Reddit PRAW not initialized")
        return [sub.display_name for sub in reddit.subreddits.popular(limit=limit)]
    
    try:
        return reddit_circuit_breaker.call(_fetch_trending_subreddits)
    except Exception as e:
        logging.error(f"Circuit breaker prevented Reddit API call for trending subreddits: {e}")
        return []

@app.route('/trending_threads')
def trending_threads_api():
    """API endpoint to get trending Reddit threads as JSON."""
    force_refresh = request.args.get('force_refresh') == '1'
    threads = get_trending_threads(limit=10)
    if threads:
        return jsonify({
            "threads": threads,
            "timestamp": datetime.now().isoformat(),
            "fresh": force_refresh or False
        })
    else:
        reddit_status = reddit_circuit_breaker.get_status()
        error_message = "Could not fetch trending threads. Please try again later."
        if reddit_status['state'] == CircuitState.OPEN.value:
            error_message = f"Reddit API is temporarily unavailable ({reddit_status['state']} state). Please try again in {reddit_status['recovery_timeout_seconds']} seconds."
        return jsonify({"threads": [], "error": error_message}), 500

@app.route('/random_thread_url')
def random_thread_url():
    url = get_random_trending_thread_url()
    if url:
        return jsonify({'url': url})
    else:
        return jsonify({'url': None}), 500

@app.route('/progress/<job_id>')
def progress_api(job_id):
    prog = cache.get(f"progress_{job_id}")
    if not prog:
        prog = {'step': 1, 'total_steps': 5, 'label': 'Fetching Thread', 'progress': 5}
    return jsonify(prog)

@app.route("/loading/<job_id>")
def loading(job_id):
    """Displays a loading page while the summary is generated."""
    return render_template("loading.html", job_id=job_id)

@app.route("/result_ready/<job_id>")
def result_ready(job_id):
    """API endpoint to check if summarization results are ready."""
    result = cache.get(job_id)
    return jsonify({"ready": bool(result)})

@app.route("/result/<job_id>")
def result(job_id):
    """Displays the summarization results."""
    result = cache.get(job_id)
    if not result:
        # If result not found, it might have expired or never existed
        return render_template("error.html", error="Result not found or expired. Please go back and try again."), 404

    if 'error' in result:
        return render_template("error.html", error=result['error'])

    # Prepare data for rendering
    action_items_html = bullets_to_html(result.get('action_items_notes', ''))
    key_facts_html = bullets_to_html(result.get('key_facts', ''))

    return render_template("result.html", result=result,
                                  action_items_html=action_items_html, key_facts_html=key_facts_html)

def bullets_to_html(text):
    """Converts a bulleted string or a list to an HTML unordered list, handling potential intro text."""
    if not text:
        return ""
    # If text is a list, treat each item as a bullet
    if isinstance(text, list):
        lines = [str(line).strip() for line in text if str(line).strip()]
    else:
        lines = [line.strip() for line in str(text).split('\n') if line.strip()]
    html_content = []
    in_list = False
    for line in lines:
        # Check for various bullet point formats
        if (line.startswith('*') or line.startswith('-') or \
            line.startswith('•') or line.startswith('·') or
            line.startswith('‣') or line.startswith('◦') or
            line.startswith('▪') or line.startswith('▫') or
            line.startswith('○') or line.startswith('●') or
            line.startswith('◆') or line.startswith('◇') or
            line.startswith('■') or line.startswith('□') or
            line.startswith('▶') or line.startswith('►') or
            line.startswith('➤') or line.startswith('➜') or
            line.startswith('→') or line.startswith('⇒') or
            line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or
            line.startswith('4.') or line.startswith('5.') or line.startswith('6.') or
            line.startswith('7.') or line.startswith('8.') or line.startswith('9.') or
            line.startswith('10.') or line.startswith('11.') or line.startswith('12.') or
            line.startswith('13.') or line.startswith('14.') or line.startswith('15.') or
            line.startswith('16.') or line.startswith('17.') or line.startswith('18.') or
            line.startswith('19.') or line.startswith('20.') or
            line.startswith('1)') or line.startswith('2)') or line.startswith('3)') or
            line.startswith('4)') or line.startswith('5)') or line.startswith('6)') or
            line.startswith('7)') or line.startswith('8)') or line.startswith('9)') or
            line.startswith('10)') or line.startswith('11)') or line.startswith('12)') or
            line.startswith('13)') or line.startswith('14)') or line.startswith('15)') or
            line.startswith('16)') or line.startswith('17)') or line.startswith('18)') or
            line.startswith('19)') or line.startswith('20)')):
            if not in_list:
                html_content.append("<ul>")
                in_list = True
            # Remove the bullet character and any leading whitespace
            clean_line = line.lstrip('*-•·‣◦▪▫○●◆◇■□▶►➤➜→⇒0123456789.() ')
            html_content.append(f"<li>{clean_line}</li>")
        else:
            if in_list:
                html_content.append("</ul>")
                in_list = False
            html_content.append(f"<p>{line}</p>")
    if in_list:
        html_content.append("</ul>")
    return "".join(html_content)

@app.route("/test")
def test():
    """A simple test route to check if the app is running."""
    return "App is running!"

@app.route("/status")
def status():
    """API endpoint to check the status of circuit breakers and services."""
    # Count available API keys (mask them for security)
    available_keys = len(GEMINI_API_KEYS)
    masked_keys = []
    for i, key in enumerate(GEMINI_API_KEYS):
        if key:
            masked_keys.append(f"Key {i+1}: {key[:8]}...{key[-4:]}")
        else:
            masked_keys.append(f"Key {i+1}: Not set")

    return jsonify({
        'reddit_circuit_breaker': reddit_circuit_breaker.get_status(),
        'gemini_circuit_breaker': gemini_circuit_breaker.get_status(),
        'reddit_initialized': reddit_circuit_breaker.get_status()['state'] != CircuitState.OPEN.value, # Check if PRAW is initialized
        'gemini_api_keys': {
            'total_available': available_keys,
            'current_key_index': current_key_index,
            'keys': masked_keys
        }
    })

@app.route("/", methods=["GET", "POST"])
def index():
    thread_url_val = ""
    summary_length_val = "medium"
    error = None
    if request.method == "POST":
        thread_url_val = request.form.get("thread_url", "").strip()
        summary_length_val = request.form.get("summary_length", "medium")
        if not thread_url_val:
            error = "Please enter a Reddit thread URL."
        else:
            job_id = str(uuid.uuid4())
            threading.Thread(target=background_summarize, args=(job_id, thread_url_val, "en", summary_length_val)).start()
            return redirect(url_for("loading", job_id=job_id))
    return render_template("index.html", thread_url_val=thread_url_val, summary_length_val=summary_length_val, error=error)

@app.route('/favicon.ico')
def favicon():
    return redirect('https://www.redditstatic.com/desktop2x/img/favicon/apple-icon-180x180.png')

if __name__ == '__main__':
    print("App import and setup time:", time.time() - startup_start)
    app.run(debug=True)