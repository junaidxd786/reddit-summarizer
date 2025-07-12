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

load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLI_MODEL = os.getenv("GEMINI_CLI_MODEL", "gemini-2.5-pro")

# Validate environment variables
if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT:
    logging.error("Missing Reddit API credentials. Please set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your .env file.")
    # Consider sys.exit(1) here if missing Reddit creds makes app unusable
if not GEMINI_API_KEY:
    logging.error("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
    # Consider sys.exit(1) here if missing Gemini key makes app unusable

try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    logging.info("PRAW initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize PRAW: {e}. Check your Reddit API credentials and user agent.")
    reddit = None # Ensure reddit is None if initialization fails to prevent further errors


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
    if not reddit: # Check if PRAW was initialized
        logging.error("Reddit PRAW not initialized, cannot fetch trending threads.")
        return None
    try:
        subreddit = reddit.subreddit("popular")
        # Fetch fewer posts for quicker response, and filter as before
        posts = [post for post in subreddit.hot(limit=25) if not post.stickied and post.num_comments > 5]
        if not posts:
            logging.warning("No suitable trending posts found.")
            return None
        post = random.choice(posts)
        return f"https://www.reddit.com{post.permalink}"
    except Exception as e:
        logging.error(f"Error fetching random trending thread: {e}")
        return None

def fetch_post_and_comments(thread_url, limit=50): # Reduced default limit
    """Fetches the main post text and a limited number of comments from a Reddit thread."""
    if not reddit: # Check if PRAW was initialized
        raise ValueError("Reddit PRAW not initialized, cannot fetch post and comments.")
    try:
        submission = reddit.submission(url=thread_url)
        # Replacing more comments with limit=0 can be very slow for large threads.
        # Consider a more aggressive limit or a different strategy if this is the bottleneck.
        submission.comments.replace_more(limit=0) # Fetches all top-level 'More Comments'
        
        comments = []
        # Iterate over submission.comments.list() which flattens the comment tree
        # Take only up to `limit` comments to control processing time
        for comment in submission.comments.list()[:limit]:
            if isinstance(comment, Comment):
                comments.append((comment.body, getattr(comment, 'score', 0)))
        
        post_text = submission.title + "\n\n" + (submission.selftext or "")
        return post_text, comments
    except Exception as e:
        logging.error(f"Error fetching post and comments for {thread_url}: {e}")
        raise

def gemini_api_process(text_prompt, model=GEMINI_CLI_MODEL, api_key=None):
    """
    Calls the Gemini API to process a text prompt and returns the output.
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Gemini API key not set.", True

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

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        if response.status_code != 200:
            return f"Gemini API error: {response.text}", True
        result = response.json()
        # Extract the text from the response
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip(), False
    except Exception as e:
        return f"Gemini API call failed: {e}", True

def clean_gemini_output(output):
    """Removes common unwanted output from Gemini CLI."""
    # This might need to be adjusted based on the new JSON output format.
    # For now, it will just clean general CLI chatter.
    return output.replace("Loaded cached credentials.", "").strip()

def extract_json_from_markdown(text):
    """
    Extracts JSON from a Markdown code block if present, otherwise returns the text as is.
    """
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

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
As an AI assistant, analyze the following Reddit thread and provide the requested information in a JSON format.
The JSON object should have the following keys:
- "summary": A plain English summary based on the length requested.
- "action_items_notes": A bulleted list of 3-5 main action items and important notes. If fewer, list only what's important.
- "key_facts": A bulleted list of 3-5 key facts. If fewer, list only what's important.

Ensure the output is valid JSON.

Here is the Reddit Thread:
---
{truncated_text}
---

Instructions:
1. Summary: {summary_directive}
2. Action Items & Notes: List concise bullet points.
3. Key Facts: List concise bullet points.
4. Language: All output should be in {language} if possible, otherwise default to English.

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
        return {"summary": summary, "action_items_notes": action_items_notes, "key_facts": key_facts}, False
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Gemini output: {e}\nRaw output: {cleaned_result}")
        return {"summary": f"Error parsing Gemini response: {e}. Raw output: {cleaned_result[:200]}...", "action_items_notes": "", "key_facts": ""}, True
    except Exception as e:
        logging.error(f"Unexpected error processing Gemini output: {e}")
        return {"summary": f"An unexpected error occurred: {e}", "action_items_notes": "", "key_facts": ""}, True


def extract_links(texts):
    """Extracts all URLs from a list of texts."""
    url_pattern = re.compile(r'https?://\S+')
    links = set()
    for text in texts:
        links.update(url_pattern.findall(text))
    return sorted(list(links))

def analyze_sentiment(comments):
    """Analyzes the sentiment of comments."""
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text[0])['compound'] for text in comments if text[0].strip()]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score > 0.15:
        return "Most people like this post. (Positive sentiment)"
    elif avg_score < -0.15:
        return "Most people dislike this post. (Negative sentiment)"
    else:
        return "The sentiment is mixed. (Neutral sentiment)"

def extract_faqs(comments, max_faqs=5):
    """Extracts potential FAQs from comments based on questions and upvotes."""
    questions = [(body, score) for body, score in comments if body.strip().endswith('?')]
    top_questions = sorted(questions, key=lambda x: x[1], reverse=True)[:max_faqs]
    return top_questions

def extract_expert_opinions(thread_url, limit=50): # Reduced limit for consistency
    """Extracts comments from users with specific 'expert' flairs."""
    if not reddit: # Check if PRAW was initialized
        logging.error("Reddit PRAW not initialized, cannot extract expert opinions.")
        return []
    try:
        submission = reddit.submission(url=thread_url)
        # Be careful with replace_more(limit=0) if it's the bottleneck
        submission.comments.replace_more(limit=0)
        
        expert_keywords = ["expert", "phd", "dr", "professor", "official", "mod"] # Added 'mod'
        expert_comments = []
        
        for comment in submission.comments.list()[:limit]: # Only process up to limit comments
            if isinstance(comment, Comment):
                flair = (getattr(comment.author_flair_text, 'lower', lambda: "")() if comment.author_flair_text else "")
                if any(keyword in flair for keyword in expert_keywords):
                    expert_comments.append((comment.body, getattr(comment, 'score', 0), flair))
        
        expert_comments = sorted(expert_comments, key=lambda x: x[1], reverse=True)
        return expert_comments
    except Exception as e:
        logging.error(f"Error extracting expert opinions for {thread_url}: {e}")
        return []

def extract_funny_comments(comments, max_funny=3):
    """Extracts comments likely intended to be funny."""
    funny_keywords = ["lol", "funny", "hilarious", "lmao", "rofl", "haha", "xd", "lmfao", "hehe", "humor", "joke"]
    funny_comments = [(body, score) for body, score in comments if any(kw in body.lower() for kw in funny_keywords)]
    top_funny = sorted(funny_comments, key=lambda x: x[1], reverse=True)[:max_funny]
    return top_funny

def safe_truncate(text, max_chars=4000):
    """Truncates text to a safe character limit."""
    return text[:max_chars]

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
        if line.startswith('*') or line.startswith('-'):
            if not in_list:
                html_content.append("<ul>")
                in_list = True
            html_content.append(f"<li>{line[1:].strip()}</li>") # Remove bullet char
    else:
            if in_list:
                html_content.append("</ul>")
                in_list = False
            html_content.append(f"<p>{line}</p>")
    
    if in_list:
        html_content.append("</ul>")

    return "".join(html_content)

def background_summarize(job_id, thread_url, language="en", summary_length="medium"):
    """Performs summarization and data extraction in a background thread."""
    try:
        post_text, comments = fetch_post_and_comments(thread_url, limit=50) # Use reduced limit here
        
        # Truncate the combined text BEFORE sending to LLM for all purposes
        combined_for_llm = safe_truncate(post_text + "\n\n" + "\n".join([c[0] for c in comments[:20]]), 8000) # Use up to 20 comments, 8000 char limit

        # Consolidated Gemini call
        gemini_outputs, gemini_error = process_thread_with_gemini(combined_for_llm, language=language, summary_length=summary_length)

        if gemini_error:
            # If the Gemini call itself failed or parsing failed, store the error
            cache.set(job_id, {'error': gemini_outputs.get("summary", "An unknown error occurred during Gemini processing.")}, timeout=600)
            return

        summary = gemini_outputs["summary"]
        action_items_notes = gemini_outputs["action_items_notes"]
        key_facts = gemini_outputs["key_facts"]

        top_comments = sorted(comments, key=lambda x: x[1], reverse=True)[:3]
        
        all_texts_for_links = [post_text] + [c[0] for c in comments]
        links = extract_links(all_texts_for_links)
        
        sentiment_verdict = analyze_sentiment(comments)
        faqs = extract_faqs(comments)
        expert_opinions = extract_expert_opinions(thread_url, limit=50) # Keep consistent limit
        funny_comments = extract_funny_comments(comments)
        
        cache.set(job_id, {
            'summary': summary,
            'top_comments': top_comments,
            'links': links,
            'sentiment_verdict': sentiment_verdict,
            'faqs': faqs,
            'expert_opinions': expert_opinions,
            'funny_comments': funny_comments,
            'action_items_notes': action_items_notes,
            'key_facts': key_facts,
            'thread_url': thread_url
        }, timeout=600)
        logging.info(f"Summarization job {job_id} completed successfully.")
    except ValueError as ve:
        logging.error(f"Configuration error for job {job_id}: {ve}", exc_info=True)
        cache.set(job_id, {'error': f"Configuration Error: {ve}"}, timeout=600)
    except Exception as e:
        logging.error(f"Error in background summarization for job {job_id} ({thread_url}): {e}", exc_info=True)
        cache.set(job_id, {'error': f"An error occurred during processing: {e}. Please check the URL or try again later."}, timeout=600)

def get_trending_subreddits(limit=5):
    """Fetches a list of trending subreddits."""
    if not reddit: # Check if PRAW was initialized
        logging.error("Reddit PRAW not initialized, cannot fetch trending subreddits.")
        return []
    try:
        return [sub.display_name for sub in reddit.subreddits.popular(limit=limit)]
    except Exception as e:
        logging.error(f"Error fetching trending subreddits: {e}")
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the main page, input form, and starts summarization."""
    # Remove language selection, always use English
    error = None
    if request.method == "POST":
        thread_url = request.form.get("thread_url")

        if not thread_url:
            error = "Please enter a Reddit thread URL."
            return render_template("index.html", error=error, thread_url_val=request.form.get('thread_url', ''))
        
        job_id = str(uuid.uuid4())
        threading.Thread(target=background_summarize, args=(job_id, thread_url, "en")).start()
        return redirect(url_for('loading', job_id=job_id))
    
    return render_template("index.html", thread_url_val="")

@app.route("/random_thread_url")
def random_thread_url():
    """API endpoint to get a random trending Reddit thread URL."""
    url = get_random_trending_thread_url()
    return jsonify({"url": url})

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

@app.route("/test")
def test():
    """A simple test route to check if the app is running."""
    return "App is running!"

if __name__ == '__main__':
    app.run(debug=True)