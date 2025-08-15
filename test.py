import os
import re
import json
import requests
from flask import Flask, request, jsonify, render_template_string
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from openai import OpenAI
from urllib.parse import urlparse, parse_qs
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Replace with your actual API keys
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '...')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '...')


youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


REQUEST_DELAY_SECONDS = float(os.getenv('REQUEST_DELAY_SECONDS', '5'))


def rate_limit_delay():
    try:
        time.sleep(REQUEST_DELAY_SECONDS)
    except Exception:
        pass

class YouTubeAnalyzer:
    def __init__(self):
        self.youtube = youtube
        self.openai_client = openai_client
        # Proxy rotation disabled
        # self._proxy_configs = [
        #     GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        #     for proxy_url in PROXY_LIST
        # ] if PROXY_LIST else []
        # self._proxy_cycle = itertools.cycle(self._proxy_configs) if self._proxy_configs else None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:vi\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def extract_channel_id(self, url: str) -> Optional[str]:
        """Extract channel ID from YouTube URL or get it from video"""
        # Direct channel URL patterns
        if '/channel/' in url:
            return url.split('/channel/')[-1].split('/')[0]
        
        if '/c/' in url or '/@' in url:
            # Handle custom URLs - need to resolve to channel ID
            return self.resolve_channel_from_custom_url(url)
        
        # If it's a video URL, get channel from video
        video_id = self.extract_video_id(url)
        if video_id:
            return self.get_channel_from_video(video_id)
        
        return None
    
    def resolve_channel_from_custom_url(self, url: str) -> Optional[str]:
        """Resolve custom URL to channel ID using search"""
        try:
            # Extract username/custom name
            if '/@' in url:
                username = url.split('/@')[-1].split('/')[0]
            elif '/c/' in url:
                username = url.split('/c/')[-1].split('/')[0]
            else:
                return None
            
            # Search for the channel
            search_response = self.youtube.search().list(
                q=username,
                type='channel',
                part='snippet',
                maxResults=1,
                relevanceLanguage='en'
            ).execute()
            rate_limit_delay()
            
            if search_response['items']:
                return search_response['items'][0]['snippet']['channelId']
        except Exception as e:
            logger.error(f"Error resolving custom URL: {e}")
        
        return None
    
    def get_channel_from_video(self, video_id: str) -> Optional[str]:
        """Get channel ID from video ID"""
        try:
            video_response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            rate_limit_delay()
            
            if video_response['items']:
                return video_response['items'][0]['snippet']['channelId']
        except Exception as e:
            logger.error(f"Error getting channel from video: {e}")
        
        return None
    
    def get_channel_videos(self, channel_id: str, max_results: int = 5) -> List[Dict]:
        """Get latest videos from a channel"""
        try:
            # Get uploads playlist ID
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            rate_limit_delay()
            
            if not channel_response['items']:
                return []
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get latest videos from uploads playlist
            playlist_response = self.youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=max_results
            ).execute()
            rate_limit_delay()
            
            videos = []
            for item in playlist_response['items']:
                video_data = {
                    'video_id': item['snippet']['resourceId']['videoId'],
                    'title': item['snippet']['title'],
                    'published_at': item['snippet']['publishedAt'],
                    'description': item['snippet']['description']
                }
                videos.append(video_data)
            
            return videos
        
        except Exception as e:
            logger.error(f"Error getting channel videos: {e}")
            return []
    
    def get_video_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript for a video using delay only (proxy rotation disabled)"""
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id, languages=("en",))
            raw_data=transcript.to_raw_data()
            full_transcript = ' '.join(entry['text'] for entry in raw_data)
            return full_transcript
        except Exception as e:
            logger.error(f"Error getting transcript for {video_id}: {e}")
            return None
        finally:
            rate_limit_delay()
    
    def analyze_transcript_with_gpt(self, transcript: str, video_title: str) -> Dict:
        """Analyze transcript using GPT-4"""
        try:
            prompt = f"""
            Analyze the following podcast/video transcript and extract:
            
            1. Main topic titles discussed (list)
            2. Guests mentioned (list of names)
            3. Key insights (list of important points)
            4. Topic gaps (what related topics weren't covered but could be)
            
            Video Title: {video_title}
            
            Transcript: {transcript[:8000]}  # Limit to avoid token limits
            
            Please respond in valid JSON format with the following structure ONLY:
            {{
                "topics": ["topic1", "topic2", ..., "topic5"],
                "guests": ["guest1", "guest2", ...],
                "insights": ["insight1", "insight2", ..., "insight5"],
                "topic_gaps": ["gap1", "gap2", ..., "gap5"]
            }}
            Do not include any other text or comments.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are an expert podcast content analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000,
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing transcript with GPT: {e}")
            return {
                "topics": [],
                "guests": [],
                "insights": [],
                "topic_gaps": []
            }
    
    def search_similar_channels(self, topics: List[str], max_results: int = 3) -> List[str]:
        """Search for similar podcast channels based on topics"""
        try:
            search_query = f"{' '.join(topics[:3])} podcast or interview"  # Use first 3 topics
            logger.info(f"YouTubeSearch query: {search_query}")

            search_response = self.youtube.search().list(
                q=search_query,
                type='channel',
                part='snippet',
                maxResults=max_results,
                order='relevance',
                relevanceLanguage='en'
            ).execute()
            rate_limit_delay()
            
            
            channel_ids = []
            for item in search_response['items']:
                channel_ids.append(item['id']['channelId'])
            
            return channel_ids
        
        except Exception as e:
            logger.error(f"Error searching similar channels: {e}")
            return []
    
    def process_channel(self, channel_id: str, channel_name: str = "Unknown") -> Dict:
        """Process a single channel and return analysis"""
        logger.info(f"Processing channel: {channel_name}")
        
        # Get latest 5 videos
        videos = self.get_channel_videos(channel_id, 5)
        channel_analysis = {
            'channel_id': channel_id,
            'channel_name': channel_name,
            'videos_analyzed': [],
            'overall_topics': set(),
            'overall_guests': set(),
            'overall_insights': [],
            'overall_topic_gaps': set()
        }
        
        for video in videos:
            logger.info(f"Processing video: {video['title']}")
            
            # Get transcript
            transcript = self.get_video_transcript(video['video_id'])
            if not transcript:
                continue

            analysis = self.analyze_transcript_with_gpt(transcript, video['title'])
            
            video_data = {
                'video_id': video['video_id'],
                'title': video['title'],
                'published_at': video['published_at'],
                'analysis': analysis
            }
            
            channel_analysis['videos_analyzed'].append(video_data)
            
            # Aggregate data
            channel_analysis['overall_topics'].update(analysis.get('topics', []))
            channel_analysis['overall_guests'].update(analysis.get('guests', []))
            channel_analysis['overall_insights'].extend(analysis.get('insights', []))
            channel_analysis['overall_topic_gaps'].update(analysis.get('topic_gaps', []))
        
        # Convert sets to lists for JSON serialization
        channel_analysis['overall_topics'] = list(channel_analysis['overall_topics'])
        channel_analysis['overall_guests'] = list(channel_analysis['overall_guests'])
        channel_analysis['overall_topic_gaps'] = list(channel_analysis['overall_topic_gaps'])
        
        return channel_analysis
    
    def analyze_youtube_channel(self, url: str) -> Dict:
        """Main function to analyze a YouTube channel and find similar ones"""
        try:
            # Extract channel ID
            channel_id = self.extract_channel_id(url)
            if not channel_id:
                return {"error": "Could not extract channel ID from URL"}
            
            channel_response = self.youtube.channels().list(
                part='snippet',
                id=channel_id
            ).execute()
            rate_limit_delay()
            
            if not channel_response['items']:
                return {"error": "Channel not found"}
            
            channel_name = channel_response['items'][0]['snippet']['title']
            
            main_channel_analysis = self.process_channel(channel_id, channel_name)
            
            similar_channel_ids = self.search_similar_channels(
                main_channel_analysis['overall_topics']
            )
            similar_search_results = similar_channel_ids["channel_ids"]
            search_query_used = similar_channel_ids["search_query"]
            
            similar_channels_analysis = []
            for similar_id in similar_channel_ids:
                if similar_id != channel_id:  # Don't analyze the same channel
                    try:
                        similar_channel_response = self.youtube.channels().list(
                            part='snippet',
                            id=similar_id
                        ).execute()
                        rate_limit_delay()
                        
                        if similar_channel_response['items']:
                            similar_name = similar_channel_response['items'][0]['snippet']['title']
                            similar_analysis = self.process_channel(similar_id, similar_name)
                            similar_channels_analysis.append(similar_analysis)
                    except Exception as e:
                        logger.error(f"Error processing similar channel {similar_id}: {e}")
            
           
            result = {
                'timestamp': datetime.now().isoformat(),
                'main_channel': main_channel_analysis,
                'similar_channels': similar_channels_analysis,
                'summary': {
                    'search_query_used': search_query_used,
                    'total_channels_analyzed': 1 + len(similar_channels_analysis),
                    'total_videos_analyzed': len(main_channel_analysis['videos_analyzed']) + 
                                           sum(len(ch['videos_analyzed']) for ch in similar_channels_analysis)
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in main analysis: {e}")
            return {"error": str(e)}

analyzer = YouTubeAnalyzer()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Transcript Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="url"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .loading { display: none; color: #666; margin-top: 10px; }
        .result { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 4px; }
        .error { color: red; }
        pre { background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>YouTube Transcript Analyzer</h1>
    <p>Enter a YouTube channel URL or video URL to analyze transcripts and find similar channels.</p>
    
    <form id="analyzeForm">
        <div class="form-group">
            <label for="url">YouTube URL:</label>
            <input type="url" id="url" name="url" placeholder="https://youtube.com/channel/..." required>
        </div>
        <button type="submit">Analyze</button>
    </form>
    
    <div class="loading" id="loading">
        Analyzing... This may take several minutes as we process transcripts and analyze content with AI.
    </div>
    
    <div id="result"></div>

    <script>
        document.getElementById('analyzeForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const url = document.getElementById('url').value;
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.innerHTML = '';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({url: url})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    result.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                } else {
                    result.innerHTML = `
                        <div class="result">
                            <h3>Analysis Complete!</h3>
                            <p><strong>Channels analyzed:</strong> ${data.summary.total_channels_analyzed}</p>
                            <p><strong>Videos analyzed:</strong> ${data.summary.total_videos_analyzed}</p>
                            <p><a href="/download/${data.timestamp}" target="_blank">Download JSON Results</a></p>
                            <h4>Preview:</h4>
                            <pre>${JSON.stringify(data, null, 2).substring(0, 2000)}...</pre>
                        </div>
                    `;
                }
            } catch (error) {
                result.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            }
            
            loading.style.display = 'none';
        });
    </script>
</body>
</html>
"""

results_cache = {}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL is required'})
        
        # Perform analysis
        result = analyzer.analyze_youtube_channel(url)
        
        # Cache result
        if 'timestamp' in result:
            results_cache[result['timestamp']] = result
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({'error': str(e)})

@app.route('/download/<timestamp>')
def download_results(timestamp):
    if timestamp in results_cache:
        response = app.response_class(
            response=json.dumps(results_cache[timestamp], indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment; filename=youtube_analysis_{timestamp}.json'
        return response
    else:
        return jsonify({'error': 'Results not found'}), 404

if __name__ == '__main__':
    print()
    
    app.run(debug=True, host='0.0.0.0', port=8080)