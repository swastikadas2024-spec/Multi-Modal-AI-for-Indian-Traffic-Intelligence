import tweepy
import requests
import pandas as pd
from datetime import datetime
import os
import json


def fetch_twitter_complaints():
    """
    Fetch traffic-related complaints from Twitter using Tweepy.
    
    Returns:
        list: List of tweet data dictionaries
    """
    print("[*] Fetching Twitter complaints...")
    
    twitter_data = []
    
    try:
        # Get bearer token from environment
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        
        if not bearer_token:
            print("[!] Warning: TWITTER_BEARER_TOKEN not found in environment variables")
            return twitter_data
        
        # Initialize Tweepy client
        client = tweepy.Client(bearer_token=bearer_token)
        
        # Search query for traffic-related tweets
        query = "(traffic OR congestion OR jam OR accident OR potholes) -is:retweet"
        
        print(f"[*] Searching Twitter with query: {query}")
        
        # Search for tweets
        tweets = client.search_recent_tweets(
            query=query,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics']
        )
        
        if tweets.data:
            for tweet in tweets.data:
                twitter_data.append({
                    'source': 'twitter',
                    'text': tweet.text,
                    'timestamp': tweet.created_at,
                    'likes': tweet.public_metrics['like_count'],
                    'retweets': tweet.public_metrics['retweet_count']
                })
            print(f"[+] Fetched {len(twitter_data)} tweets from Twitter")
        else:
            print("[*] No tweets found")
            
    except tweepy.TweepyException as e:
        print(f"[!] Twitter API error: {str(e)}")
    except Exception as e:
        print(f"[!] Error fetching Twitter data: {str(e)}")
    
    return twitter_data


def fetch_311_complaints():
    """
    Fetch traffic complaints from 311 API.
    
    Returns:
        list: List of 311 complaint data dictionaries
    """
    print("[*] Fetching 311 complaints...")
    
    complaints_311 = []
    
    try:
        # Placeholder 311 API endpoint
        # Replace with actual 311 data portal API (e.g., NYC Open Data)
        api_url = "https://data.cityofnewyork.us/api/views/erm2-nwe9/rows.json?accessType=DOWNLOAD"
        
        print(f"[*] Making request to: {api_url}")
        
        # Make GET request
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        
        print("[+] Successfully connected to 311 API")
        
        data = response.json()
        
        # Parse JSON response - structure depends on actual API
        if 'data' in data:
            for record in data['data']:
                # Typical 311 complaint fields
                complaints_311.append({
                    'source': '311',
                    'text': record.get('complaint_description', ''),
                    'timestamp': record.get('created_date', ''),
                    'category': record.get('complaint_type', 'traffic'),
                    'location': record.get('location', '')
                })
            print(f"[+] Fetched {len(complaints_311)} complaints from 311")
        else:
            print("[*] No complaints found in response")
            
    except requests.exceptions.Timeout:
        print("[!] Request timeout: 311 API took too long to respond")
    except requests.exceptions.ConnectionError:
        print("[!] Connection error: Unable to reach 311 API")
    except requests.exceptions.HTTPError as e:
        print(f"[!] HTTP error: {str(e)}")
    except json.JSONDecodeError:
        print("[!] Error: Response is not valid JSON")
    except Exception as e:
        print(f"[!] Error fetching 311 data: {str(e)}")
    
    return complaints_311


def main():
    """
    Main function to fetch complaints from both sources, combine them, and save to CSV.
    """
    print("[*] Starting complaint data fetch...")
    
    # Fetch data from both sources
    twitter_complaints = fetch_twitter_complaints()
    complaints_311 = fetch_311_complaints()
    
    # Combine results
    all_complaints = twitter_complaints + complaints_311
    
    if not all_complaints:
        print("[!] No complaints fetched from any source")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_complaints)
    
    print(f"[+] Total complaints collected: {len(df)}")
    print(f"[+] DataFrame columns: {list(df.columns)}")
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    filepath = os.path.join(raw_data_dir, f'complaints_{timestamp}.csv')
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"[+] Data saved to: {filepath}")
    print(f"[+] File size: {os.path.getsize(filepath)} bytes")
    
    # Print sample data
    print("\n[*] Sample data:")
    print(df.head())


if __name__ == "__main__":
    main()
