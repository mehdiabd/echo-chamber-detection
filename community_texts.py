"""
Fetch top or recent tweets for a community's nodes from Elasticsearch.
"""

import json
from elastic import es


def fetch_recent_tweets(username, index="twitter_temp_data", size=5):
    """
    Fetch most recent tweets for a given username.
    :param username: str - account handle
    :param index: str - elastic index name
    :param size: int - number of tweets to fetch
    :return: list of tweet texts
    """
    try:
        query = {
            "query": {
                "term": {"sender": username}  # Removed .keyword
            },
            "sort": [{"date": {"order": "desc"}}],
            "_source": ["normalized_text", "content"],
            "size": size
        }
        res = es.search(index=index, body=query)
        tweets = [] 
        for hit in res["hits"]["hits"]:
            source = hit["_source"]
            text = source.get("normalized_text") or source.get("content")
            if text:
                tweets.append(text.strip())
        return tweets

    except Exception as e:
        print(f"[error] failed to fetch tweets for {username}: {e}")
        return []


def fetch_community_texts(center_node, neighbors, index="twitter_temp_data", size=5):
    """
    Fetch top or most recent tweets for a community's center node and its neighbors
    from Elasticsearch. Prefers normalized_text if available.
    """
    community_nodes = [center_node] + list(neighbors)
    results = {}

    for node in community_nodes:
        try:
            query = {
                "query": {"term": {"user_name": node}},  # Changed sender.keyword
                "sort": [{"date": {"order": "desc"}}],
                "size": size,
                "_source": ["normalized_text", "content"]
            }
            resp = es.search(index=index, body=query)
            tweets = []
            for hit in resp["hits"]["hits"]:
                # prefer normalized_text
                tweet_text = hit["_source"].get("normalized_text") or \
                             hit["_source"].get("content", "")
                if tweet_text:
                    tweets.append(tweet_text.strip())
            results[node] = tweets

            # debug preview for center node
            if node == center_node and tweets:
                print(f"[debug] Sample tweets for {node}:")
                for t in tweets[:3]:
                    print("   →", t[:200])

        except Exception as e:
            print(f"[error] failed to fetch tweets for {node}: {e}")
            results[node] = []

    return results


def fetch_community_texts_from_file(center_node, neighbors, filepath="res.json",
                                    size=5):
    """
    Alternative: Fetch tweets from the res.json file created by elastic.py
    This is more reliable since elastic.py already fetched all the data.
    """
    community_nodes = [center_node] + list(neighbors)
    results = {node: [] for node in community_nodes}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    username = doc.get("user_name")

                    if username in community_nodes:
                        # Get tweet text
                        text = doc.get("normalized_text") or doc.get("content", "")
                        if text and len(results[username]) < size:
                            results[username].append(text.strip())
                except json.JSONDecodeError:
                    continue

        # Debug output for center node
        if center_node in results and results[center_node]:
            print(f"[debug] Sample tweets for {center_node} from file:")
            for t in results[center_node][:3]:
                print("   →", t[:200])

        return results

    except FileNotFoundError:
        print(f"[error] {filepath} not found, falling back to ES query")
        return fetch_community_texts(center_node, neighbors, size=size)
    except Exception as e:
        print(f"[error] failed to read from {filepath}: {e}")
        return {node: [] for node in community_nodes}
