"""Feed configuration persistence module for multi-camera setup.

Handles saving/loading feed configurations to/from JSON files.
"""
import json
import uuid
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

DEFAULT_CONFIG_PATH = "feeds.json"

def generate_feed_id() -> str:
    """Generate a short unique ID for a new feed."""
    return str(uuid.uuid4())[:8]

def load_feeds(config_path: str = DEFAULT_CONFIG_PATH) -> List[Dict[str, Any]]:
    """Load feed configurations from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        List of feed configurations
    """
    config_file = Path(config_path)
    if not config_file.exists():
        return []
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Warning: Failed to load feeds from {config_path}")
        return []

def save_feeds(feeds: List[Dict[str, Any]], config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Save feed configurations to JSON file.
    
    Args:
        feeds: List of feed configurations
        config_path: Path to save the JSON configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(feeds, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving feeds: {e}")
        return False

def add_feed(feed_config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Add a new feed to the configuration file.
    
    Args:
        feed_config: Feed configuration dictionary
        config_path: Path to the JSON configuration file
        
    Returns:
        ID of the new feed
    """
    feeds = load_feeds(config_path)
    
    # Generate ID if not provided
    if 'id' not in feed_config:
        feed_config['id'] = generate_feed_id()
    
    # Add feed to list
    feeds.append(feed_config)
    save_feeds(feeds, config_path)
    
    return feed_config['id']

def update_feed(feed_id: str, feed_config: Dict[str, Any], config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Update an existing feed in the configuration file.
    
    Args:
        feed_id: ID of the feed to update
        feed_config: Updated feed configuration
        config_path: Path to the JSON configuration file
        
    Returns:
        True if successful, False if feed not found
    """
    feeds = load_feeds(config_path)
    
    for i, feed in enumerate(feeds):
        if feed.get('id') == feed_id:
            feeds[i] = {**feed, **feed_config}  # Update with new config
            save_feeds(feeds, config_path)
            return True
    
    return False

def remove_feed(feed_id: str, config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Remove a feed from the configuration file.
    
    Args:
        feed_id: ID of the feed to remove
        config_path: Path to the JSON configuration file
        
    Returns:
        True if successful, False if feed not found
    """
    feeds = load_feeds(config_path)
    
    for i, feed in enumerate(feeds):
        if feed.get('id') == feed_id:
            del feeds[i]
            save_feeds(feeds, config_path)
            return True
    
    return False

def convert_to_camera_manager_config(feed_json: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a feed JSON to CameraManager config format."""
    return {
        "id": feed_json.get("id", generate_feed_id()),
        "source": feed_json.get("source", "0"),
        "type": feed_json.get("type", "webcam"),
        "resolution": feed_json.get("resolution", [640, 480]),
        "fps_cap": feed_json.get("fps_cap", 15),
        "task": feed_json.get("task", {"type": "detect", "model": "models/yolov8n.pt"}),
    }
