import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def save_analysis_results(results, source, date=None):
    """
    Save analysis results to a JSON file
    
    Args:
        results (list): List of analysis results
        source (str): News source (e.g., 'cnn', 'bbc')
        date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Create output directory structure
    output_dir = os.path.join("../data", "analysis", date)
    ensure_dir(output_dir)
    
    # Create filename
    filename = f"analysis_{source}_{date}.json"
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved analysis results to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save analysis results: {e}")
        return False


def load_articles(date=None):
    """
    Load all articles from processed directory for a specific date
    
    Args:
        date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
    
    Returns:
        dict: Dictionary of source -> articles
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    articles = {}
    processed_dir = os.path.join("../data", "processed", date)
    
    if not os.path.exists(processed_dir):
        logger.warning(f"No processed directory found for date: {date}")
        return articles
    
    for filename in os.listdir(processed_dir):
        if filename.startswith("articles_") and filename.endswith(".json"):
            source = filename.split("_")[1]  # Extract source from filename
            filepath = os.path.join(processed_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    articles[source] = json.load(f)
                logger.info(f"Loaded articles from {source}")
            except Exception as e:
                logger.error(f"Failed to load articles from {source}: {e}")
    
    return articles