import logging
from datetime import datetime
import os
from nlp_processor import NLPProcessor
from crawler.util import save_json, load_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_all_articles(date=None):
    """
    Process all articles for a given date and save the results
    
    Args:
        date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize NLP processor
    nlp_processor = NLPProcessor()
    
    # Directory paths
    processed_dir = os.path.join("../data", "processed", date)
    analysis_dir = os.path.join("../data", "analysis", date)
    
    if not os.path.exists(processed_dir):
        logger.warning(f"No processed directory found for date: {date}")
        return
    
    # Process each source's articles
    for filename in os.listdir(processed_dir):
        if filename.startswith("articles_") and filename.endswith(".json"):
            source = filename.split("_")[1]  # Extract source from filename
            logger.info(f"Processing articles from {source}")
            
            # Load articles
            articles = load_json(os.path.join(processed_dir, filename))
            if not articles:
                logger.warning(f"No articles loaded from {source}")
                continue
            
            results = []
            for article in articles:
                try:
                    # Process article
                    result = nlp_processor.process_article(article)
                    
                    if result:
                        # Add original article info to result
                        result['article_info'] = {
                            'title': article.get('title'),
                            'url': article.get('url'),
                            'published': article.get('published'),
                            'source': article.get('source')
                        }
                        results.append(result)
                        logger.info(f"Successfully processed article: {article.get('title')}")
                    else:
                        logger.warning(f"Failed to process article: {article.get('title')}")
                
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
            
            # Save results
            if results:
                output_filename = f"analysis_{source}_{date}.json"
                output_path = os.path.join(analysis_dir, output_filename)
                save_json(results, output_path)
                logger.info(f"Saved {len(results)} analysis results for {source}")
            else:
                logger.warning(f"No valid results to save for {source}")

if __name__ == "__main__":
    # Process articles for today
    process_all_articles()
    
    # To process articles for a specific date:
    # process_all_articles("2025-04-15") 