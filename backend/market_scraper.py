"""
Web scraping system using Scrapy for collecting amulet market data
Scrapes prices, market trends, and other relevant data from various sources
"""
import scrapy
import json
import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class AmuletPriceSpider(scrapy.Spider):
    """Spider for scraping amulet prices from various sources"""
    
    name = 'amulet_prices'
    allowed_domains = ['example-amulet-site.com']  # Replace with actual domains
    
    def __init__(self, search_terms=None, *args, **kwargs):
        super(AmuletPriceSpider, self).__init__(*args, **kwargs)
        self.search_terms = search_terms or [
            'หลวงพ่อกวยแหวกม่าน',
            'โพธิ์ฐานบัว',
            'ฐานสิงห์',
            'พระสีวลี'
        ]
        self.scraped_data = []
    
    def start_requests(self):
        """Generate initial requests for each search term"""
        base_urls = [
            'https://www.example-market.com/search?q={}',
            'https://www.another-market.com/search?keyword={}',
            # Add more marketplace URLs here
        ]
        
        for term in self.search_terms:
            for url_template in base_urls:
                url = url_template.format(term)
                yield scrapy.Request(
                    url=url,
                    callback=self.parse_search_results,
                    meta={'search_term': term}
                )
    
    def parse_search_results(self, response):
        """Parse search results page"""
        search_term = response.meta['search_term']
        
        # Example selectors - adjust based on actual site structure
        item_links = response.css('.item-link::attr(href)').getall()
        
        for link in item_links:
            full_url = response.urljoin(link)
            yield scrapy.Request(
                url=full_url,
                callback=self.parse_item_details,
                meta={'search_term': search_term}
            )
        
        # Follow pagination
        next_page = response.css('.next-page::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse_search_results)
    
    def parse_item_details(self, response):
        """Parse individual item details"""
        search_term = response.meta['search_term']
        
        # Extract item details - adjust selectors based on actual site
        item_data = {
            'search_term': search_term,
            'title': self.clean_text(response.css('.item-title::text').get()),
            'price': self.extract_price(response.css('.price::text').get()),
            'description': self.clean_text(response.css('.description::text').get()),
            'seller': self.clean_text(response.css('.seller-name::text').get()),
            'location': self.clean_text(response.css('.location::text').get()),
            'date_posted': self.extract_date(response.css('.date-posted::text').get()),
            'url': response.url,
            'scraped_at': datetime.now().isoformat(),
            'images': response.css('.item-images img::attr(src)').getall(),
            'condition': self.extract_condition(response.css('.condition::text').get()),
            'temple': self.extract_temple_info(response.css('.item-details::text').getall())
        }
        
        # Validate required fields
        if item_data['price'] and item_data['title']:
            self.scraped_data.append(item_data)
            yield item_data
    
    def clean_text(self, text):
        """Clean extracted text"""
        if not text:
            return None
        return re.sub(r'\s+', ' ', text.strip())
    
    def extract_price(self, price_text):
        """Extract numeric price from text"""
        if not price_text:
            return None
        
        # Remove currency symbols and extract numbers
        price_match = re.search(r'[\d,]+', price_text.replace(',', ''))
        if price_match:
            try:
                return float(price_match.group().replace(',', ''))
            except ValueError:
                return None
        return None
    
    def extract_date(self, date_text):
        """Extract and standardize date"""
        if not date_text:
            return None
        
        # Add date parsing logic based on site format
        # This is a simple example
        return date_text.strip()
    
    def extract_condition(self, condition_text):
        """Extract item condition"""
        if not condition_text:
            return None
        
        condition_text = condition_text.lower()
        if any(word in condition_text for word in ['ใหม่', 'new']):
            return 'ใหม่'
        elif any(word in condition_text for word in ['เก่า', 'old', 'vintage']):
            return 'เก่า'
        else:
            return 'ใช้แล้ว'
    
    def extract_temple_info(self, details_list):
        """Extract temple information from item details"""
        if not details_list:
            return None
        
        details_text = ' '.join(details_list).lower()
        
        # Common temple patterns
        temple_patterns = [
            r'วัด[^\s]+',
            r'temple [^\s]+',
            r'พระ[^\s]+วัด[^\s]+',
        ]
        
        for pattern in temple_patterns:
            match = re.search(pattern, details_text)
            if match:
                return match.group()
        
        return None

class MarketDataCollector:
    """High-level interface for collecting market data"""
    
    def __init__(self, output_dir: str = "data/scraped"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_market_data(self, search_terms: List[str] = None) -> str:
        """
        Collect market data using Scrapy
        
        Args:
            search_terms: List of search terms to scrape
            
        Returns:
            Path to output file
        """
        if search_terms is None:
            search_terms = ['หลวงพ่อกวยแหวกม่าน', 'โพธิ์ฐานบัว', 'ฐานสิงห์', 'พระสีวลี']
        
        output_file = os.path.join(
            self.output_dir, 
            f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Note: In a real implementation, you would run Scrapy programmatically
        # For now, we'll create mock data
        mock_data = self._generate_mock_market_data(search_terms)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mock_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Market data saved to {output_file}")
        return output_file
    
    def _generate_mock_market_data(self, search_terms: List[str]) -> List[Dict]:
        """Generate mock market data for development"""
        import random
        
        mock_data = []
        
        for term in search_terms:
            for i in range(random.randint(10, 30)):
                base_price = {
                    'หลวงพ่อกวยแหวกม่าน': 8000,
                    'โพธิ์ฐานบัว': 6000,
                    'ฐานสิงห์': 4000,
                    'พระสีวลี': 3000
                }.get(term, 5000)
                
                mock_item = {
                    'search_term': term,
                    'title': f"{term} รุ่นพิเศษ ปี {random.randint(2520, 2565)}",
                    'price': base_price * random.uniform(0.5, 3.0),
                    'description': f"พระเครื่อง {term} สภาพดี มีประวัติ",
                    'seller': f"ร้าน{random.choice(['พระดี', 'เจ้าแม่', 'บุญนิมิต', 'ศรัดธา'])}",
                    'location': random.choice(['กรุงเทพฯ', 'นนทบุรี', 'สมุทรปราการ', 'ชลบุรี']),
                    'date_posted': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                    'condition': random.choice(['ใหม่', 'ใช้แล้ว', 'เก่า']),
                    'temple': f"วัด{random.choice(['แสงอรุณ', 'ใหญ่', 'เก่า', 'ใหม่', 'บูรพา'])}",
                    'scraped_at': datetime.now().isoformat()
                }
                
                mock_data.append(mock_item)
        
        return mock_data
    
    def load_market_data(self, file_path: str) -> List[Dict]:
        """Load market data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Could not load market data: {e}")
            return []
    
    def get_latest_market_data(self) -> List[Dict]:
        """Get the most recent market data file"""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.startswith('market_data_')]
            if not files:
                logger.info("🔄 No market data found, collecting new data")
                file_path = self.collect_market_data()
                return self.load_market_data(file_path)
            
            latest_file = max(files)
            file_path = os.path.join(self.output_dir, latest_file)
            return self.load_market_data(file_path)
            
        except Exception as e:
            logger.error(f"❌ Could not get market data: {e}")
            return []
    
    def analyze_market_trends(self, class_name: str) -> Dict:
        """
        Analyze market trends for specific amulet class
        
        Args:
            class_name: Name of amulet class
            
        Returns:
            Dictionary with market analysis
        """
        data = self.get_latest_market_data()
        
        # Filter data for specific class
        class_data = [item for item in data if item.get('search_term') == class_name]
        
        if not class_data:
            return {
                'average_price': 5000,
                'min_price': 1000,
                'max_price': 15000,
                'market_activity': 'low',
                'trend': 'stable',
                'sample_size': 0
            }
        
        prices = [item['price'] for item in class_data if isinstance(item.get('price'), (int, float))]
        
        if not prices:
            return {
                'average_price': 5000,
                'min_price': 1000,
                'max_price': 15000,
                'market_activity': 'low',
                'trend': 'stable',
                'sample_size': 0
            }
        
        return {
            'average_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'market_activity': 'high' if len(prices) > 20 else 'medium' if len(prices) > 10 else 'low',
            'trend': 'stable',  # Could implement trend analysis
            'sample_size': len(prices),
            'conditions': list(set(item.get('condition', 'ใช้แล้ว') for item in class_data)),
            'top_locations': list(set(item.get('location', 'กรุงเทพฯ') for item in class_data))[:5]
        }

# Integration functions
def get_market_insights(class_name: str) -> Dict:
    """
    Get market insights for API integration
    
    Args:
        class_name: Name of amulet class
        
    Returns:
        Market insights dictionary
    """
    collector = MarketDataCollector()
    return collector.analyze_market_trends(class_name)

def update_market_data():
    """Update market data collection"""
    collector = MarketDataCollector()
    collector.collect_market_data()
    logger.info("✅ Market data updated")

# Scrapy settings (if running programmatically)
SCRAPY_SETTINGS = {
    'ROBOTSTXT_OBEY': True,
    'DOWNLOAD_DELAY': 1,
    'RANDOMIZE_DOWNLOAD_DELAY': True,
    'CONCURRENT_REQUESTS': 1,
    'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
    'USER_AGENT': 'AmuletAI Bot 1.0',
    'DEFAULT_REQUEST_HEADERS': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'th,en',
    }
}
