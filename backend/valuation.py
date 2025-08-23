"""
Optimized Valuation System
ระบบประเมินราคาที่ปรับปรุงใหม่
"""
import random
import logging
from typing import Dict, Optional

# Import config if available
try:
    from .config import price_config
except ImportError:
    # Fallback config
    class PriceConfig:
        base_prices = {
            "หลวงพ่อกวยแหวกม่าน": {"low": 15000, "mid": 45000, "high": 120000},
            "โพธิ์ฐานบัว": {"low": 8000, "mid": 25000, "high": 75000},
            "ฐานสิงห์": {"low": 12000, "mid": 35000, "high": 85000},
            "สีวลี": {"low": 5000, "mid": 18000, "high": 50000}
        }
        condition_multiplier = {"excellent": 1.3, "good": 1.0, "fair": 0.7, "poor": 0.4}
    
    price_config = PriceConfig()

logger = logging.getLogger(__name__)

async def get_optimized_valuation(class_name: str, confidence: float, condition: str = "good") -> Dict:
    """
    Get optimized price valuation with multiple factors
    """
    base_prices = price_config.base_prices.get(class_name, price_config.base_prices["หลวงพ่อกวยแหวกม่าน"])
    
    # Confidence factor (higher confidence = more accurate pricing)
    confidence_factor = 0.8 + (confidence * 0.4)  # 0.8 to 1.2 range
    
    # Condition factor
    condition_factor = price_config.condition_multiplier.get(condition, 1.0)
    
    # Market variation (±20%)
    market_variation = random.uniform(0.8, 1.2)
    
    # Calculate final prices
    final_factor = confidence_factor * condition_factor * market_variation
    
    p05 = int(base_prices["low"] * final_factor)
    p50 = int(base_prices["mid"] * final_factor)
    p95 = int(base_prices["high"] * final_factor)
    
    # Determine confidence level
    if confidence > 0.8:
        conf_level = "high"
    elif confidence > 0.6:
        conf_level = "medium"
    else:
        conf_level = "low"
    
    return {
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "confidence": conf_level,
        "condition_factor": condition_factor
    }

def get_quantiles(class_id: int) -> Dict:
    """Legacy function for backward compatibility"""
    
    # Map class_id to class_name
    class_mapping = {
        0: "หลวงพ่อกวยแหวกม่าน",
        1: "โพธิ์ฐานบัว",
        2: "ฐานสิงห์",
        3: "สีวลี"
    }
    
    class_name = class_mapping.get(class_id, "หลวงพ่อกวยแหวกม่าน")
    
    # Simple synchronous version for backward compatibility
    base_prices = price_config.base_prices.get(class_name, price_config.base_prices["หลวงพ่อกวยแหวกม่าน"])
    variation = random.uniform(0.8, 1.3)
    
    return {
        "p05": int(base_prices["low"] * variation),
        "p50": int(base_prices["mid"] * variation),
        "p95": int(base_prices["high"] * variation),
        "confidence": "high"
    }

def get_market_based_valuation(class_name: str):
    """
    Get valuation based on current market data
    
    Args:
        class_name (str): Name of the amulet class
        
    Returns:
        dict: Market-based valuation
    """
    try:
        from market_scraper import get_market_insights
        market_data = get_market_insights(class_name)
        
        avg_price = market_data.get("average_price", 5000)
        
        # Calculate quantiles based on market data
        # Assume normal distribution with std = 0.3 * mean
        std = avg_price * 0.3
        
        p05 = max(500, avg_price - 1.65 * std)  # 5th percentile
        p50 = avg_price                         # 50th percentile (median)
        p95 = avg_price + 1.65 * std           # 95th percentile
        
        return {
            "p05": float(p05),
            "p50": float(p50), 
            "p95": float(p95),
            "confidence": "market_based",
            "market_activity": market_data.get("market_activity", "unknown"),
            "sample_size": market_data.get("sample_size", 0)
        }
        
    except ImportError:
        logger.warning("⚠️ Market scraper not available")
        return None
    except Exception as e:
        logger.error(f"❌ Market valuation failed: {e}")
        return None

def get_market_based_valuation(class_name: str):
    """
    Get valuation based on current market data
    
    Args:
        class_name (str): Name of the amulet class
        
    Returns:
        dict: Market-based valuation
    """
    try:
        from market_scraper import get_market_insights
        market_data = get_market_insights(class_name)
        
        avg_price = market_data.get("average_price", 5000)
        
        # Calculate quantiles based on market data
        # Assume normal distribution with std = 0.3 * mean
        std = avg_price * 0.3
        
        p05 = max(500, avg_price - 1.65 * std)  # 5th percentile
        p50 = avg_price                         # 50th percentile (median)
        p95 = avg_price + 1.65 * std           # 95th percentile
        
        return {
            "p05": float(p05),
            "p50": float(p50), 
            "p95": float(p95),
            "confidence": "market_based",
            "market_activity": market_data.get("market_activity", "unknown"),
            "sample_size": market_data.get("sample_size", 0)
        }
        
    except ImportError:
        logger.warning("⚠️ Market scraper not available")
        return None
    except Exception as e:
        logger.error(f"❌ Market valuation failed: {e}")
        return None
