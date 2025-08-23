"""
Enhanced valuation system with multiple data sources
Integrates with price estimator and market scraper for accurate pricing
"""
import logging

logger = logging.getLogger(__name__)

def get_quantiles(class_id):
    """
    Get valuation quantiles for a given class
    Enhanced version that tries to use ML models first
    
    Args:
        class_id (int): The predicted class ID
        
    Returns:
        dict: Dictionary with p05, p50, p95 values
    """
    # Map class_id to class_name
    class_names = {
        0: "หลวงพ่อกวยแหวกม่าน",
        1: "โพธิ์ฐานบัว",
        2: "ฐานสิงห์", 
        3: "สีวลี"
    }
    
    class_name = class_names.get(class_id, "unknown")
    
    # Try enhanced price estimation first
    try:
        from price_estimator import get_enhanced_price_estimation
        result = get_enhanced_price_estimation(class_id, class_name)
        logger.info("✅ Using enhanced price estimation for valuation")
        return result
    except ImportError:
        logger.info("⚠️ Enhanced price estimator not available, using fallback")
    except Exception as e:
        logger.warning(f"⚠️ Enhanced price estimation failed: {e}, using fallback")
    
    # Fallback to mock values with more realistic ranges
    price_ranges = {
        0: {"p05": 2000.0, "p50": 8000.0, "p95": 25000.0},  # หลวงพ่อกวยแหวกม่าน
        1: {"p05": 1500.0, "p50": 6000.0, "p95": 18000.0},  # โพธิ์ฐานบัว
        2: {"p05": 1200.0, "p50": 4500.0, "p95": 15000.0},  # ฐานสิงห์
        3: {"p05": 800.0, "p50": 3500.0, "p95": 12000.0},   # สีวลี
    }
    
    result = price_ranges.get(class_id, {"p05": 1000.0, "p50": 5000.0, "p95": 15000.0})
    result["confidence"] = "fallback"
    
    return result

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
