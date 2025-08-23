"""
Optimized Recommendation System
ระบบแนะนำตลาดที่ปรับปรุงใหม่
"""
import random
from typing import List, Dict

async def get_optimized_recommendations(class_name: str, valuation: Dict) -> List[Dict]:
    """
    Get optimized market recommendations based on class and price range
    """
    
    # Base market database
    all_markets = [
        {
            "name": "ตลาดพระจตุจักร",
            "distance": 5.2,
            "specialty": "พระหลากหลาย",
            "rating": 4.5,
            "price_range": "mid_to_high",
            "expertise": ["หลวงพ่อกวยแหวกม่าน", "ฐานสิงห์"]
        },
        {
            "name": "ตลาดพระสมเด็จเจ้าพระยา", 
            "distance": 8.1,
            "specialty": "พระโบราณ",
            "rating": 4.7,
            "price_range": "high",
            "expertise": ["หลวงพ่อกวยแหวกม่าน", "โพธิ์ฐานบัว"]
        },
        {
            "name": "ตลาดพระสราญรมย์",
            "distance": 12.5,
            "specialty": "พระหายาก",
            "rating": 4.2,
            "price_range": "very_high",
            "expertise": ["หลวงพ่อกวยแหวกม่าน"]
        },
        {
            "name": "ตลาดพระวัดระฆัง",
            "distance": 6.8,
            "specialty": "พระยุคใหม่",
            "rating": 4.0,
            "price_range": "low_to_mid",
            "expertise": ["สีวลี", "โพธิ์ฐานบัว"]
        },
        {
            "name": "ตลาดพระอมรินทร์",
            "distance": 15.2,
            "specialty": "พระเครื่องแท้",
            "rating": 4.6,
            "price_range": "high",
            "expertise": ["ฐานสิงห์", "หลวงพ่อกวยแหวกม่าน"]
        },
        {
            "name": "ตลาดนัดพระเสาร์-อาทิตย์",
            "distance": 7.3,
            "specialty": "พระราคาดี",
            "rating": 3.8,
            "price_range": "low_to_mid",
            "expertise": ["สีวลี", "โพธิ์ฐานบัว", "ฐานสิงห์"]
        }
    ]
    
    # Determine price category
    median_price = valuation.get("p50", 25000)
    if median_price < 10000:
        price_category = "low_to_mid"
    elif median_price < 50000:
        price_category = "mid_to_high"
    elif median_price < 100000:
        price_category = "high"
    else:
        price_category = "very_high"
    
    # Score markets based on relevance
    scored_markets = []
    for market in all_markets:
        score = 0
        
        # Expertise matching
        if class_name in market["expertise"]:
            score += 40
        
        # Price range matching
        if market["price_range"] == price_category:
            score += 30
        elif abs(["low_to_mid", "mid_to_high", "high", "very_high"].index(market["price_range"]) - 
                 ["low_to_mid", "mid_to_high", "high", "very_high"].index(price_category)) == 1:
            score += 15
        
        # Distance factor (closer is better)
        if market["distance"] < 10:
            score += 20
        elif market["distance"] < 20:
            score += 10
        
        # Rating factor
        score += market["rating"] * 2
        
        # Add estimated price range
        price_ranges = {
            "low_to_mid": f"฿{median_price*0.7:,.0f} - ฿{median_price*1.2:,.0f}",
            "mid_to_high": f"฿{median_price*0.8:,.0f} - ฿{median_price*1.3:,.0f}",
            "high": f"฿{median_price*0.9:,.0f} - ฿{median_price*1.4:,.0f}",
            "very_high": f"฿{median_price*1.0:,.0f} - ฿{median_price*1.5:,.0f}"
        }
        
        market_info = {
            "name": market["name"],
            "distance": market["distance"],
            "specialty": market["specialty"],
            "rating": market["rating"],
            "estimated_price_range": price_ranges.get(market["price_range"], "ราคาแปรผัน"),
            "relevance_score": score
        }
        
        scored_markets.append(market_info)
    
    # Sort by relevance score and return top 3
    scored_markets.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Remove score from final result
    recommendations = []
    for market in scored_markets[:3]:
        market.pop("relevance_score")
        recommendations.append(market)
    
    return recommendations

# Legacy function for backward compatibility
def recommend_markets(class_id: int, valuation: Dict) -> List[Dict]:
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
    basic_markets = [
        {
            "name": "ตลาดพระจตุจักร",
            "distance": 5.2 + random.uniform(-1, 1),
            "specialty": "พระหลากหลาย",
            "rating": 4.5,
            "estimated_price_range": f"฿{valuation.get('p05', 15000):,.0f} - ฿{valuation.get('p95', 120000):,.0f}"
        },
        {
            "name": "ตลาดพระสมเด็จเจ้าพระยา",
            "distance": 8.1 + random.uniform(-1, 1), 
            "specialty": "พระโบราณ",
            "rating": 4.7,
            "estimated_price_range": f"฿{int(valuation.get('p50', 45000) * 0.8):,.0f} - ฿{int(valuation.get('p95', 120000) * 1.2):,.0f}"
        }
    ]
    
    return basic_markets
