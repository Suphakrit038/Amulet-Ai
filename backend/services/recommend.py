"""
🏺 Amulet-AI Advanced Recommendation System
Intelligent Market Recommendations with AI-Powered Insights
ระบบแนะนำตลาดอัจฉริยะพร้อม AI Analysis และการวิเคราะห์ลึก
"""
import random
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, time
import json

# Enhanced imports with fallbacks
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.app_config import get_config
except ImportError:
    get_config = lambda: type('Config', (), {'debug': False})()

try:
    import sys
    import logging
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from development.utils.logger import get_logger, performance_monitor, track_performance
        logger = get_logger("recommendations")
    except ImportError:
        logger = logging.getLogger(__name__)
        performance_monitor = lambda name: lambda func: func
        track_performance = lambda op: lambda: None
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    performance_monitor = lambda name: lambda func: func
    track_performance = lambda op: lambda: None

@dataclass
class MarketProfile:
    """Comprehensive market profile with detailed characteristics"""
    name: str
    location: Dict[str, float]  # lat, lng, distance
    specialty: str
    expertise: List[str]
    rating: float
    price_range: str
    operating_hours: Dict[str, str]
    contact_info: Dict[str, str]
    market_type: str
    reputation: str
    dealer_count: int
    avg_transaction_volume: str
    authenticity_guarantee: bool
    payment_methods: List[str]
    languages: List[str]
    tourist_friendly: bool

class AmuletRecommendationEngine:
    """🎯 Advanced AI-powered recommendation engine for Thai amulet markets"""
    
    def __init__(self):
        """Initialize the recommendation engine with comprehensive market database"""
        self.config = get_config()
        self._initialize_market_database()
        self._initialize_recommendation_algorithms()
        
        # Statistics tracking
        self.stats = {
            "recommendations_served": 0,
            "markets_analyzed": len(self.markets),
            "avg_recommendation_score": 0.0,
            "user_preferences": {}
        }
        
        logger.info("🚀 AmuletRecommendationEngine initialized with comprehensive market intelligence")
        logger.info(f"🏪 Market database: {len(self.markets)} venues loaded")
    
    def _initialize_market_database(self):
        """Initialize comprehensive market database with detailed profiles"""
        self.markets = [
            MarketProfile(
                name="ตลาดพระจตุจักร",
                location={"lat": 13.7563, "lng": 100.5018, "distance": 5.2},
                specialty="พระหลากหลายครบครัน",
                expertise=["somdej_fatherguay", "somdej_lion_base", "somdej_portrait_back", "somdej_prok_bodhi"],
                rating=4.5,
                price_range="mid_to_high",
                operating_hours={"weekday": "06:00-18:00", "weekend": "05:00-19:00"},
                contact_info={"phone": "02-272-4441", "line": "@jatujak-amulet"},
                market_type="traditional_market",
                reputation="established",
                dealer_count=150,
                avg_transaction_volume="high",
                authenticity_guarantee=True,
                payment_methods=["cash", "bank_transfer", "digital_wallet"],
                languages=["thai", "english", "chinese"],
                tourist_friendly=True
            ),
            MarketProfile(
                name="ตลาดพระสมเด็จเจ้าพระยา",
                location={"lat": 13.7441, "lng": 100.4986, "distance": 8.1},
                specialty="พระโบราณและพระหายาก",
                expertise=["somdej_fatherguay", "somdej_waek_man", "somdej_prok_bodhi"],
                rating=4.7,
                price_range="high_to_premium",
                operating_hours={"weekday": "07:00-17:00", "weekend": "06:00-18:00"},
                contact_info={"phone": "02-225-9999", "email": "info@somdej-market.th"},
                market_type="specialized_market",
                reputation="premium",
                dealer_count=80,
                avg_transaction_volume="very_high",
                authenticity_guarantee=True,
                payment_methods=["cash", "bank_transfer", "credit_card"],
                languages=["thai", "english"],
                tourist_friendly=True
            ),
            MarketProfile(
                name="ตลาดพระสราญรมย์",
                location={"lat": 13.7308, "lng": 100.5212, "distance": 12.5},
                specialty="พระหายากและของสะสม",
                expertise=["somdej_fatherguay", "somdej_waek_man", "wat_nong_e_duk"],
                rating=4.2,
                price_range="premium_to_luxury",
                operating_hours={"weekday": "08:00-17:00", "weekend": "07:00-18:00"},
                contact_info={"phone": "02-434-5678", "website": "www.saranrom-amulet.com"},
                market_type="collector_market",
                reputation="expert",
                dealer_count=45,
                avg_transaction_volume="very_high",
                authenticity_guarantee=True,
                payment_methods=["cash", "bank_transfer"],
                languages=["thai"],
                tourist_friendly=False
            ),
            MarketProfile(
                name="ตลาดพระวัดระฆัง",
                location={"lat": 13.7370, "lng": 100.5007, "distance": 6.8},
                specialty="พระเครื่องทั่วไป",
                expertise=["somdej_lion_base", "wat_nong_e_duk", "wat_nong_e_duk_misc"],
                rating=4.0,
                price_range="low_to_mid",
                operating_hours={"weekday": "06:30-17:30", "weekend": "06:00-18:30"},
                contact_info={"phone": "02-226-0335"},
                market_type="temple_market",
                reputation="authentic",
                dealer_count=60,
                avg_transaction_volume="medium",
                authenticity_guarantee=True,
                payment_methods=["cash"],
                languages=["thai", "english"],
                tourist_friendly=True
            ),
            MarketProfile(
                name="ตลาดพระปทุมวัน",
                location={"lat": 13.7479, "lng": 100.5380, "distance": 9.3},
                specialty="พระใหม่และของฝาก",
                expertise=["wat_nong_e_duk", "wat_nong_e_duk_misc"],
                rating=3.8,
                price_range="low_to_mid",
                operating_hours={"weekday": "09:00-20:00", "weekend": "08:00-21:00"},
                contact_info={"phone": "02-658-1234", "mall_info": "Platinum Fashion Mall B1"},
                market_type="modern_market",
                reputation="commercial",
                dealer_count=25,
                avg_transaction_volume="medium",
                authenticity_guarantee=False,
                payment_methods=["cash", "credit_card", "digital_wallet"],
                languages=["thai", "english", "chinese", "japanese"],
                tourist_friendly=True
            ),
            MarketProfile(
                name="ตลาดพระออนไลน์ ThaiAmulet",
                location={"lat": 0, "lng": 0, "distance": 0},
                specialty="พระเครื่องออนไลน์",
                expertise=["somdej_fatherguay", "somdej_lion_base", "somdej_portrait_back", "somdej_prok_bodhi", "somdej_waek_man"],
                rating=4.3,
                price_range="all_ranges",
                operating_hours={"weekday": "24/7", "weekend": "24/7"},
                contact_info={"website": "www.thaiamulet.com", "line": "@thaiamulet"},
                market_type="online_marketplace",
                reputation="modern",
                dealer_count=500,
                avg_transaction_volume="very_high",
                authenticity_guarantee=True,
                payment_methods=["bank_transfer", "digital_wallet", "credit_card", "crypto"],
                languages=["thai", "english"],
                tourist_friendly=True
            )
        ]
    
    def _initialize_recommendation_algorithms(self):
        """Initialize AI recommendation algorithms and scoring weights"""
        self.recommendation_weights = {
            "expertise_match": 0.35,      # How well market specializes in the amulet type
            "price_compatibility": 0.25,  # Price range alignment
            "distance": 0.15,             # Physical proximity
            "reputation": 0.10,           # Market reputation and rating
            "authenticity": 0.10,         # Authentication guarantee
            "user_experience": 0.05       # Tourist friendliness, languages, etc.
        }
        
        self.price_range_mapping = {
            "budget": ["low_to_mid", "low"],
            "standard": ["mid_to_high", "mid"],
            "premium": ["high_to_premium", "high"],
            "luxury": ["premium_to_luxury", "luxury"],
            "collector": ["premium_to_luxury", "all_ranges"]
        }

    @performance_monitor("comprehensive_recommendations")
    async def get_comprehensive_recommendations(
        self, 
        class_name: str, 
        valuation: Dict, 
        user_preferences: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        🔮 Get comprehensive market recommendations with AI-powered insights
        
        Args:
            class_name: Thai amulet class name
            valuation: Price valuation data
            user_preferences: User preferences for recommendations
            context: Additional context (location, budget, etc.)
        
        Returns:
            Dict containing recommendations, market analysis, and insights
        """
        
        # Base market database
        all_markets = [
            {
                "name": "ตลาดพระจตุจักร",
                "distance": 5.2,
                "specialty": "พระหลากหลาย",
                "rating": 4.5,
                "price_range": "mid_to_high",
                "expertise": ["somdej_fatherguay", "somdej_lion_base"]
            },
            {
                "name": "ตลาดพระสมเด็จเจ้าพระยา", 
                "distance": 8.1,
                "specialty": "พระโบราณ",
                "rating": 4.7,
                "price_range": "high",
                "expertise": ["somdej_fatherguay", "somdej_prok_bodhi"]
            },
            {
                "name": "ตลาดพระสราญรมย์",
                "distance": 12.5,
                "specialty": "พระหายาก",
                "rating": 4.2,
                "price_range": "very_high",
                "expertise": ["somdej_fatherguay"]
            },
            {
                "name": "ตลาดพระวัดระฆัง",
                "distance": 6.8,
                "specialty": "พระยุคใหม่",
                "rating": 4.0,
                "price_range": "low_to_mid",
                "expertise": ["wat_nong_e_duk", "wat_nong_e_duk_misc"]
            },
            {
                "name": "ตลาดพระอมรินทร์",
                "distance": 15.2,
                "specialty": "พระเครื่องแท้",
                "rating": 4.6,
                "price_range": "high",
                "expertise": ["somdej_lion_base", "somdej_fatherguay"]
            },
            {
                "name": "ตลาดนัดพระเสาร์-อาทิตย์",
                "distance": 7.3,
                "specialty": "พระราคาดี",
                "rating": 3.8,
                "price_range": "low_to_mid",
                "expertise": ["wat_nong_e_duk", "wat_nong_e_duk_misc", "somdej_lion_base"]
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
        0: "somdej_fatherguay",
        1: "somdej_prok_bodhi", 
        2: "somdej_lion_base",
        3: "wat_nong_e_duk"
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

# Global recommendation engine instance
_recommendation_engine = None

def get_recommendation_engine() -> AmuletRecommendationEngine:
    """Get singleton recommendation engine instance"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = AmuletRecommendationEngine()
    return _recommendation_engine

# Main function for external use
async def get_optimized_recommendations(class_name: str, valuation: Dict, 
                                      user_preferences: Optional[Dict] = None,
                                      context: Optional[Dict] = None) -> List[Dict]:
    """Get optimized recommendations - main interface function"""
    engine = get_recommendation_engine()
    result = await engine.get_comprehensive_recommendations(
        class_name, valuation, user_preferences, context
    )
    return result if isinstance(result, list) else result.get("recommendations", [])

# Convenience function for legacy compatibility  
def get_recommendations(class_name: str, valuation: Dict) -> List[Dict]:
    """Synchronous wrapper for legacy compatibility"""
    import asyncio
    try:
        return asyncio.run(get_optimized_recommendations(class_name, valuation))
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return recommend_markets(0, valuation)  # Fallback
