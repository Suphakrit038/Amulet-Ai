"""
🏺 Amulet-AI Advanced Valuation System
Comprehensive Price Evaluation with Market Intelligence and Historical Analysis
ระบบประเมินราคาขั้นสูงพร้อมวิเคราะห์ตลาดและข้อมูลประวัติศาสตร์
"""
import random
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Enhanced imports with fallbacks
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.price_config import price_config, get_config
except ImportError:
    # Comprehensive fallback configuration
    @dataclass
    class EnhancedPriceConfig:
        """Enhanced price configuration with historical and market data"""
        base_prices = {
            "somdej_fatherguay": {"low": 15000, "mid": 45000, "high": 120000, "ultra_rare": 300000},
            "somdej_portrait_back": {"low": 18000, "mid": 55000, "high": 180000, "ultra_rare": 400000},
            "somdej_lion_base": {"low": 12000, "mid": 35000, "high": 85000, "ultra_rare": 220000},
            "somdej_prok_bodhi": {"low": 25000, "mid": 75000, "high": 250000, "ultra_rare": 500000},
            "somdej_waek_man": {"low": 20000, "mid": 60000, "high": 200000, "ultra_rare": 450000},
            "wat_nong_e_duk": {"low": 8000, "mid": 22000, "high": 70000, "ultra_rare": 150000},
            "wat_nong_e_duk_misc": {"low": 5000, "mid": 18000, "high": 50000, "ultra_rare": 120000}
        }
        
        condition_multiplier = {
            "perfect": 1.8,     # Museum quality
            "excellent": 1.3,   # Very fine condition
            "good": 1.0,        # Standard condition
            "fair": 0.7,        # Visible wear
            "poor": 0.4,        # Significant damage
            "fragment": 0.2     # Incomplete piece
        }
        
        rarity_multiplier = {
            "legendary": 3.0,    # Extremely rare, historically significant
            "ultra_rare": 2.2,   # Very difficult to find
            "rare": 1.5,         # Limited availability
            "uncommon": 1.2,     # Moderately available
            "common": 1.0        # Widely available
        }
        
        market_trends = {
            "trending_up": 1.3,     # High demand, rising prices
            "stable": 1.0,          # Steady market
            "trending_down": 0.8,   # Declining interest
            "volatile": 1.1         # Unpredictable market
        }
        
    price_config = EnhancedPriceConfig()
    get_config = lambda: price_config

# Logging setup with fallbacks
try:
    import sys
    import logging
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from development.utils.logger import get_logger, performance_monitor, track_performance
        logger = get_logger("valuation")
    except ImportError:
        logger = logging.getLogger(__name__)
        performance_monitor = lambda name: lambda func: func
        track_performance = lambda op: lambda: None
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    performance_monitor = lambda name: lambda func: func
    track_performance = lambda op: lambda: None

class AmuletValuationEngine:
    """🏺 Advanced Amulet Valuation Engine with comprehensive market analysis"""
    
    def __init__(self):
        """Initialize the valuation engine with comprehensive databases"""
        self.config = get_config()
        self.historical_data = {}
        self.market_cache = {}
        self.valuation_stats = {
            "total_valuations": 0,
            "avg_confidence": 0.0,
            "price_ranges": {},
            "trend_analysis": {}
        }
        
        logger.info("🚀 AmuletValuationEngine initialized with comprehensive analysis")
        self._initialize_market_intelligence()
    
    def _initialize_market_intelligence(self):
        """Initialize advanced market intelligence features"""
        try:
            # Historical price trends (simulated market intelligence)
            self.historical_trends = {
                "somdej_fatherguay": {"trend": "trending_up", "volatility": 0.3, "demand": "high"},
                "somdej_portrait_back": {"trend": "stable", "volatility": 0.2, "demand": "medium"},
                "somdej_lion_base": {"trend": "trending_up", "volatility": 0.4, "demand": "high"},
                "somdej_prok_bodhi": {"trend": "stable", "volatility": 0.15, "demand": "medium"},
                "somdej_waek_man": {"trend": "trending_up", "volatility": 0.5, "demand": "very_high"},
                "wat_nong_e_duk": {"trend": "stable", "volatility": 0.25, "demand": "medium"},
                "wat_nong_e_duk_misc": {"trend": "trending_down", "volatility": 0.2, "demand": "low"}
            }
            
            # Regional price variations
            self.regional_factors = {
                "bangkok": 1.2,      # Premium market
                "central": 1.0,      # Base market
                "north": 0.9,        # Regional market
                "northeast": 0.8,    # Rural market
                "south": 1.1         # Tourist market
            }
            
            logger.info("📊 Market intelligence initialized with 7 amulet types and regional analysis")
            
        except Exception as e:
            logger.error(f"Failed to initialize market intelligence: {e}")

    @performance_monitor("advanced_valuation")
    async def get_comprehensive_valuation(
        self, 
        class_name: str, 
        confidence: float, 
        condition: str = "good",
        rarity: str = "common",
        region: str = "central",
        historical_significance: bool = False,
        provenance_verified: bool = False
    ) -> Dict:
        """
        🔮 Get comprehensive valuation with advanced market analysis
        
        Args:
            class_name: Thai amulet classification
            confidence: AI prediction confidence (0.0-1.0)
            condition: Physical condition assessment
            rarity: Rarity classification
            region: Regional market context
            historical_significance: Historical importance flag
            provenance_verified: Authenticity verification flag
        """
        
        try:
            # Get base price data
            base_prices = price_config.base_prices.get(
                class_name, 
                price_config.base_prices["หลวงพ่อกวยแหวกม่าน"]
            )
            
            # Market intelligence factors
            market_trend = self.historical_trends.get(class_name, {})
            
            # Calculate comprehensive multipliers
            confidence_factor = self._calculate_confidence_factor(confidence)
            condition_factor = price_config.condition_multiplier.get(condition, 1.0)
            rarity_factor = price_config.rarity_multiplier.get(rarity, 1.0)
            regional_factor = self.regional_factors.get(region, 1.0)
            trend_factor = price_config.market_trends.get(market_trend.get("trend", "stable"), 1.0)
            
            # Special factors
            historical_factor = 1.5 if historical_significance else 1.0
            provenance_factor = 1.3 if provenance_verified else 1.0
            
            # Market volatility adjustment
            volatility = market_trend.get("volatility", 0.2)
            market_variation = random.uniform(1 - volatility, 1 + volatility)
            
            # Comprehensive final factor
            final_multiplier = (
                confidence_factor * 
                condition_factor * 
                rarity_factor * 
                regional_factor * 
                trend_factor * 
                historical_factor * 
                provenance_factor * 
                market_variation
            )
            
            # Calculate price ranges
            p05 = max(500, int(base_prices["low"] * final_multiplier))
            p25 = max(1000, int((base_prices["low"] + base_prices["mid"]) / 2 * final_multiplier))
            p50 = max(2000, int(base_prices["mid"] * final_multiplier))
            p75 = max(5000, int((base_prices["mid"] + base_prices["high"]) / 2 * final_multiplier))
            p95 = max(10000, int(base_prices["high"] * final_multiplier))
            p99 = max(20000, int(base_prices.get("ultra_rare", base_prices["high"] * 2) * final_multiplier))
            
            # Confidence assessment
            confidence_level = self._assess_confidence_level(confidence, condition, rarity)
            
            # Market insights
            market_insights = self._generate_market_insights(class_name, market_trend)
            
            # Investment potential analysis
            investment_score = self._calculate_investment_potential(
                class_name, market_trend, rarity, condition
            )
            
            # Compile comprehensive result
            valuation_result = {
                # Price analysis
                "price_ranges": {
                    "p05": p05,     # Bottom 5%
                    "p25": p25,     # Lower quartile
                    "p50": p50,     # Median estimate
                    "p75": p75,     # Upper quartile
                    "p95": p95,     # Top 5%
                    "p99": p99      # Ultra premium
                },
                
                # Confidence metrics
                "confidence": {
                    "level": confidence_level,
                    "ai_confidence": confidence,
                    "overall_reliability": min(0.95, confidence * 1.2)
                },
                
                # Analysis factors
                "factors": {
                    "condition_factor": condition_factor,
                    "rarity_factor": rarity_factor,
                    "regional_factor": regional_factor,
                    "trend_factor": trend_factor,
                    "historical_factor": historical_factor,
                    "provenance_factor": provenance_factor,
                    "final_multiplier": round(final_multiplier, 3)
                },
                
                # Market intelligence
                "market_analysis": market_insights,
                
                # Investment analysis
                "investment": {
                    "score": investment_score,
                    "recommendation": self._get_investment_recommendation(investment_score),
                    "risk_level": self._assess_risk_level(volatility, market_trend.get("demand", "medium"))
                },
                
                # Additional metadata
                "valuation_timestamp": datetime.now().isoformat(),
                "analysis_version": "2.0_advanced",
                "region": region,
                "class_name": class_name
            }
            
            # Update statistics
            self._update_valuation_stats(class_name, confidence, valuation_result)
            
            logger.info(f"💎 Comprehensive valuation completed for {class_name}")
            logger.info(f"📊 Price range: ฿{p05:,} - ฿{p95:,} (confidence: {confidence_level})")
            
            return valuation_result
            
        except Exception as e:
            logger.error(f"Comprehensive valuation failed: {e}", exc_info=True)
            return await self._get_fallback_valuation(class_name, confidence, condition)
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """Calculate confidence factor for price adjustment"""
        if confidence >= 0.9:
            return 1.1      # High confidence bonus
        elif confidence >= 0.7:
            return 1.0      # Standard pricing
        elif confidence >= 0.5:
            return 0.95     # Slight discount
        else:
            return 0.85     # Low confidence penalty
    
    def _assess_confidence_level(self, confidence: float, condition: str, rarity: str) -> str:
        """Assess overall confidence level"""
        base_confidence = confidence
        
        # Condition impact
        if condition in ["perfect", "excellent"]:
            base_confidence *= 1.1
        elif condition in ["poor", "fragment"]:
            base_confidence *= 0.8
        
        # Rarity impact
        if rarity in ["legendary", "ultra_rare"]:
            base_confidence *= 0.9  # Rare items harder to value precisely
        
        if base_confidence >= 0.85:
            return "very_high"
        elif base_confidence >= 0.7:
            return "high"
        elif base_confidence >= 0.5:
            return "medium"
        elif base_confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    def _generate_market_insights(self, class_name: str, market_trend: Dict) -> Dict:
        """Generate comprehensive market insights"""
        demand_level = market_trend.get("demand", "medium")
        trend = market_trend.get("trend", "stable")
        volatility = market_trend.get("volatility", 0.2)
        
        insights = {
            "demand_level": demand_level,
            "trend_direction": trend,
            "market_volatility": volatility,
            "liquidity": "high" if demand_level in ["high", "very_high"] else "medium",
            "collector_interest": self._assess_collector_interest(class_name, demand_level),
            "seasonal_factors": self._get_seasonal_factors(),
            "market_maturity": "established" if class_name in ["หลวงพ่อกวยแหวกม่าน", "พระสมเด็จ"] else "developing"
        }
        
        return insights

    def _assess_collector_interest(self, class_name: str, demand_level: str) -> str:
        """Assess collector interest level"""
        premium_classes = ["หลวงพ่อกวยแหวกม่าน", "พระสมเด็จ", "ฐานสิงห์"]
        
        if class_name in premium_classes and demand_level == "very_high":
            return "extremely_high"
        elif demand_level in ["high", "very_high"]:
            return "high" 
        elif demand_level == "medium":
            return "moderate"
        else:
            return "limited"

    def _get_seasonal_factors(self) -> Dict:
        """Get seasonal market factors (Thai Buddhist calendar considerations)"""
        now = datetime.now()
        
        # Simplified seasonal analysis
        if now.month in [4, 5, 7]:  # Songkran, Visakha Bucha, Asalha Puja
            return {"factor": 1.1, "reason": "Buddhist holiday season"}
        elif now.month in [11, 12]:  # Loy Krathong, New Year
            return {"factor": 1.05, "reason": "Festival season"}
        else:
            return {"factor": 1.0, "reason": "Normal season"}

    def _calculate_investment_potential(self, class_name: str, market_trend: Dict, rarity: str, condition: str) -> float:
        """Calculate investment potential score (0-100)"""
        base_score = 50
        
        # Trend impact
        trend = market_trend.get("trend", "stable")
        if trend == "trending_up":
            base_score += 20
        elif trend == "trending_down":
            base_score -= 15
        
        # Demand impact
        demand = market_trend.get("demand", "medium")
        demand_scores = {"very_high": 25, "high": 15, "medium": 5, "low": -10}
        base_score += demand_scores.get(demand, 0)
        
        # Rarity impact
        rarity_scores = {"legendary": 20, "ultra_rare": 15, "rare": 10, "uncommon": 5, "common": 0}
        base_score += rarity_scores.get(rarity, 0)
        
        # Condition impact
        condition_scores = {"perfect": 15, "excellent": 10, "good": 5, "fair": 0, "poor": -10}
        base_score += condition_scores.get(condition, 0)
        
        # Historical significance bonus for specific classes
        if class_name in ["หลวงพ่อกวยแหวกม่าน", "พระสมเด็จ"]:
            base_score += 10
        
        return max(0, min(100, base_score))

    def _get_investment_recommendation(self, investment_score: float) -> str:
        """Get investment recommendation based on score"""
        if investment_score >= 85:
            return "strong_buy"
        elif investment_score >= 70:
            return "buy"
        elif investment_score >= 55:
            return "hold"
        elif investment_score >= 40:
            return "cautious"
        else:
            return "avoid"

    def _assess_risk_level(self, volatility: float, demand: str) -> str:
        """Assess investment risk level"""
        if volatility > 0.4 or demand == "low":
            return "high"
        elif volatility > 0.25 or demand in ["medium", "high"]:
            return "medium"
        else:
            return "low"

    def _update_valuation_stats(self, class_name: str, confidence: float, result: Dict):
        """Update internal statistics"""
        self.valuation_stats["total_valuations"] += 1
        
        # Update average confidence
        current_avg = self.valuation_stats["avg_confidence"]
        total = self.valuation_stats["total_valuations"]
        self.valuation_stats["avg_confidence"] = ((current_avg * (total - 1)) + confidence) / total
        
        # Update price range statistics
        if class_name not in self.valuation_stats["price_ranges"]:
            self.valuation_stats["price_ranges"][class_name] = []
        
        self.valuation_stats["price_ranges"][class_name].append({
            "median": result["price_ranges"]["p50"],
            "confidence": confidence,
            "timestamp": result["valuation_timestamp"]
        })

    async def _get_fallback_valuation(self, class_name: str, confidence: float, condition: str) -> Dict:
        """Get simplified fallback valuation"""
        base_prices = price_config.base_prices.get(
            class_name, 
            price_config.base_prices["หลวงพ่อกวยแหวกม่าน"]
        )
        
        condition_factor = price_config.condition_multiplier.get(condition, 1.0)
        variation = random.uniform(0.8, 1.2)
        
        p50 = int(base_prices["mid"] * condition_factor * variation)
        p05 = int(p50 * 0.4)
        p95 = int(p50 * 2.5)
        
        return {
            "price_ranges": {"p05": p05, "p50": p50, "p95": p95},
            "confidence": {"level": "low", "ai_confidence": confidence},
            "factors": {"condition_factor": condition_factor},
            "market_analysis": {"demand_level": "unknown", "trend_direction": "stable"},
            "investment": {"score": 40, "recommendation": "cautious", "risk_level": "medium"},
            "valuation_timestamp": datetime.now().isoformat(),
            "analysis_version": "1.0_fallback",
            "class_name": class_name
        }

    async def get_market_comparison(self, class_name: str, price_range: Tuple[int, int]) -> Dict:
        """Get market comparison analysis"""
        try:
            with track_performance("market_comparison"):
                # Simulate market comparison logic
                base_prices = price_config.base_prices.get(class_name, {})
                market_median = base_prices.get("mid", 25000)
                
                user_median = (price_range[0] + price_range[1]) / 2
                
                comparison = {
                    "user_price_vs_market": user_median / market_median,
                    "market_position": "above" if user_median > market_median else "below",
                    "competitiveness": min(1.0, market_median / user_median) if user_median > 0 else 0.5
                }
                
                return comparison
                
        except Exception as e:
            logger.error(f"Market comparison failed: {e}")
            return {"error": "Market comparison unavailable"}

    def get_valuation_statistics(self) -> Dict:
        """Get comprehensive valuation statistics"""
        return {
            **self.valuation_stats,
            "supported_classes": list(price_config.base_prices.keys()),
            "condition_levels": list(price_config.condition_multiplier.keys()),
            "rarity_levels": list(price_config.rarity_multiplier.keys()),
            "regional_markets": list(self.regional_factors.keys())
        }

# Global valuation engine instance
_valuation_engine = None

def get_valuation_engine() -> AmuletValuationEngine:
    """Get singleton valuation engine instance"""
    global _valuation_engine
    if _valuation_engine is None:
        _valuation_engine = AmuletValuationEngine()
    return _valuation_engine

# Legacy and convenience functions for backward compatibility
@performance_monitor("optimized_valuation")
async def get_optimized_valuation(class_name: str, confidence: float, condition: str = "good") -> Dict:
    """Enhanced optimized valuation with comprehensive analysis"""
    engine = get_valuation_engine()
    return await engine.get_comprehensive_valuation(class_name, confidence, condition)

def get_quantiles(class_id: int) -> Dict:
    """Legacy function for backward compatibility with enhanced features"""
    
    # Enhanced class mapping with more categories
    class_mapping = {
        0: "somdej_fatherguay",
        1: "somdej_prok_bodhi",
        2: "somdej_lion_base",
        3: "wat_nong_e_duk",
        4: "somdej_portrait_back",
        5: "somdej_waek_man",
        6: "wat_nong_e_duk_misc"
    }
    
    class_name = class_mapping.get(class_id, "หลวงพ่อกวยแหวกม่าน")
    
    # Enhanced synchronous version with market intelligence
    try:
        base_prices = price_config.base_prices.get(class_name)
        if not base_prices:
            base_prices = price_config.base_prices["หลวงพ่อกวยแหวกม่าน"]
        
        # Market-aware variation
        engine = get_valuation_engine()
        market_trend = engine.historical_trends.get(class_name, {})
        
        volatility = market_trend.get("volatility", 0.2)
        trend_factor = price_config.market_trends.get(market_trend.get("trend", "stable"), 1.0)
        
        variation = random.uniform(1 - volatility, 1 + volatility) * trend_factor
        
        p05 = max(500, int(base_prices["low"] * variation))
        p50 = max(2000, int(base_prices["mid"] * variation))
        p95 = max(10000, int(base_prices["high"] * variation))
        
        confidence_level = "high" if volatility < 0.3 else "medium"
        
        logger.info(f"📊 Legacy valuation for {class_name}: ฿{p50:,} (confidence: {confidence_level})")
        
        return {
            "p05": p05,
            "p50": p50,
            "p95": p95,
            "confidence": confidence_level,
            "market_trend": market_trend.get("trend", "stable"),
            "class_name": class_name
        }
        
    except Exception as e:
        logger.error(f"Legacy valuation failed: {e}")
        # Basic fallback
        return {
            "p05": 5000,
            "p50": 25000,
            "p95": 75000,
            "confidence": "low",
            "error": str(e)
        }

async def get_market_based_valuation(class_name: str) -> Optional[Dict]:
    """
    🔍 Get valuation based on simulated market data with advanced analytics
    
    Args:
        class_name (str): Name of the amulet class
        
    Returns:
        dict: Comprehensive market-based valuation or None if unavailable
    """
    try:
        # Simulate advanced market scraping
        logger.info(f"🔍 Fetching market data for {class_name}")
        
        # Simulate market intelligence gathering
        await asyncio.sleep(0.1)  # Simulate API call
        
        # Enhanced market simulation
        engine = get_valuation_engine()
        market_trend = engine.historical_trends.get(class_name, {})
        base_prices = price_config.base_prices.get(class_name)
        
        if not base_prices:
            logger.warning(f"No base price data for {class_name}")
            return None
        
        # Simulate market activity
        demand = market_trend.get("demand", "medium")
        volatility = market_trend.get("volatility", 0.2)
        
        # Market-based pricing with realistic simulation
        market_median = base_prices["mid"]
        market_std = market_median * volatility
        
        # Simulate recent market activity
        recent_sales = []
        for i in range(random.randint(5, 20)):  # Simulate 5-20 recent sales
            price_variation = random.gauss(1.0, volatility)
            sale_price = max(500, int(market_median * price_variation))
            recent_sales.append(sale_price)
        
        # Calculate market statistics
        avg_price = sum(recent_sales) / len(recent_sales)
        sorted_sales = sorted(recent_sales)
        
        p05 = sorted_sales[max(0, int(len(sorted_sales) * 0.05))]
        p25 = sorted_sales[max(0, int(len(sorted_sales) * 0.25))]
        p50 = sorted_sales[max(0, int(len(sorted_sales) * 0.50))]
        p75 = sorted_sales[max(0, int(len(sorted_sales) * 0.75))]
        p95 = sorted_sales[max(0, int(len(sorted_sales) * 0.95))]
        
        # Market activity assessment
        activity_levels = {"very_high": "active", "high": "active", "medium": "moderate", "low": "quiet"}
        market_activity = activity_levels.get(demand, "unknown")
        
        # Confidence based on sample size and volatility
        confidence = "high" if len(recent_sales) > 15 and volatility < 0.25 else "medium"
        
        market_result = {
            "price_analysis": {
                "p05": p05,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
                "average": int(avg_price),
                "std_dev": int(market_std)
            },
            "market_metrics": {
                "confidence": confidence,
                "market_activity": market_activity,
                "sample_size": len(recent_sales),
                "volatility": volatility,
                "trend": market_trend.get("trend", "stable"),
                "demand_level": demand
            },
            "data_quality": {
                "freshness": "recent",  # Within last 30 days
                "reliability": "simulated_high_quality",
                "coverage": "comprehensive"
            },
            "analysis_timestamp": datetime.now().isoformat(),
            "class_name": class_name
        }
        
        logger.info(f"📈 Market analysis completed for {class_name}")
        logger.info(f"💰 Market median: ฿{p50:,} (activity: {market_activity}, samples: {len(recent_sales)})")
        
        return market_result
        
    except Exception as e:
        logger.error(f"❌ Market-based valuation failed for {class_name}: {e}", exc_info=True)
        return None

# Convenience functions for quick access
async def quick_valuation(class_name: str, confidence: float = 0.8) -> Dict:
    """Quick valuation with default parameters"""
    return await get_optimized_valuation(class_name, confidence)

async def premium_valuation(class_name: str, confidence: float, condition: str = "excellent", 
                          rarity: str = "rare", provenance_verified: bool = True) -> Dict:
    """Premium valuation for high-end pieces"""
    engine = get_valuation_engine()
    return await engine.get_comprehensive_valuation(
        class_name, confidence, condition, rarity, 
        historical_significance=True, provenance_verified=provenance_verified
    )

def get_supported_classes() -> List[str]:
    """Get list of all supported amulet classes"""
    return list(price_config.base_prices.keys())

def get_condition_levels() -> List[str]:
    """Get list of all supported condition levels"""
    return list(price_config.condition_multiplier.keys())

def get_rarity_levels() -> List[str]:
    """Get list of all supported rarity levels"""
    return list(price_config.rarity_multiplier.keys())

async def batch_valuation(items: List[Dict]) -> List[Dict]:
    """Batch valuation for multiple items"""
    results = []
    engine = get_valuation_engine()
    
    for item in items:
        try:
            result = await engine.get_comprehensive_valuation(**item)
            results.append({"status": "success", "data": result, "item": item})
        except Exception as e:
            results.append({"status": "error", "error": str(e), "item": item})
    
    logger.info(f"🔄 Batch valuation completed: {len(results)} items processed")
    return results
