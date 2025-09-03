"""
Price Configuration for Amulet-AI System
การกำหนดค่าราคาและการประเมินมูลค่า
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import the main configuration
from .config import SystemConfig, get_config, config

# Re-export the price configuration from the main config
price_config = config.price

# Base price ranges (THB) - English category names
base_prices = {
    "somdej_fatherguay": {"low": 15000, "mid": 45000, "high": 120000, "ultra_rare": 300000},
    "somdej_portrait_back": {"low": 18000, "mid": 55000, "high": 180000, "ultra_rare": 400000},
    "somdej_lion_base": {"low": 12000, "mid": 35000, "high": 85000, "ultra_rare": 220000},
    "somdej_prok_bodhi": {"low": 25000, "mid": 75000, "high": 250000, "ultra_rare": 500000},
    "somdej_waek_man": {"low": 20000, "mid": 60000, "high": 200000, "ultra_rare": 450000},
    "wat_nong_e_duk": {"low": 8000, "mid": 22000, "high": 70000, "ultra_rare": 150000},
    "wat_nong_e_duk_misc": {"low": 5000, "mid": 18000, "high": 50000, "ultra_rare": 120000}
}

# Ensure the Thai category names are also available for backward compatibility
thai_base_prices = {
    "หลวงพ่อกวยแหวกม่าน": base_prices["somdej_fatherguay"],
    "โพธิ์ฐานบัว": base_prices["somdej_prok_bodhi"],
    "ฐานสิงห์": base_prices["somdej_lion_base"],
    "สีวลี": base_prices["wat_nong_e_duk"],
    "พระสมเด็จ": base_prices["somdej_fatherguay"]  # Default fallback
}

# Update the price config with both English and Thai names
price_config.base_prices.update(base_prices)
price_config.base_prices.update(thai_base_prices)

# Condition multipliers
condition_multiplier = price_config.condition_multiplier

# Rarity multipliers
rarity_multiplier = price_config.rarity_multiplier

# Market trend factors
market_trends = {
    "trending_up": 1.3,     # High demand, rising prices
    "stable": 1.0,          # Steady market
    "trending_down": 0.8,   # Declining interest
    "volatile": 1.1         # Unpredictable market
}

# Function to get configuration - reexport from main config
def get_config() -> SystemConfig:
    """Get the current system configuration"""
    from .config import get_config as main_get_config
    return main_get_config()

# Helper functions for price estimation
def get_base_price(class_name: str) -> Dict:
    """Get base price for a specific amulet class"""
    return price_config.base_prices.get(class_name, price_config.base_prices["somdej_fatherguay"])

def get_condition_factor(condition: str) -> float:
    """Get price multiplier for a specific condition"""
    return price_config.condition_multiplier.get(condition, 1.0)

def get_rarity_factor(rarity: str) -> float:
    """Get price multiplier for a specific rarity level"""
    return price_config.rarity_multiplier.get(rarity, 1.0)

def get_market_trend_factor(trend: str) -> float:
    """Get price multiplier for a specific market trend"""
    return market_trends.get(trend, 1.0)

def get_supported_classes() -> List[str]:
    """Get list of all supported amulet classes"""
    return list(price_config.base_prices.keys())

def get_condition_levels() -> List[str]:
    """Get list of all supported condition levels"""
    return list(price_config.condition_multiplier.keys())

def get_rarity_levels() -> List[str]:
    """Get list of all supported rarity levels"""
    return list(price_config.rarity_multiplier.keys())
