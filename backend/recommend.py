"""
Enhanced recommendation system for amulet selling
Integrates market data and pricing intelligence
"""
import logging

logger = logging.getLogger(__name__)

def recommend_markets(class_id, valuation):
    """
    Recommend markets for selling based on class and valuation
    Enhanced with market intelligence
    
    Args:
        class_id (int): The predicted class ID
        valuation (dict): Valuation dictionary with p05, p50, p95
        
    Returns:
        list: List of market recommendations
    """
    # Map class_id to class_name
    class_names = {
        0: "หลวงพ่อกวยแหวกม่าน",
        1: "โพธิ์ฐานบัว",
        2: "ฐานสิงห์",
        3: "สีวลี"
    }
    
    class_name = class_names.get(class_id, "unknown")
    price_range = valuation.get("p50", 5000)
    
    # Try to get market insights
    market_activity = "medium"  # default
    try:
        from market_scraper import get_market_insights
        market_data = get_market_insights(class_name)
        market_activity = market_data.get("market_activity", "medium")
        logger.info("✅ Using market data for recommendations")
    except ImportError:
        logger.info("⚠️ Market insights not available, using default recommendations")
    except Exception as e:
        logger.warning(f"⚠️ Market insights failed: {e}")
    
    recommendations = []
    
    # High-value items (> 10,000 baht)
    if price_range > 10000:
        recommendations.extend([
            {
                "market": "Facebook กลุ่มพระเครื่อง VIP", 
                "reason": "เหมาะสำหรับพระเครื่องราคาสูง มีผู้ซื้อที่มีกำลังซื้อดี",
                "priority": "high",
                "estimated_time": "7-14 วัน"
            },
            {
                "market": "Instagram พระเครื่องแท้", 
                "reason": "แพลตฟอร์มสำหรับคนรุ่นใหม่ ชอบสินค้าคุณภาพสูง",
                "priority": "high", 
                "estimated_time": "5-10 วัน"
            },
            {
                "market": "ตลาดนัดจตุจักร (วันเสาร์-อาทิทย์)",
                "reason": "ตลาดแบบดั้งเดิม มีผู้เชี่ยวชาญและนักสะสม",
                "priority": "medium",
                "estimated_time": "1-2 สัปดาห์"
            }
        ])
    
    # Medium-value items (3,000-10,000 baht)
    elif price_range > 3000:
        recommendations.extend([
            {
                "market": "Facebook Marketplace", 
                "reason": "ราคาดี เหมาะสำหรับพระเครื่องทั่วไป มีผู้ใช้งานเยอะ",
                "priority": "high",
                "estimated_time": "3-7 วัน"
            },
            {
                "market": "Shopee", 
                "reason": "มีคนซื้อเยอะ ระบบรีวิวช่วยสร้างความน่าเชื่อถือ",
                "priority": "high",
                "estimated_time": "1-5 วัน"
            },
            {
                "market": "LINE Official Account ร้านพระ",
                "reason": "ลูกค้าประจำ บริการส่วนตัว",
                "priority": "medium",
                "estimated_time": "1-3 วัน"
            }
        ])
    
    # Lower-value items (< 3,000 baht)  
    else:
        recommendations.extend([
            {
                "market": "Shopee", 
                "reason": "เหมาะสำหรับสินค้าราคาประหยัด มีผู้ซื้อเยอะ",
                "priority": "high",
                "estimated_time": "1-3 วัน"
            },
            {
                "market": "Facebook Marketplace",
                "reason": "ขายง่าย ไม่มีค่าธรรมเนียม",
                "priority": "high", 
                "estimated_time": "2-5 วัน"
            },
            {
                "market": "Lazada",
                "reason": "แพลตฟอร์มใหญ่ เหมาะสำหรับผู้ขายรายใหม่",
                "priority": "medium",
                "estimated_time": "3-7 วัน"
            }
        ])
    
    # Add class-specific recommendations
    if class_name == "หลวงพ่อกวยแหวกม่าน":
        recommendations.append({
            "market": "กลุ่ม LP Kuay แหวกม่าน Facebook",
            "reason": "กลุ่มเฉพาะสำหรับหลวงพ่อกวย มีผู้เชี่ยวชาญและนักสะสม",
            "priority": "high",
            "estimated_time": "1-7 วัน"
        })
    elif class_name == "สีวลี":
        recommendations.append({
            "market": "กลุ่มพระสีวลี Facebook",
            "reason": "ชุมชนคนรักพระสีวลี มีความรู้เฉพาะด้าน",
            "priority": "high", 
            "estimated_time": "1-7 วัน"
        })
    
    # Adjust recommendations based on market activity
    if market_activity == "high":
        # Add online marketplaces for high activity
        recommendations.insert(0, {
            "market": "TikTok Shop",
            "reason": "ตลาดร้อนแรง เทรนด์ใหม่ เหมาะสำหรับการโปรโมท",
            "priority": "high",
            "estimated_time": "1-3 วัน"
        })
    elif market_activity == "low":
        # Focus on traditional channels for low activity
        recommendations.append({
            "market": "ตลาดนัดท้องถิ่น",
            "reason": "ตลาดแบบดั้งเดิม เหมาะสำหรับช่วงตลาดเงียบ",
            "priority": "medium",
            "estimated_time": "1-2 สัปดาห์"
        })
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
    
    # Return top 5 recommendations
    return recommendations[:5]

def get_selling_tips(class_id, valuation):
    """
    Get specific selling tips for the amulet class
    
    Args:
        class_id (int): The predicted class ID
        valuation (dict): Valuation dictionary
        
    Returns:
        list: List of selling tips
    """
    class_names = {
        0: "หลวงพ่อกวยแหวกม่าน",
        1: "โพธิ์ฐานบัว", 
        2: "ฐานสิงห์",
        3: "สีวลี"
    }
    
    class_name = class_names.get(class_id, "unknown")
    
    general_tips = [
        "📸 ถ่ายรูปใสและสวยงาม แสงสว่างเพียงพอ",
        "📝 เขียนรายละเอียดครบถ้วน (วัด, ปี, ขนาด)",
        "🔍 แนบประวัติและเอกสารถ้ามี",
        "💰 ตั้งราคาตามตลาดในช่วงนั้น",
        "🤝 ตอบข้อความรวดเร็วและสุภาพ"
    ]
    
    class_specific_tips = {
        "หลวงพ่อกวยแหวกม่าน": [
            "🏛️ เน้นย้ำประวัติวัดหนองอีดุก",
            "📅 ระบุรุ่นและปีให้ชัดเจน", 
            "🔮 เน้นคุณค่าทางจิตวิญญาณ"
        ],
        "สีวลี": [
            "💼 เหมาะสำหรับคนทำธุรกิจ",
            "💎 เน้นความงาม และรายละเอียด",
            "🎯 กลุ่มเป้าหมายคือผู้ประกอบการ"
        ],
        "โพธิ์ฐานบัว": [
            "🌸 เน้นความสวยงามของฐานบัว",
            "📿 เหมาะสำหรับการสวดมนต์",
            "🎨 เน้นศิลปกรรมและช่างฝีมือ"
        ],
        "ฐานสิงห์": [
            "🦁 เน้นความโดดเด่นของฐานสิงห์",
            "👑 เน้นความสง่างามและเก่าแก่",
            "🏺 เหมาะสำหรับนักสะสมโบราณวัตถุ"
        ]
    }
    
    tips = general_tips.copy()
    if class_name in class_specific_tips:
        tips.extend(class_specific_tips[class_name])
    
    return tips
