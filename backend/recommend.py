def recommend_markets(class_id, valuation):
    """
    Recommend markets for selling based on class and valuation
    
    Args:
        class_id (int): The predicted class ID
        valuation (dict): Valuation dictionary with p05, p50, p95
        
    Returns:
        list: List of market recommendations
    """
    # TODO: Implement actual recommendation logic
    # Mock recommendations for now
    return [
        {"market": "Facebook Marketplace", "reason": "ราคาดี เหมาะสำหรับพระเครื่องทั่วไป"},
        {"market": "Shopee", "reason": "มีคนซื้อเยอะ"}
    ]
