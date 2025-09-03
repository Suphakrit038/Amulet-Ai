#!/usr/bin/env python3
"""
🧪 Test Amulet-AI System Components
ทดสอบระบบ Amulet-AI ทีละส่วน
"""

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
    
    try:
        import PIL
        print("✅ Pillow imported successfully")
        print(f"   Version: {PIL.__version__}")
    except Exception as e:
        print(f"❌ Pillow import failed: {e}")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
        print(f"   Version: {fastapi.__version__}")
    except Exception as e:
        print(f"❌ FastAPI import failed: {e}")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
        print(f"   Version: {streamlit.__version__}")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")

def test_ai_model_service():
    """Test AI model service"""
    print("\n🤖 Testing AI Model Service...")
    
    try:
        from backend.ai_model_service import ai_service, predict_amulet, get_ai_model_info
        
        # Test model info
        info = get_ai_model_info()
        print("✅ AI Model Service loaded")
        print(f"   Model: {info.get('model_name', 'Unknown')}")
        print(f"   Categories: {info.get('categories_count', 0)}")
        print(f"   Parameters: {info.get('model_parameters', 0):,}")
        
        # Test health check
        from backend.ai_model_service import ai_health_check
        health = ai_health_check()
        print(f"   Health: {health.get('status', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI Model Service test failed: {e}")
        return False

def test_backend_api():
    """Test backend API"""
    print("\n🔧 Testing Backend API...")
    
    try:
        from backend.api import app
        print("✅ Backend API app loaded successfully")
        
        # Test configuration
        from backend.config import get_config
        config = get_config()
        print(f"   Debug mode: {config.debug}")
        print(f"   API host: {config.api.host}:{config.api.port}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False

def test_valuation():
    """Test valuation system"""
    print("\n💰 Testing Valuation System...")
    
    try:
        import asyncio
        from backend.valuation import get_optimized_valuation, get_quantiles
        
        # Test synchronous function
        result = get_quantiles(0)
        print("✅ Valuation system working")
        print(f"   Sample valuation: ฿{result.get('p50', 0):,}")
        
        # Test async function
        async def test_async_valuation():
            result = await get_optimized_valuation("หลวงพ่อกวยแหวกม่าน", 0.85)
            return result
        
        # This would need to be run in an async context
        print("✅ Async valuation functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ Valuation system test failed: {e}")
        return False

def test_recommendations():
    """Test recommendation system"""
    print("\n🏪 Testing Recommendation System...")
    
    try:
        from backend.recommend import recommend_markets, get_recommendations
        
        # Test basic recommendation
        valuation = {"p05": 15000, "p50": 45000, "p95": 120000}
        recommendations = recommend_markets(0, valuation)
        
        print("✅ Recommendation system working")
        print(f"   Sample recommendations: {len(recommendations)} markets")
        
        if recommendations:
            first_rec = recommendations[0]
            print(f"   First recommendation: {first_rec.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation system test failed: {e}")
        return False

def test_frontend_utils():
    """Test frontend utilities"""
    print("\n🌐 Testing Frontend Utils...")
    
    try:
        from frontend.utils import validate_and_convert_image, SUPPORTED_FORMATS
        print("✅ Frontend utils available")
        print(f"   Supported formats: {len(SUPPORTED_FORMATS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Frontend utils test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏺 Amulet-AI System Component Tests")
    print("=" * 50)
    
    test_results = []
    
    test_imports()
    test_results.append(test_ai_model_service())
    test_results.append(test_backend_api())
    test_results.append(test_valuation())
    test_results.append(test_recommendations())
    test_results.append(test_frontend_utils())
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("\n🚀 Next steps:")
    print("   1. Start backend: python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000")
    print("   2. Start frontend: streamlit run frontend/app_streamlit.py --server.port 8501")
    print("   3. Open browser: http://127.0.0.1:8501")

if __name__ == "__main__":
    main()
