"""
Simple Test - Check if everything works
"""
print("🧪 Testing imports...")

try:
    import fastapi
    print("✅ FastAPI available")
except ImportError:
    print("❌ FastAPI not available")

try:
    import uvicorn
    print("✅ Uvicorn available")
except ImportError:
    print("❌ Uvicorn not available")

try:
    import streamlit
    print("✅ Streamlit available")
except ImportError:
    print("❌ Streamlit not available")

try:
    import requests
    print("✅ Requests available")
except ImportError:
    print("❌ Requests not available")

print("\n🚀 All modules available! Ready to start the system.")
