import sys
sys.path.insert(0, 'backend')

try:
    from api_with_real_model import app
    print('✅ API app imported successfully')
except Exception as e:
    print(f'❌ API import error: {e}')
    import traceback
    traceback.print_exc()
