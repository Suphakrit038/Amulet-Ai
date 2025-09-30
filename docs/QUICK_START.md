# 🚀 Amulet-AI Quick Start Guide

## การติดตั้งและการใช้งาน Enhanced Features

### 📋 ความต้องการของระบบ

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt
```

### ⚙️ การตั้งค่า Environment

```bash
# คัดลอก environment template
cp .env.example .env

# แก้ไขไฟล์ .env ตามความต้องการ
# สำคัญ: เปลี่ยน AMULET_SECRET_KEY ใน production!
```

### 🔧 การรันระบบ

#### 1. รัน API Server พร้อม Enhanced Features

```bash
# รัน API server พร้อม production monitoring
python production_runner.py api
```

**Features ที่เปิดใช้งาน:**
- ✅ **Memory Monitoring**: ตรวจสอบการใช้ memory แบบ real-time
- ✅ **Automatic GC**: Garbage collection อัตโนมัติเมื่อ memory เต็มจึง
- ✅ **Thread Safety**: Thread-safe operations ทั้งหมด
- ✅ **Performance Caching**: Image caching และ response caching
- ✅ **Error Recovery**: Retry mechanisms และ circuit breakers
- ✅ **Resource Cleanup**: Automatic cleanup เมื่อปิดระบบ

#### 2. รัน Frontend

```bash
# รัน Streamlit frontend
python production_runner.py frontend
```

**Enhanced Features:**
- ✅ **Session Caching**: Cache API responses ในเซสชัน
- ✅ **Memory-Efficient Image Handling**: จัดการรูปภาพขนาดใหญ่
- ✅ **Error Handling**: การจัดการ error ที่เป็นมิตรกับผู้ใช้
- ✅ **Performance Optimization**: Load CSS และ images แบบ cached

#### 3. ทดสอบ Enhanced Features

```bash
# รันการทดสอบระบบ
python production_runner.py test
```

### 📊 การตรวจสอบสถานะระบบ

#### Health Check API
```bash
curl http://localhost:8000/health
```

**Response ตัวอย่าง:**
```json
{
  "status": "healthy",
  "memory_stats": {
    "current": {"rss_mb": 245.6, "percent": 0.65},
    "pressure_check": {"level": "normal"}
  },
  "cache_stats": {
    "hit_rate": 0.85,
    "size": 450,
    "total_size_bytes": 95000000
  },
  "thread_stats": {
    "counters": {"api_calls": 1234, "errors": 5}
  },
  "api_metrics": {
    "total_requests": 1234,
    "error_rate": 0.004,
    "uptime_minutes": 120.5
  }
}
```

### 🔍 การใช้งาน Enhanced Features

#### 1. Memory Management

```python
from memory_management import memory_monitor, memory_limit_context

# ตรวจสอบการใช้ memory
usage = memory_monitor.get_current_usage()
print(f"Memory: {usage.rss_mb:.1f}MB ({usage.percent:.1%})")

# จำกัดการใช้ memory ในบางการทำงาน
with memory_limit_context(100):  # จำกัด 100MB
    # ทำงานที่ใช้ memory มาก
    result = process_large_data()
```

#### 2. Thread Safety

```python
from thread_safety import AtomicCounter, ThreadSafeDict, thread_safe_operation

# ใช้ atomic counter
counter = AtomicCounter()
count = counter.increment()

# ใช้ thread-safe dictionary
safe_data = ThreadSafeDict()
safe_data.set("key", "value")

# ใช้ thread-safe decorator
@thread_safe_operation("my_operation")
def critical_function():
    # โค้ดที่ต้องการ thread safety
    pass
```

#### 3. Performance Optimization

```python
from performance import image_cache, timed_operation

# Cache รูปภาพ
cache_key = image_cache.cache_image(image_bytes, "my_image.jpg")
cached_image = image_cache.get_image(cache_key)

# ตรวจสอบเวลาการทำงาน
@timed_operation("image_processing")
def process_image(image_data):
    # ประมวลผลรูปภาพ
    return processed_result
```

### 🛠️ Configuration Options

#### Memory Management
```bash
# ใน .env file
AMULET_MEMORY_WARNING_THRESHOLD=0.8    # เตือนที่ 80%
AMULET_MEMORY_CRITICAL_THRESHOLD=0.9   # Critical ที่ 90%
AMULET_MMAP_THRESHOLD=52428800         # ใช้ mmap สำหรับไฟล์ > 50MB
AMULET_LARGE_IMAGE_THRESHOLD=10485760  # Image ขนาดใหญ่ > 10MB
```

#### Cache Configuration
```bash
AMULET_CACHE_MAX_SIZE=104857600        # Cache size 100MB
AMULET_IMAGE_CACHE_SIZE=500            # เก็บ 500 รูป
AMULET_IMAGE_CACHE_TTL=1800            # TTL 30 นาที
```

#### Performance Settings
```bash
AMULET_CONNECTION_POOL_SIZE=100        # Connection pool
AMULET_THREAD_POOL_SIZE=4              # Thread pool
AMULET_CHUNK_SIZE=8192                 # Chunk size สำหรับ streaming
```

### 📈 การ Monitor ประสิทธิภาพ

#### 1. Memory Usage
```python
from memory_management import memory_monitor

# ดู memory stats
stats = memory_monitor.get_stats()
print(f"Current memory: {stats['current']['rss_mb']:.1f}MB")
print(f"Memory pressure: {stats['pressure_check']['level']}")
```

#### 2. Cache Performance
```python
from performance import image_cache

# ดู cache stats
stats = image_cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']} items")
```

#### 3. Thread Operations
```python
from thread_safety import global_stats

# ดู thread stats
stats = global_stats.get_stats()
for operation, count in stats.items():
    print(f"{operation}: {count}")
```

### 🚨 การแก้ไขปัญหา

#### ปัญหา Memory

```bash
# ถ้า memory ใช้มากเกินไป
# 1. ลด cache size
AMULET_CACHE_MAX_SIZE=52428800  # ลดเป็น 50MB

# 2. ลด image cache
AMULET_IMAGE_CACHE_SIZE=250     # ลดเป็น 250 รูป

# 3. เพิ่ม cleanup frequency
# แก้ไขใน memory_management.py line 223
self._cleanup_interval = 180    # ทุก 3 นาที
```

#### ปัญหา Performance

```bash
# ถ้าช้า
# 1. เพิ่ม connection pool
AMULET_CONNECTION_POOL_SIZE=200

# 2. เพิ่ม thread pool
AMULET_THREAD_POOL_SIZE=8

# 3. เพิ่ม cache TTL
AMULET_IMAGE_CACHE_TTL=3600     # 1 ชั่วโมง
```

#### ปัญหา Thread Safety

```python
# ถ้าเจอ race conditions
# ใช้ lock manager
from thread_safety import lock_manager

with lock_manager.acquire_lock("my_critical_section"):
    # โค้ดที่ต้องการ synchronization
    pass
```

### 🎯 Best Practices

#### 1. Production Deployment
- ตั้ง `AMULET_DEBUG=false`
- เปลี่ยน `AMULET_SECRET_KEY`
- ตั้งค่า memory thresholds ตาม server specs
- เปิด monitoring และ alerting

#### 2. Memory Management
- ใช้ `memory_limit_context` สำหรับการทำงานที่ใช้ memory มาก
- Monitor memory pressure เป็นประจำ
- ใช้ streaming สำหรับไฟล์ขนาดใหญ่

#### 3. Thread Safety
- ใช้ `AtomicCounter` แทน global counters
- ใช้ `ThreadSafeDict` สำหรับ shared data
- ใช้ `thread_safe_operation` decorator

#### 4. Performance
- Cache รูปภาพที่ใช้บ่อย
- ใช้ connection pooling
- Monitor cache hit rates

### 📞 การขอความช่วยเหลือ

#### Logs Location
- Error logs: `logs/amulet-ai.log` (ถ้าตั้งค่าไว้)
- Console output: รวม memory, cache, และ performance stats

#### Common Issues
1. **High Memory Usage**: ดู memory stats และปรับ cache settings
2. **Slow Performance**: ตรวจสอบ cache hit rates และ connection pools
3. **Thread Issues**: ดู thread stats และใช้ proper locking

#### Debug Mode
```bash
# เปิด debug mode
AMULET_DEBUG=true
AMULET_LOG_LEVEL=DEBUG
```

---

## 🎉 สรุป Enhanced Features

✅ **Memory Management**: Smart memory monitoring และ automatic cleanup  
✅ **Thread Safety**: Thread-safe operations และ data structures  
✅ **Performance**: Intelligent caching และ connection pooling  
✅ **Error Handling**: Comprehensive error management และ recovery  
✅ **Monitoring**: Real-time performance และ resource monitoring  

ระบบของคุณตอนนี้พร้อมสำหรับ production ด้วยความปลอดภัย ประสิทธิภาพ และความเสถียรที่สูง! 🚀