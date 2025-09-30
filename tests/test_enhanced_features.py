#!/usr/bin/env python3
"""
Demo และการทดสอบระบบ Memory Management และ Thread Safety
สำหรับ Amulet-AI
"""

import asyncio
import threading
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import io
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules ที่ปรับปรุงแล้ว
from core.memory_management import (
    memory_monitor, streaming_handler, gc_manager, 
    memory_efficient_operation, memory_limit_context
)
from core.thread_safety import (
    AtomicCounter, ThreadSafeDict, ThreadSafeQueue,
    thread_safe_operation, global_stats, lock_manager
)
from core.performance import image_cache, timed_operation, performance_monitor

def demo_memory_management():
    """Demo การใช้งาน Memory Management"""
    print("🧠 = Memory Management Demo =")
    
    # 1. Memory Monitoring
    print("1. Memory Usage Monitoring:")
    usage = memory_monitor.get_current_usage()
    print(f"   Current Memory: {usage.rss_mb:.1f}MB ({usage.percent:.1%})")
    
    # 2. Memory Pressure Detection
    pressure = memory_monitor.check_memory_pressure()
    print(f"   Memory Pressure Level: {pressure['level']}")
    print(f"   Should GC: {pressure['should_gc']}")
    
    # 3. Large File Streaming Demo
    print("\n2. Large File Streaming:")
    
    @memory_efficient_operation("large_file_demo")
    def process_large_data():
        # สร้างข้อมูลจำลองขนาดใหญ่
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB
        
        # ใช้ memory limit context
        with memory_limit_context(50):  # จำกัด 50MB
            processed_chunks = []
            chunk_size = 1024 * 1024  # 1MB chunks
            
            for i in range(0, len(large_data), chunk_size):
                chunk = large_data[i:i + chunk_size]
                processed_chunks.append(len(chunk))
                
                # ตรวจสอบ memory pressure
                if i % (5 * chunk_size) == 0:  # ทุก 5MB
                    pressure = memory_monitor.check_memory_pressure()
                    if pressure["should_gc"]:
                        print(f"   Triggering GC at {i // chunk_size}MB processed")
                        gc_manager.perform_cleanup()
            
            return sum(processed_chunks)
    
    total_processed = process_large_data()
    print(f"   Processed {total_processed // (1024*1024)}MB in chunks")
    
    # 4. Garbage Collection Demo
    print("\n3. Garbage Collection:")
    gc_result = gc_manager.perform_cleanup(force=True)
    if 'memory_freed_mb' in gc_result:
        print(f"   Memory freed: {gc_result['memory_freed_mb']:.1f}MB")
    
    # 5. Memory Statistics
    print("\n4. Memory Statistics:")
    stats = memory_monitor.get_stats()
    if 'current' in stats:
        current = stats['current']
        print(f"   Current RSS: {current['rss_mb']:.1f}MB")
        print(f"   Available: {current['available_mb']:.1f}MB")

def demo_thread_safety():
    """Demo การใช้งาน Thread Safety"""
    print("\n🔒 = Thread Safety Demo =")
    
    # 1. Atomic Counter Demo
    print("1. Atomic Counter Operations:")
    counter = AtomicCounter(0)
    
    def increment_worker(worker_id, iterations):
        for i in range(iterations):
            count = counter.increment()
            if i % 100 == 0:
                print(f"   Worker {worker_id}: Count = {count}")
    
    # เรียกใช้ multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=increment_worker, 
            args=(i, 500)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    final_count = counter.get()
    print(f"   Final atomic counter value: {final_count}")
    
    # 2. Thread-Safe Dictionary Demo
    print("\n2. Thread-Safe Dictionary:")
    safe_dict = ThreadSafeDict()
    
    def dict_worker(worker_id):
        for i in range(100):
            key = f"worker_{worker_id}_item_{i}"
            safe_dict.set(key, f"value_{i}")
        
        # อ่านข้อมูล
        item_count = 0
        for key in safe_dict.keys():
            if f"worker_{worker_id}" in key:
                item_count += 1
        
        print(f"   Worker {worker_id} added {item_count} items")
    
    # เรียกใช้ multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=dict_worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    print(f"   Total items in safe dict: {safe_dict.size()}")
    
    # 3. Thread-Safe Queue Demo
    print("\n3. Thread-Safe Queue:")
    safe_queue = ThreadSafeQueue(maxsize=50)
    
    def producer(producer_id):
        for i in range(20):
            item = f"item_{producer_id}_{i}"
            success = safe_queue.put(item, timeout=1.0)
            if not success:
                print(f"   Producer {producer_id}: Queue full!")
    
    def consumer(consumer_id):
        consumed = 0
        while consumed < 15:  # แต่ละ consumer จะดึง 15 items
            item = safe_queue.get(timeout=1.0)
            if item:
                consumed += 1
            else:
                break
        print(f"   Consumer {consumer_id} processed {consumed} items")
    
    # เริ่ม producers และ consumers
    threads = []
    
    # 2 producers
    for i in range(2):
        thread = threading.Thread(target=producer, args=(i,))
        threads.append(thread)
        thread.start()
    
    # 2 consumers
    for i in range(2):
        thread = threading.Thread(target=consumer, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    queue_stats = safe_queue.get_stats()
    print(f"   Queue final size: {queue_stats['current_size']}")
    print(f"   Items added: {queue_stats.get('items_added', 0)}")
    print(f"   Items retrieved: {queue_stats.get('items_retrieved', 0)}")

def demo_performance_optimization():
    """Demo การใช้งาน Performance Optimization"""
    print("\n⚡ = Performance Optimization Demo =")
    
    # 1. Image Caching Demo
    print("1. Image Caching:")
    
    # สร้างรูปภาพทดสอบ
    test_image = Image.new('RGB', (800, 600), color='red')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_bytes = img_buffer.getvalue()
    
    @timed_operation("image_cache_test")
    def test_image_caching():
        # ครั้งแรก - จะต้อง encode
        cache_key1 = image_cache.cache_image(img_bytes, "test_image_1.jpg")
        
        # ครั้งที่สอง - ควรได้จาก cache
        cache_key2 = image_cache.cache_image(img_bytes, "test_image_2.jpg")
        
        # ควรได้ cache key เดียวกัน (same hash)
        return cache_key1 == cache_key2
    
    is_cached = test_image_caching()
    print(f"   Image cached successfully: {is_cached}")
    
    # 2. Cache Statistics
    cache_stats = image_cache.get_stats()
    print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Cache size: {cache_stats['size']} images")
    print(f"   Total cache size: {cache_stats['total_size_bytes'] // 1024}KB")
    
    # 3. Performance Statistics
    print("\n2. Performance Statistics:")
    perf_stats = performance_monitor.get_stats("image_cache_test")
    if perf_stats.get('count', 0) > 0:
        print(f"   Operations: {perf_stats['count']}")
        print(f"   Average time: {perf_stats['avg_duration']:.4f}s")

@thread_safe_operation("demo_operation")
def thread_safe_demo_function(data):
    """ตัวอย่างฟังก์ชันที่ใช้ thread-safe decorator"""
    time.sleep(0.1)  # จำลองการประมวลผล
    return len(data)

def demo_integrated_features():
    """Demo การใช้งานแบบรวม"""
    print("\n🚀 = Integrated Features Demo =")
    
    # 1. Thread-safe operation with memory monitoring
    print("1. Thread-safe operations with memory monitoring:")
    
    def worker_with_monitoring(worker_id):
        # ใช้ memory monitoring
        memory_monitor.log_usage(f"worker_{worker_id}_start")
        
        # ทำงานที่ thread-safe
        result = thread_safe_demo_function(f"data_for_worker_{worker_id}")
        
        # เก็บ statistics
        global_stats.increment(f"worker_{worker_id}_completed")
        
        memory_monitor.log_usage(f"worker_{worker_id}_end")
        return result
    
    # เรียกใช้หลาย threads พร้อมกัน
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(6):
            future = executor.submit(worker_with_monitoring, i)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=5)
                results.append(result)
            except Exception as e:
                print(f"   Worker failed: {e}")
    
    print(f"   Completed {len(results)} workers successfully")
    
    # 2. Global statistics
    print("\n2. Global Statistics:")
    thread_stats = global_stats.get_stats()
    for key, value in thread_stats.items():
        if 'worker' in key:
            print(f"   {key}: {value}")

def main():
    """รันการทดสอบทั้งหมด"""
    print("🧪 = Amulet-AI Enhanced Features Testing =")
    print("=" * 50)
    
    try:
        # Memory Management Tests
        demo_memory_management()
        
        # Thread Safety Tests
        demo_thread_safety()
        
        # Performance Optimization Tests
        demo_performance_optimization()
        
        # Integrated Features Tests
        demo_integrated_features()
        
        print("\n✅ All tests completed successfully!")
        
        # สรุปผลรวม
        print("\n📊 = Final Summary =")
        print("Memory Usage:", memory_monitor.get_current_usage().rss_mb, "MB")
        print("Cache Performance:", image_cache.get_stats()['hit_rate'])
        print("Thread Operations:", len(global_stats.get_stats()))
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()