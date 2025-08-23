# Amulet-AI

## 🌟 คำนิยาม
Amulet-AI คือแพลตฟอร์มอัจฉริยะที่ใช้เทคโนโลยีปัญญาประดิษฐ์ (AI) ในการวิเคราะห์และจำแนกประเภทพระเครื่องไทย พร้อมทั้งให้ข้อมูลที่เป็นประโยชน์ เช่น ความน่าจะเป็นของรุ่น/พิมพ์ และคำแนะนำที่เกี่ยวข้อง เพื่อช่วยให้ผู้ใช้งานสามารถตัดสินใจได้อย่างมั่นใจ

## 🖥️ ชื่อเว็บ
**Amulet-AI — วิเคราะห์พระเครื่องลึกลับ**

## ✨ คุณสมบัติเด่น
- **อัปโหลดรูปภาพ**: รองรับการอัปโหลดภาพด้านหน้าและด้านหลังของพระเครื่อง
- **การวิเคราะห์ด้วย AI**: ใช้โมเดล TensorFlow ในการจำแนกประเภทพระเครื่อง
- **ผลลัพธ์ที่เข้าใจง่าย**: แสดงผลลัพธ์ Top-3 พร้อมความน่าจะเป็น
- **อินเทอร์เฟซที่ใช้งานง่าย**: ออกแบบด้วย Streamlit เพื่อประสบการณ์การใช้งานที่ราบรื่น

## 🎯 วัตถุประสงค์
Amulet-AI ถูกพัฒนาขึ้นเพื่อ:
1. ช่วยนักสะสมและผู้สนใจพระเครื่องในการวิเคราะห์และจำแนกประเภทพระเครื่องได้อย่างรวดเร็ว
2. เพิ่มความมั่นใจในการซื้อขายและประเมินมูลค่าพระเครื่อง
3. สนับสนุนการอนุรักษ์และเผยแพร่วัฒนธรรมไทยผ่านเทคโนโลยี

## 🚀 วิธีการใช้งาน (ใช้ได้แค่สำหรับมีไฟล์คอม Local)
1. **เปิด Backend API**: รันคำสั่ง `uvicorn backend.api:app --reload --port 8000`
2. **เปิด Frontend**: รันคำสั่ง `streamlit run frontend/app_streamlit.py`
3. **เข้าใช้งาน**: เปิดเบราว์เซอร์ที่ `http://localhost:8501`

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

### � หลักการทำงานหลัก
Amulet-AI ใช้สถาปัตยกรรม Microservices ที่ประกอบด้วย 4 เทคโนโลยีหลัก:

#### 1) 🤖 TensorFlow
**เป้าหมายหลัก**: ใช้ในการฝึกโมเดล AI สำหรับการจำแนกพระเครื่อง โดยใช้ Transfer Learning

**ฟังก์ชันหลัก**:
- **Image Classification**: จำแนกประเภทของพระเครื่อง (รุ่น, พิมพ์) จากภาพที่อัปโหลด
- **Training + Fine-Tuning**: ใช้ dataset ที่มีอยู่ในการฝึกโมเดลจากโมเดลพรีเทรน (EfficientNetV2, ResNet50)
- **Inference**: ใช้ใน `model_loader.py` สำหรับทำการทำนายเมื่อมีการอัปโหลดภาพ
- **Mobile Optimization**: Export โมเดลเป็น TFLite สำหรับการ deploy บนอุปกรณ์มือถือ

#### 2) 🕷️ Scrapy
**เป้าหมายหลัก**: ใช้ในการเก็บข้อมูลราคาพระเครื่อง จากเว็บไซต์ตลาดพระเครื่อง

**ฟังก์ชันหลัก**:
- **Web Scraping**: ดึงข้อมูลราคา, ความนิยม, ปีที่ผลิตของพระเครื่องจากเว็บ
- **Data Collection**: เก็บข้อมูลในรูปแบบ CSV หรือ JSON เพื่อใช้ในการประเมินราคาใน `valuation.py`
- **Historical Data**: ข้อมูลราคาจะถูกนำมาจัดเก็บและใช้ในการคำนวณช่วงราคาประเมิน (p05, p50, p95)
- **Multi-Source Crawling**: สามารถตั้งค่า crawl spider เพื่อดึงข้อมูลจากหลายเว็บได้

#### 3) 🔍 FAISS (Facebook AI Similarity Search)
**เป้าหมายหลัก**: ใช้ในการค้นหาภาพที่คล้ายกัน (similarity search) สำหรับพระเครื่อง

**ฟังก์ชันหลัก**:
- **Similarity Search**: ใช้ embedding ของภาพพระเครื่องจากโมเดล AI เพื่อหาภาพที่มีลักษณะคล้ายกันจากฐานข้อมูล
- **Scalable Search**: รองรับการขยายขนาดเมื่อ dataset ขยายใหญ่ขึ้น
- **Vector Database**: สร้างฐานข้อมูล embedding โดยแปลงภาพทุกภาพเป็น vector embeddings
- **Fast Retrieval**: ค้นหาความคล้ายคลึงได้อย่างรวดเร็วแม้กับข้อมูลขนาดใหญ่

#### 4) 📈 Scikit-learn
**เป้าหมายหลัก**: ใช้ในการประเมินราคา (Valuation) ของพระเครื่อง

**ฟังก์ชันหลัก**:
- **Price Estimation**: ใช้ regression models (Linear Regression, Random Forest, Gradient Boosting) ในการคำนวณช่วงราคาประเมิน
- **Machine Learning Models**: ฝึก models บนข้อมูลราคาพระเครื่องในช่วงเวลาต่างๆ
- **Model Evaluation**: ใช้ metrics (MSE, R-squared) เพื่อตรวจสอบประสิทธิภาพของโมเดลประเมินราคา
- **Cross Validation**: ทดสอบความแม่นยำของโมเดลประเมินราคา

### 🔄 Data Flow Architecture
```
📱 User Upload Image
    ↓
🖼️ Image Preprocessing (Pillow)
    ↓
🤖 TensorFlow Model (Classification)
    ↓
🔍 FAISS (Similarity Search)
    ↓
📈 Scikit-learn (Price Estimation)
    ↓
🕷️ Scrapy Data (Market Prices)
    ↓
📊 Final Recommendation
```

## �💡 เทคโนโลยีที่ใช้
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI Model**: TensorFlow + TensorFlow Lite
- **Web Scraping**: Scrapy
- **Similarity Search**: FAISS
- **Price Modeling**: Scikit-learn
- **Image Processing**: Pillow, NumPy
- **API Communication**: Requests