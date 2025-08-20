import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Amulet-AI", page_icon="🔍", layout="centered")
st.title("Amulet-AI — วิเคราะห์พระเครื่องลึกลับ")

front = st.file_uploader("อัปโหลดภาพด้านหน้า", type=["jpg","jpeg","png"])
back  = st.file_uploader("อัปโหลดภาพด้านหลัง (ถ้ามี)", type=["jpg","jpeg","png"])

if st.button("วิเคราะห์ตอนนี้", disabled=not front):
    files = {"front": (front.name, front, front.type)}
    if back:
        files["back"] = (back.name, back, back.type)
    with st.spinner("กำลังประมวลผล..."):
        r = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    if r.ok:
        data = r.json()
        st.success("สำเร็จ 🎉")
        st.subheader(f"รุ่น/พิมพ์: {data['top1']['class_name']} ({data['top1']['confidence']:.1%})")
        st.write("ตัวเลือกถัดไป (Top-3):")
        st.table([{k:v for k,v in x.items() if k!="class_id"} for x in data['topk']])
        st.divider()
        st.subheader("ช่วงราคาประเมิน")
        st.metric("P05", f"{data['valuation']['p05']:,} บาท")
        st.metric("Median", f"{data['valuation']['p50']:,} บาท")
        st.metric("P95", f"{data['valuation']['p95']:,} บาท")
        st.divider()
        st.subheader("แนะนำช่องทางการขาย")
        for rec in data["recommendations"]:
            st.write(f"• {rec['market']} — เหตุผล: {rec['reason']}")
    else:
        st.error(f"เกิดข้อผิดพลาด: {r.text}")