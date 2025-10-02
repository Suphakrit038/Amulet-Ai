import streamlit as st

# Simple test page
st.title("🔮 Amulet-AI Test Page")
st.write("หากคุณเห็นข้อความนี้ แสดงว่า Streamlit ทำงานได้แล้ว!")

st.success("✅ ระบบพร้อมใช้งาน")

if st.button("ทดสอบปุ่ม"):
    st.balloons()
    st.write("🎉 ปุ่มทำงานได้!")

st.sidebar.write("Sidebar ทำงานด้วย")