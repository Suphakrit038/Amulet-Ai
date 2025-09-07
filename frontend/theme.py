"""
โมดูลธีมสำหรับ Amulet-AI

ประกอบด้วย CSS และ JS สำหรับการออกแบบ UI ในธีมไทย
เนื้อหาทั้งหมดได้รับการออกแบบให้โหลดผ่าน st.markdown(..., unsafe_allow_html=True)
"""

# CSS ธีมไทยสมัยใหม่
THAI_THEME_CSS = """
<style>
:root {
    --bg-1: #5b0f12;
    --bg-2: #3f0a0b;
    --gold: #ffd166;
    --card: #ffffff;
    --muted: #f1f5f9;
    --accent: #ffd166;
    --border: rgba(0,0,0,0.06);
    --success: #16a34a;
    --error: #dc2626;
    --warning: #d97706;
}

html, body {
    background: linear-gradient(180deg, var(--bg-1), var(--bg-2));
    font-family: 'Inter', system-ui, sans-serif;
}

.stApp {
    min-height: 100vh;
    background: transparent;
}

.stApp .block-container {
    background: rgba(255, 255, 255, 0.96);
    color: #111;
    border-radius: 12px;
    padding: 1.75rem;
    margin: 1rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    border: 1px solid var(--border);
}

.app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: linear-gradient(135deg, rgba(128, 0, 0, 0.9), rgba(84, 19, 28, 0.9));
    border: 1px solid rgba(255, 215, 0, 0.3);
    border-radius: 12px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    color: var(--gold);
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
    animation: thaiShimmer 4s infinite;
}

@keyframes thaiShimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

.header-text h1 {
    margin: 0.1rem 0;
    font-size: 2.5rem;
    background: linear-gradient(135deg, #ffd700, #cd7f32);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 2px 15px rgba(255, 215, 0, 0.3);
}

.header-text p {
    margin: 0;
    font-size: 1rem;
    color: #ffd700;
    opacity: 0.9;
}

.upload-section {
    background: linear-gradient(135deg, #fff 0%, #f8f9ff 100%);
    border: 2px dashed rgba(212,175,55,0.4);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    margin: 1rem 0;
}

.upload-section:hover {
    border-color: rgba(212,175,55,0.7);
    background: linear-gradient(135deg, #fdf9f2 0%, #f0f4ff 100%);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(212,175,55,0.15);
}

.upload-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.1), transparent);
    transition: left 0.5s ease;
}

.upload-section:hover::before {
    left: 100%;
}

.stButton > button {
    background: linear-gradient(135deg, #800000 0%, #5a0000 100%);
    color: #ffd700;
    border: 1px solid rgba(255, 215, 0, 0.5);
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(128, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #9a0000 0%, #700000 100%);
    border-color: #ffd700;
    box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 12px rgba(128, 0, 0, 0.3);
}

.stButton > button:disabled {
    background: #ccc;
    color: #666;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.validation-success {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border: 1px solid #34d399;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.validation-error {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 1px solid #f87171;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}

@media (max-width: 768px) {
    .app-header {
        flex-direction: column;
        text-align: center;
    }
    .header-text h1 {
        font-size: 2rem;
    }
    .upload-section {
        padding: 1rem;
    }
}
</style>
"""

# JS ขั้นต่ำสำหรับ drag-and-drop (น้ำหนักเบา ไม่มี setInterval)
DRAG_DROP_JS = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    const uploadSections = document.querySelectorAll('.upload-section');

    uploadSections.forEach(function(section) {
        const fileInput = section.querySelector('input[type="file"]');

        if (fileInput) {
            // Prevent default drag behaviors
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                section.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });

            // Highlight on drag
            ['dragenter', 'dragover'].forEach(eventName => {
                section.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                section.addEventListener(eventName, unhighlight, false);
            });

            // Handle drop
            section.addEventListener('drop', handleDrop, false);

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            function highlight(e) {
                section.classList.add('drag-over');
            }

            function unhighlight(e) {
                section.classList.remove('drag-over');
            }

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;

                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('image/')) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            }
        }
    });
});
</script>
"""

def get_theme_css():
    """ส่งกลับ CSS ของธีมไทยเป็นสตริง"""
    return THAI_THEME_CSS

def get_drag_drop_js():
    """ส่งกลับ JS สำหรับ drag-and-drop เป็นสตริง"""
    return DRAG_DROP_JS

def load_theme():
    """โหลดธีมทั้งหมด (CSS + JS) สำหรับใช้ใน st.markdown"""
    return THAI_THEME_CSS + DRAG_DROP_JS
