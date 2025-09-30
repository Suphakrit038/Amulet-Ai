#!/usr/bin/env python3
"""
Production-Ready Frontend for Amulet-AI
Frontend ที่พร้อมใช้งานจริงสำหรับการจำแนกพระเครื่อง
Fixed version using modular components
"""

import streamlit as st
import requests
import sys
import os
from PIL import Image
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components and utils
try:
    from frontend.components.image_display import ImageDisplayComponent
    from frontend.components.analysis_results import AnalysisResultsComponent
    from frontend.components.file_uploader import FileUploaderComponent
    from frontend.components.mode_selector import ModeSelectorComponent
    from frontend.utils.image_processor import ImagePreprocessor
    from frontend.utils.file_validator import FileValidator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Amulet-AI - AI-Powered Amulet Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def load_css():
    """Load CSS from style.css file and add JavaScript for interactivity"""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Add JavaScript for navbar scroll effect and dark mode
        js_code = """
        <script>
        // Navbar scroll effect
        window.addEventListener('scroll', function() {
            const navbar = document.getElementById('navbar');
            if (navbar) {
                if (window.scrollY > 50) {
                    navbar.classList.add('navbar-scrolled');
                } else {
                    navbar.classList.remove('navbar-scrolled');
                }
            }
        });
        
        // Dark mode toggle functionality
        document.addEventListener('DOMContentLoaded', function() {
            const darkModeToggle = document.getElementById('darkModeToggle');
            const currentTheme = localStorage.getItem('theme') || 'light';
            
            // Apply saved theme
            document.documentElement.setAttribute('data-theme', currentTheme);
            
            if (darkModeToggle) {
                // Update toggle icon
                darkModeToggle.innerHTML = currentTheme === 'dark' ? '☀️' : '🌙';
                
                darkModeToggle.addEventListener('click', function() {
                    const currentTheme = document.documentElement.getAttribute('data-theme');
                    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                    
                    document.documentElement.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    
                    // Update toggle icon with animation
                    this.style.transform = 'scale(0.8)';
                    setTimeout(() => {
                        this.innerHTML = newTheme === 'dark' ? '☀️' : '🌙';
                        this.style.transform = 'scale(1)';
                    }, 150);
                });
            }
        });
        
        // Smooth scroll for anchor links
        document.addEventListener('click', function(e) {
            if (e.target.tagName === 'A' && e.target.getAttribute('href').startsWith('#')) {
                e.preventDefault();
                const targetId = e.target.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
        
        // Add ripple effect to buttons
        document.addEventListener('click', function(e) {
            if (e.target.tagName === 'BUTTON') {
                const button = e.target;
                const ripple = document.createElement('span');
                const rect = button.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                ripple.classList.add('ripple-effect');
                
                button.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            }
        });
        </script>
        
        <style>
        """ + css_content + """
        
        /* Ripple Effect */
        .ripple-effect {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.6);
            transform: scale(0);
            animation: ripple 0.6s linear;
            pointer-events: none;
        }
        </style>
        """
        
        st.markdown(js_code, unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default styling.")

class AmuletFrontend:
    """Simple Frontend class for API communication"""
    
    def __init__(self):
        self.base_url = API_BASE_URL
    
    def check_api_health(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

def main():
    # Load CSS
    load_css()
    
    # Initialize components
    image_display = ImageDisplayComponent()
    analysis_results = AnalysisResultsComponent()
    file_uploader = FileUploaderComponent(MAX_FILE_SIZE)
    mode_selector = ModeSelectorComponent()
    frontend = AmuletFrontend()
    
    # Check API health
    api_healthy = frontend.check_api_health()
    
    # ==========================================================
    # Header Section - Simple and Clean
    # ==========================================================
    
    # Modern Navbar with transparency and blur effect
    st.markdown(f"""
    <nav id="navbar" style="
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(20px); 
        -webkit-backdrop-filter: blur(20px); border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 20px rgba(0, 0, 0, 0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1400px; margin: 0 auto;">
            <!-- Logo Section -->
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           width: 40px; height: 40px; border-radius: 12px; display: flex; 
                           align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(102, 126, 234, 0.3);">
                    <span style="color: white; font-size: 1.2rem; font-weight: 700;">🔮</span>
                </div>
                <div>
                    <h1 style="margin: 0; color: #1e293b; font-weight: 800; font-size: 1.4rem; 
                              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Amulet-AI</h1>
                    <p style="margin: 0; color: #64748b; font-size: 0.75rem; font-weight: 500;">AI-Powered Analysis</p>
                </div>
            </div>
            
            <!-- Navigation Links -->
            <div style="display: flex; align-items: center; gap: 2rem;">
                <a href="#docs" style="color: #64748b; text-decoration: none; font-weight: 500; 
                                     transition: color 0.2s ease; font-size: 0.9rem;">Documentation</a>
                <a href="#contact" style="color: #64748b; text-decoration: none; font-weight: 500;
                                        transition: color 0.2s ease; font-size: 0.9rem;">Contact</a>
                <a href="#github" style="color: #64748b; text-decoration: none; font-weight: 500;
                                       transition: color 0.2s ease; font-size: 0.9rem;">GitHub</a>
            </div>
            
            <!-- API Status -->
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; 
                           background: {'rgba(16, 185, 129, 0.1)' if api_healthy else 'rgba(239, 68, 68, 0.1)'}; 
                           padding: 0.4rem 0.8rem; border-radius: 20px; border: 1px solid {'rgba(16, 185, 129, 0.2)' if api_healthy else 'rgba(239, 68, 68, 0.2)'};">
                    <div style="width: 8px; height: 8px; border-radius: 50%; 
                               background: {'#10b981' if api_healthy else '#ef4444'}; 
                               box-shadow: 0 0 0 2px {'rgba(16, 185, 129, 0.2)' if api_healthy else 'rgba(239, 68, 68, 0.2)'};">
                    </div>
                    <span style="font-size: 0.75rem; font-weight: 600; 
                                color: {'#10b981' if api_healthy else '#ef4444'};">
                        {'API Online' if api_healthy else 'API Offline'}
                    </span>
                </div>
            </div>
        </div>
    </nav>

    
    # ==========================================================
    # Main Hero Section - Streamlit Native Components
    # ==========================================================
    
    # Welcome Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2rem;
                color: white; text-align: center;">
        <h2 style="font-size: 2rem; font-weight: 700; margin: 0 0 1rem 0;">ยินดีต้อนรับสู่ Amulet-AI</h2>
        <p style="font-size: 1.1rem; margin: 0; opacity: 0.9; line-height: 1.6;">
                    เทคโนโลยี <strong>Deep Learning</strong> และ <strong>Computer Vision</strong> ล้ำสมัย<br>
                    เพื่อการจำแนกพระเครื่องไทยด้วยความแม่นยำสูงสุด
                </p>
                
                <!-- Feature Highlights -->
                <div style="display: flex; gap: 2rem; margin: 2rem 0;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="background: rgba(255,255,255,0.2); padding: 0.4rem; border-radius: 8px;">
                            <span style="font-size: 1.2rem;">⚡</span>
                        </div>
                        <span style="font-weight: 600;">ความแม่นยำ 94.5%</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="background: rgba(255,255,255,0.2); padding: 0.4rem; border-radius: 8px;">
                            <span style="font-size: 1.2rem;">🔒</span>
                        </div>
                        <span style="font-weight: 600;">ปลอดภัย 100%</span>
                    </div>
                </div>
                
                <!-- CTA Buttons -->
                <div style="display: flex; gap: 1rem; margin-top: 2rem;">
                    <button style="background: rgba(255,255,255,0.9); color: #667eea; 
                                  border: none; padding: 1rem 2rem; border-radius: 50px;
                                  font-weight: 700; font-size: 1rem; cursor: pointer;
                                  transition: all 0.3s ease; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                        เริ่มวิเคราะห์เลย →
                    </button>
                    <button style="background: transparent; color: white; 
                                  border: 2px solid rgba(255,255,255,0.3); padding: 1rem 2rem; 
                                  border-radius: 50px; font-weight: 600; font-size: 1rem; 
                                  cursor: pointer; transition: all 0.3s ease;
                                  backdrop-filter: blur(10px);">
                        ดูตัวอย่าง
                    </button>
                </div>
            </div>
            
            <!-- Right Column: Visual/Mockup -->
            <div style="text-align: center; position: relative;">
                <!-- Floating Cards Animation -->
                <div style="position: relative; height: 400px;">
                    <!-- Main Device Mockup -->
                    <div style="background: rgba(255,255,255,0.95); border-radius: 20px; 
                               padding: 1.5rem; box-shadow: 0 20px 60px rgba(0,0,0,0.1);
                               backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.3);
                               transform: rotate(-5deg); margin: 2rem;">
                        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                                   height: 200px; border-radius: 12px; display: flex; 
                                   align-items: center; justify-content: center; 
                                   border: 2px dashed #cbd5e1;">
                            <div style="text-align: center; color: #64748b;">
                                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
                                <p style="margin: 0; font-weight: 600;">Upload Amulet Image</p>
                                <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem;">AI Analysis Ready</p>
                            </div>
                        </div>
                        
                        <!-- Mock Analysis Result -->
                        <div style="margin-top: 1rem; padding: 0.75rem; 
                                   background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                                   border-radius: 8px; color: white; text-align: left;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 600; font-size: 0.9rem;">พระสมเด็จ</span>
                                <span style="font-weight: 700; font-size: 0.9rem;">94.5%</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Floating Elements -->
                    <div style="position: absolute; top: 20px; right: 20px; 
                               background: rgba(255,255,255,0.9); padding: 0.75rem; 
                               border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                               animation: float 6s ease-in-out infinite;">
                        <span style="font-size: 1.5rem;">🤖</span>
                    </div>
                    
                    <div style="position: absolute; bottom: 20px; left: 20px; 
                               background: rgba(255,255,255,0.9); padding: 0.75rem; 
                               border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                               animation: float 8s ease-in-out infinite reverse;">
                        <span style="font-size: 1.5rem;">✨</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <style>
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(-5deg); }
        50% { transform: translateY(-10px) rotate(-3deg); }
    }
    
    @media (max-width: 768px) {
        .hero-grid { grid-template-columns: 1fr !important; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern Feature Cards with Glassmorphism
    st.markdown("### ✨ คุณสมบัติเด่นที่ทำให้ Amulet-AI พิเศษ")
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
                gap: 2rem; margin: 2rem 0;">
        
        <!-- Accuracy Card -->
        <div class="feature-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                                        backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem;
                                        border: 1px solid rgba(255, 255, 255, 0.2); 
                                        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
                                        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                                        position: relative; overflow: hidden;">
            
            <!-- Floating Icon -->
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           width: 60px; height: 60px; border-radius: 16px; display: flex;
                           align-items: center; justify-content: center; margin-right: 1rem;
                           box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-weight: 700; font-size: 1.25rem;">ความแม่นยำสูงสุด</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">AI Model ที่ผ่านการฝึกฝนแล้ว</p>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5rem; font-weight: 900; margin-bottom: 0.5rem;">94.5%</div>
            
            <p style="color: #64748b; margin: 0; line-height: 1.6;">
                โมเดล AI ที่ใช้เทคโนโลยี Deep Learning ล่าสุด ผ่านการฝึกฝนด้วยข้อมูลพระเครื่องไทยกว่า 50,000 ภาพ
            </p>
            
            <!-- Glowing effect on hover -->
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                       background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                       opacity: 0; transition: opacity 0.3s ease; border-radius: 20px;"></div>
        </div>
        
        <!-- Speed Card -->
        <div class="feature-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                                        backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem;
                                        border: 1px solid rgba(255, 255, 255, 0.2);
                                        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);
                                        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                                        position: relative; overflow: hidden;">
            
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                           width: 60px; height: 60px; border-radius: 16px; display: flex;
                           align-items: center; justify-content: center; margin-right: 1rem;
                           box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-weight: 700; font-size: 1.25rem;">ความเร็วสูง</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">ประมวลผลรวดเร็วทันใจ</p>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5rem; font-weight: 900; margin-bottom: 0.5rem;">< 2 วินาที</div>
            
            <p style="color: #64748b; margin: 0; line-height: 1.6;">
                ระบบ Cloud Computing ที่มีประสิทธิภาพสูง ให้ผลการวิเคราะห์ที่รวดเร็วและแม่นยำในเวลาไม่เกิน 2 วินาที
            </p>
        </div>
        
        <!-- Security Card -->
        <div class="feature-card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
                                        backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem;
                                        border: 1px solid rgba(255, 255, 255, 0.2);
                                        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
                                        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                                        position: relative; overflow: hidden;">
            
            <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
                <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                           width: 60px; height: 60px; border-radius: 16px; display: flex;
                           align-items: center; justify-content: center; margin-right: 1rem;
                           box-shadow: 0 8px 20px rgba(139, 92, 246, 0.3);">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 22C12 22 20 18 20 12V5L12 2L4 5V12C4 18 12 22 12 22Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-weight: 700; font-size: 1.25rem;">ความปลอดภัย</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">ไม่เก็บข้อมูลส่วนตัว</p>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2.5rem; font-weight: 900; margin-bottom: 0.5rem;">100%</div>
            
            <p style="color: #64748b; margin: 0; line-height: 1.6;">
                การรับประกันความปลอดภัยระดับสูงสุด ไม่มีการเก็บรูปภาพหรือข้อมูลส่วนตัวไว้ในระบบ
            </p>
        </div>
    </div>
    
    <style>
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card:hover > div:last-child {
        opacity: 1;
    }
    
    @media (max-width: 768px) {
        .feature-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==========================================================
    # Modern Upload Section with Step-by-Step Wizard
    # ==========================================================
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            เริ่มวิเคราะห์พระเครื่องของคุณ
        </h2>
        <p style="color: #64748b; font-size: 1.1rem; margin: 0;">
            ทำตามขั้นตอนง่าย ๆ เพื่อรับผลการวิเคราะห์ที่แม่นยำที่สุด
        </p>
    </div>
    
    <!-- Step Progress Indicator -->
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; 
                           background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                           display: flex; align-items: center; justify-content: center;
                           color: white; font-weight: 700; font-size: 0.9rem;">1</div>
                <span style="font-weight: 600; color: #10b981;">เลือกโหมด</span>
            </div>
            <div style="width: 40px; height: 2px; background: #e2e8f0;"></div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           display: flex; align-items: center; justify-content: center;
                           color: white; font-weight: 700; font-size: 0.9rem;">2</div>
                <span style="font-weight: 600; color: #667eea;">อัปโหลดรูป</span>
            </div>
            <div style="width: 40px; height: 2px; background: #e2e8f0;"></div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 32px; height: 32px; border-radius: 50%; 
                           background: #e2e8f0; display: flex; align-items: center; justify-content: center;
                           color: #94a3b8; font-weight: 700; font-size: 0.9rem;">3</div>
                <span style="font-weight: 600; color: #94a3b8;">ผลลัพธ์</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================
    # Modern File Upload Section 
    # ==========================================================
    
    # Mode Selection (แสดงโหมดที่เลือกไว้แล้ว)
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 16px; 
                padding: 1.5rem; margin: 2rem 0; text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h3 style="margin: 0; color: #10b981; font-weight: 700;">Dual Image Analysis Mode</h3>
        </div>
        <p style="margin: 0; color: #059669; font-weight: 500;">
            โหมดวิเคราะห์แบบครบถ้วน - อัปโหลดรูปทั้งด้านหน้าและด้านหลังเพื่อความแม่นยำสูงสุด
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips section
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; 
                padding: 1.25rem; margin: 1.5rem 0;">
        <h4 style="margin: 0 0 0.75rem 0; color: #3b82f6; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L13.09 8.26L22 9L13.09 9.74L12 16L10.91 9.74L2 9L10.91 8.26L12 2Z" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            เคล็ดลับสำหรับผลลัพธ์ที่ดีที่สุด
        </h4>
        <ul style="margin: 0; padding-left: 1.25rem; color: #374151; line-height: 1.6;">
            <li>ถ่ายในสภาพแสงเดียวกันและพื้นหลังเดียวกัน</li>
            <li>วางพระเครื่องในตำแหน่งเดียวกันสำหรับทั้งสองด้าน</li>
            <li>ตรวจสอบให้ทั้งสองรูปมีความชัดเจนเท่าเทียมกัน</li>
            <li>หากมีข้อความหรือตัวเลข ให้แน่ใจว่าเห็นได้ชัดในทั้งสองด้าน</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern Drag & Drop Upload Areas
    col1, col2 = st.columns(2)
    
    # Handle session state for analysis mode
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'dual'
    
    with col1:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #374151; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           width: 24px; height: 24px; border-radius: 6px; display: flex;
                           align-items: center; justify-content: center; color: white; font-size: 0.8rem; font-weight: 700;">1</div>
                รูปด้านหน้า
            </h3>
            <p style="margin: 0.25rem 0 0 2rem; color: #64748b; font-size: 0.9rem;">อัปโหลดรูปด้านหน้าของพระเครื่อง</p>
        </div>
        """, unsafe_allow_html=True)
        
        front_image = st.file_uploader(
            "เลือกรูปด้านหน้า",
            type=['png', 'jpg', 'jpeg'],
            key="front_image",
            help="รองรับไฟล์: PNG, JPG, JPEG",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h3 style="margin: 0; color: #374151; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           width: 24px; height: 24px; border-radius: 6px; display: flex;
                           align-items: center; justify-content: center; color: white; font-size: 0.8rem; font-weight: 700;">2</div>
                รูปด้านหลัง
            </h3>
            <p style="margin: 0.25rem 0 0 2rem; color: #64748b; font-size: 0.9rem;">อัปโหลดรูปด้านหลังของพระเครื่อง</p>
        </div>
        """, unsafe_allow_html=True)
        
        back_image = st.file_uploader(
            "เลือกรูปด้านหลัง",
            type=['png', 'jpg', 'jpeg'],
            key="back_image",
            help="รองรับไฟล์: PNG, JPG, JPEG",
            label_visibility="collapsed"
        )
    
    # Modern Image Preview Section
    if front_image or back_image:
        st.markdown("""
        <div style="margin: 2rem 0 1rem 0;">
            <h3 style="color: #374151; font-weight: 600; margin-bottom: 1rem;">🖼️ ตัวอย่างรูปที่อัปโหลด</h3>
        </div>
        """, unsafe_allow_html=True)
        
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            if front_image:
                st.markdown("**📷 รูปด้านหน้า**")
                st.image(front_image, caption="รูปด้านหน้า", use_column_width=True)
                st.success(f"✅ อัปโหลดสำเร็จ - {front_image.name}")
        
        with preview_col2:
            if back_image:
                st.markdown("**📷 รูปด้านหลัง")
                st.image(back_image, caption="รูปด้านหลัง", use_column_width=True)
                st.success(f"✅ อัปโหลดสำเร็จ - {back_image.name}")
    
    # ==========================================================
    # Modern Analysis Section
    # ==========================================================
    
    # Analysis button with modern styling
    can_analyze = front_image is not None and back_image is not None
    
    st.markdown("<div style='margin: 3rem 0 2rem 0;'></div>", unsafe_allow_html=True)
    
    if can_analyze:
        # Ready to analyze state
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 16px; 
                    padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="margin-bottom: 1rem;">
                <div style="display: inline-block; background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                           width: 64px; height: 64px; border-radius: 50%; display: flex;
                           align-items: center; justify-content: center; margin-bottom: 1rem;
                           box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #10b981; font-weight: 700; font-size: 1.5rem;">พร้อมวิเคราะห์แล้ว!</h3>
            <p style="margin: 0; color: #059669; font-weight: 500; font-size: 1.1rem;">
                ระบบตรวจสอบรูปทั้งสองด้านเรียบร้อยแล้ว คลิกปุ่มด้านล่างเพื่อเริ่มการวิเคราะห์
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        analyze_button = st.button(
            "🚀 เริ่มวิเคราะห์ด้วย AI", 
            type="primary", 
            use_container_width=True,
            help="คลิกเพื่อเริ่มการวิเคราะห์ด้วย AI"
        )
    else:
        # Waiting for files state
        missing_front = "รูปด้านหน้า" if not front_image else ""
        missing_back = "รูปด้านหลัง" if not back_image else ""
        missing_text = " และ ".join(filter(None, [missing_front, missing_back]))
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%);
                    border: 2px dashed rgba(59, 130, 246, 0.3); border-radius: 16px; 
                    padding: 2rem; margin: 2rem 0; text-align: center;">
            <div style="margin-bottom: 1rem;">
                <div style="display: inline-block; background: rgba(59, 130, 246, 0.1);
                           width: 64px; height: 64px; border-radius: 50%; display: flex;
                           align-items: center; justify-content: center; margin-bottom: 1rem;
                           border: 2px solid rgba(59, 130, 246, 0.2);">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M14 2V8H20" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M12 18V12" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M9 15L12 12L15 15" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            <h3 style="margin: 0 0 0.5rem 0; color: #3b82f6; font-weight: 700; font-size: 1.25rem;">กรุณาอัปโหลดรูปภาพ</h3>
            <p style="margin: 0; color: #6366f1; font-weight: 500;">
                ยังต้องการ{missing_text}เพื่อเริ่มการวิเคราะห์
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        analyze_button = False
    
    # Display results with modern loading animation
    if analyze_button:
        # Update progress indicator
        st.markdown("""
        <div style="display: flex; justify-content: center; margin: 2rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                               background: #10b981; display: flex; align-items: center; justify-content: center;
                               color: white; font-weight: 700; font-size: 0.9rem;">✓</div>
                    <span style="font-weight: 600; color: #10b981;">เลือกโหมด</span>
                </div>
                <div style="width: 40px; height: 2px; background: #10b981;"></div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                               background: #10b981; display: flex; align-items: center; justify-content: center;
                               color: white; font-weight: 700; font-size: 0.9rem;">✓</div>
                    <span style="font-weight: 600; color: #10b981;">อัปโหลดรูป</span>
                </div>
                <div style="width: 40px; height: 2px; background: #667eea;"></div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; 
                               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                               display: flex; align-items: center; justify-content: center;
                               color: white; font-weight: 700; font-size: 0.9rem; 
                               animation: pulse 2s infinite;">3</div>
                    <span style="font-weight: 600; color: #667eea;">ผลลัพธ์</span>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🎯 ผลการวิเคราะห์")
        
        with st.spinner("🤖 AI กำลังวิเคราะห์รูปภาพอย่างละเอียด... กรุณารอสักครู่"):
            import time
            time.sleep(3)  # Simulate processing time
            
            # Use analysis results component
            analysis_results.display_results({}, 'dual')
    
    # ==========================================================
    # Modern Footer Section
    # ==========================================================
    
    st.markdown("""
    <footer style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                  color: white; padding: 3rem 2rem 2rem; margin-top: 4rem; 
                  border-radius: 20px 20px 0 0;">
        <div style="max-width: 1200px; margin: 0 auto;">
            <!-- Main Footer Content -->
            <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
                
                <!-- Brand Section -->
                <div>
                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   width: 40px; height: 40px; border-radius: 12px; display: flex; 
                                   align-items: center; justify-content: center;">
                            <span style="color: white; font-size: 1.2rem; font-weight: 700;">🔮</span>
                        </div>
                        <h3 style="margin: 0; font-weight: 800; font-size: 1.5rem;">Amulet-AI</h3>
                    </div>
                    <p style="color: #cbd5e1; line-height: 1.6; margin-bottom: 1.5rem;">
                        ระบบวิเคราะห์พระเครื่องด้วยปัญญาประดิษฐ์ที่ทันสมัย 
                        เพื่อการจำแนกและตรวจสอบความแท้จริงอย่างแม่นยำ
                    </p>
                    
                    <!-- Social Links -->
                    <div style="display: flex; gap: 1rem;">
                        <a href="#" style="background: rgba(255,255,255,0.1); width: 40px; height: 40px; 
                                         border-radius: 10px; display: flex; align-items: center; 
                                         justify-content: center; text-decoration: none; 
                                         transition: all 0.3s ease;">
                            <span style="color: white; font-size: 1.2rem;">🌐</span>
                        </a>
                        <a href="#" style="background: rgba(255,255,255,0.1); width: 40px; height: 40px; 
                                         border-radius: 10px; display: flex; align-items: center; 
                                         justify-content: center; text-decoration: none; 
                                         transition: all 0.3s ease;">
                            <span style="color: white; font-size: 1.2rem;">📧</span>
                        </a>
                        <a href="#" style="background: rgba(255,255,255,0.1); width: 40px; height: 40px; 
                                         border-radius: 10px; display: flex; align-items: center; 
                                         justify-content: center; text-decoration: none; 
                                         transition: all 0.3s ease;">
                            <span style="color: white; font-size: 1.2rem;">📱</span>
                        </a>
                    </div>
                </div>
                
                <!-- Products -->
                <div>
                    <h4 style="color: white; font-weight: 600; margin-bottom: 1rem;">ผลิตภัณฑ์</h4>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                AI Analysis
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                Batch Processing
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                API Integration
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                Mobile App
                            </a>
                        </li>
                    </ul>
                </div>
                
                <!-- Resources -->
                <div>
                    <h4 style="color: white; font-weight: 600; margin-bottom: 1rem;">ทรัพยากร</h4>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                Documentation
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                API Guide
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                Tutorials
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                Support
                            </a>
                        </li>
                    </ul>
                </div>
                
                <!-- Contact -->
                <div>
                    <h4 style="color: white; font-weight: 600; margin-bottom: 1rem;">ติดต่อเรา</h4>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                📧 support@amulet-ai.com
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                📞 +66 (0) 2-XXX-XXXX
                            </a>
                        </li>
                        <li style="margin-bottom: 0.5rem;">
                            <a href="#" style="color: #cbd5e1; text-decoration: none; transition: color 0.3s ease;">
                                📍 Bangkok, Thailand
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
            
            <!-- Partners Section -->
            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 2rem; margin-bottom: 1rem;">
                <h4 style="color: white; font-weight: 600; text-align: center; margin-bottom: 1.5rem;">
                    🤝 Partners & Supporters
                </h4>
                <div style="display: flex; justify-content: center; align-items: center; gap: 3rem;">
                    <!-- DEPA -->
                    <div style="text-align: center;">
                        <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 12px; 
                                   width: 80px; height: 60px; display: flex; align-items: center; 
                                   justify-content: center; margin: 0 auto 0.5rem;">
                            <span style="font-weight: 700; color: #1e293b; font-size: 0.8rem;">DEPA</span>
                        </div>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.8rem;">
                            Digital Economy<br>Promotion Agency
                        </p>
                    </div>
                    
                    <!-- Thai-Austrian -->
                    <div style="text-align: center;">
                        <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 12px; 
                                   width: 80px; height: 60px; display: flex; align-items: center; 
                                   justify-content: center; margin: 0 auto 0.5rem;">
                            <span style="font-weight: 700; color: #1e293b; font-size: 0.7rem;">🇹🇭🇦🇹</span>
                        </div>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.8rem;">
                            Thai-Austrian<br>University
                        </p>
                    </div>
                    
                    <!-- AI Research -->
                    <div style="text-align: center;">
                        <div style="background: rgba(255,255,255,0.9); padding: 1rem; border-radius: 12px; 
                                   width: 80px; height: 60px; display: flex; align-items: center; 
                                   justify-content: center; margin: 0 auto 0.5rem;">
                            <span style="font-weight: 700; color: #1e293b; font-size: 0.8rem;">🧠</span>
                        </div>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.8rem;">
                            AI Research<br>Center
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Copyright -->
            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1.5rem; 
                       text-align: center; color: #94a3b8;">
                <p style="margin: 0; font-size: 0.9rem;">
                    © 2025 Amulet-AI. All rights reserved. | 
                    <a href="#" style="color: #cbd5e1; text-decoration: none;">Privacy Policy</a> | 
                    <a href="#" style="color: #cbd5e1; text-decoration: none;">Terms of Service</a>
                </p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                    Powered by Deep Learning & Computer Vision Technology
                </p>
            </div>
        </div>
    </footer>
    
    <style>
    footer a:hover {
        color: #667eea !important;
        transform: translateY(-1px);
    }
    
    @media (max-width: 768px) {
        footer > div > div:first-child {
            grid-template-columns: 1fr !important;
            gap: 2rem !important;
        }
        
        footer .partners-section > div {
            flex-direction: column !important;
            gap: 1.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()