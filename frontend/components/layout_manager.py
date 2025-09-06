"""
Responsive Layout Manager for Amulet-AI Application
Handles responsive design and layout management
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple
import logging

class ResponsiveLayout:
    """Manages responsive layouts and screen adaptations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.breakpoints = {
            'mobile': 480,
            'tablet': 768,
            'desktop': 1024,
            'large': 1200
        }
        self._inject_responsive_css()
    
    def _inject_responsive_css(self):
        """Inject responsive CSS and JavaScript"""
        responsive_css = """
        <script>
        // Screen size detection
        function getScreenSize() {
            const width = window.innerWidth;
            if (width <= 480) return 'mobile';
            if (width <= 768) return 'tablet'; 
            if (width <= 1024) return 'desktop';
            return 'large';
        }
        
        // Update layout based on screen size
        function updateLayout() {
            const screenSize = getScreenSize();
            document.body.setAttribute('data-screen', screenSize);
            
            // Dispatch custom event for layout changes
            window.dispatchEvent(new CustomEvent('screenSizeChanged', {
                detail: { screenSize }
            }));
        }
        
        // Initialize and listen for resize
        updateLayout();
        window.addEventListener('resize', updateLayout);
        
        // Mobile-specific optimizations
        if (getScreenSize() === 'mobile') {
            // Disable zoom on mobile
            const viewport = document.querySelector('meta[name=viewport]');
            if (viewport) {
                viewport.setAttribute('content', 
                    'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no');
            }
        }
        </script>
        
        <style>
        /* Responsive utilities */
        .hidden-mobile { display: block; }
        .hidden-tablet { display: block; }
        .hidden-desktop { display: block; }
        .show-mobile { display: none; }
        .show-tablet { display: none; }
        .show-desktop { display: none; }
        
        @media (max-width: 480px) {
            .hidden-mobile { display: none !important; }
            .show-mobile { display: block !important; }
            
            /* Mobile-specific styles */
            .stSelectbox > div > div { font-size: 14px; }
            .stButton > button { 
                width: 100% !important;
                margin: 5px 0;
            }
            .block-container { 
                padding: 1rem !important;
                max-width: 100% !important;
            }
        }
        
        @media (min-width: 481px) and (max-width: 768px) {
            .hidden-tablet { display: none !important; }
            .show-tablet { display: block !important; }
            
            /* Tablet-specific styles */
            .block-container { 
                padding: 1.5rem !important;
                max-width: 100% !important;
            }
        }
        
        @media (min-width: 1024px) {
            .hidden-desktop { display: none !important; }
            .show-desktop { display: block !important; }
        }
        
        /* Responsive grid system */
        .responsive-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: 1fr;
        }
        
        @media (min-width: 768px) {
            .responsive-grid.cols-2 { grid-template-columns: 1fr 1fr; }
            .responsive-grid.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
            .responsive-grid.cols-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
        }
        
        @media (min-width: 1024px) {
            .responsive-grid.cols-lg-3 { grid-template-columns: 1fr 1fr 1fr; }
            .responsive-grid.cols-lg-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
            .responsive-grid.cols-lg-6 { grid-template-columns: repeat(6, 1fr); }
        }
        
        /* Touch-friendly styles */
        @media (hover: none) and (pointer: coarse) {
            .btn, button { 
                min-height: 44px !important;
                padding: 12px 16px !important;
            }
            .upload-area {
                padding: 2rem !important;
                border-width: 3px !important;
            }
        }
        </style>
        """
        
        st.markdown(responsive_css, unsafe_allow_html=True)
    
    def get_columns_config(self, total_columns: int = 2, 
                          mobile_stack: bool = True) -> List[float]:
        """Get responsive column configuration"""
        
        # JavaScript to detect screen size in Streamlit
        screen_size_js = """
        <script>
        const screenWidth = window.innerWidth;
        window.parent.postMessage({
            type: 'screen_width',
            width: screenWidth
        }, '*');
        </script>
        """
        
        st.markdown(screen_size_js, unsafe_allow_html=True)
        
        # Default desktop configuration
        if total_columns == 2:
            return [1, 1]
        elif total_columns == 3:
            return [1, 1, 1]
        elif total_columns == 4:
            return [1, 1, 1, 1]
        else:
            return [1] * total_columns
    
    def create_responsive_columns(self, 
                                 desktop_config: List[float],
                                 tablet_config: Optional[List[float]] = None,
                                 mobile_stack: bool = True) -> List:
        """Create responsive columns that adapt to screen size"""
        
        # For now, use desktop config (Streamlit limitation)
        # In future versions, could use custom components for true responsiveness
        columns = st.columns(desktop_config)
        
        return columns
    
    def render_mobile_navigation(self, pages: List[Dict[str, str]]):
        """Render mobile-friendly navigation"""
        
        nav_html = """
        <div class="mobile-nav show-mobile">
            <div class="nav-toggle" onclick="toggleMobileNav()">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <div class="nav-menu" id="mobile-nav-menu">
        """
        
        for page in pages:
            nav_html += f"""
                <a href="{page['url']}" class="nav-item">
                    {page.get('icon', '')} {page['title']}
                </a>
            """
        
        nav_html += """
            </div>
        </div>
        
        <style>
        .mobile-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: white;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1rem;
        }
        
        .nav-toggle {
            cursor: pointer;
            width: 30px;
            height: 25px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .nav-toggle span {
            height: 3px;
            background: #333;
            transition: 0.3s;
        }
        
        .nav-menu {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .nav-menu.active {
            display: block;
        }
        
        .nav-item {
            display: block;
            padding: 1rem;
            text-decoration: none;
            color: #333;
            border-bottom: 1px solid #eee;
        }
        
        .nav-item:hover {
            background: #f5f5f5;
        }
        </style>
        
        <script>
        function toggleMobileNav() {
            const menu = document.getElementById('mobile-nav-menu');
            menu.classList.toggle('active');
        }
        </script>
        """
        
        st.markdown(nav_html, unsafe_allow_html=True)
    
    def render_responsive_image_grid(self, 
                                   images: List[Dict],
                                   columns_desktop: int = 3,
                                   columns_tablet: int = 2,
                                   columns_mobile: int = 1):
        """Render responsive image grid"""
        
        grid_html = f"""
        <div class="responsive-image-grid">
        """
        
        for image in images:
            grid_html += f"""
            <div class="image-grid-item">
                <img src="{image['src']}" alt="{image.get('alt', '')}" 
                     class="grid-image" loading="lazy">
                <div class="image-overlay">
                    <h4>{image.get('title', '')}</h4>
                    <p>{image.get('description', '')}</p>
                </div>
            </div>
            """
        
        grid_html += """
        </div>
        
        <style>
        .responsive-image-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: 1fr;
        }
        
        @media (min-width: 481px) {
            .responsive-image-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (min-width: 769px) {
            .responsive-image-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        .image-grid-item {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .image-grid-item:hover {
            transform: translateY(-5px);
        }
        
        .grid-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .image-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            color: white;
            padding: 1rem;
            transform: translateY(100%);
            transition: transform 0.3s ease;
        }
        
        .image-grid-item:hover .image-overlay {
            transform: translateY(0);
        }
        
        .image-overlay h4 {
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }
        
        .image-overlay p {
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        </style>
        """
        
        st.markdown(grid_html, unsafe_allow_html=True)
    
    # Removed create_adaptive_sidebar: all sidebar logic deleted
    
    def optimize_for_mobile(self):
        """Apply mobile-specific optimizations"""
        
        mobile_optimizations = """
        <style>
        /* Mobile optimizations */
        @media (max-width: 480px) {
            /* Increase touch targets */
            .stButton > button {
                min-height: 44px !important;
                font-size: 16px !important; /* Prevent zoom on iOS */
            }
            
            .stSelectbox > div > div {
                min-height: 44px !important;
                font-size: 16px !important;
            }
            
            .stTextInput > div > div > input {
                font-size: 16px !important;
            }
            
            /* Improve scrolling */
            .main > div {
                -webkit-overflow-scrolling: touch;
            }
            
            /* Optimize images */
            .stImage > img {
                max-width: 100% !important;
                height: auto !important;
            }
            
            /* Better spacing */
            .stMarkdown {
                margin-bottom: 0.5rem !important;
            }
            
            /* Hide unnecessary elements */
            .stDeployButton {
                display: none !important;
            }
            
            /* Improve form usability */
            .stForm {
                padding: 1rem !important;
            }
        }
        
        /* Tablet optimizations */
        @media (min-width: 481px) and (max-width: 768px) {
            .block-container {
                padding: 2rem 1rem !important;
            }
            
            .stButton > button {
                min-height: 40px !important;
            }
        }
        </style>
        
        <script>
        // Prevent double-tap zoom on iOS
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function (event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
        
        // Optimize viewport for mobile
        if (/Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
            const viewport = document.querySelector('meta[name=viewport]');
            if (viewport) {
                viewport.setAttribute('content', 
                    'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover');
            }
        }
        </script>
        """
        
        st.markdown(mobile_optimizations, unsafe_allow_html=True)


class LayoutManager:
    """Main layout manager combining all responsive features"""
    
    def __init__(self):
        self.responsive = ResponsiveLayout()
        self.logger = logging.getLogger(__name__)
    
    def setup_page_config(self, 
                         page_title: str = "Amulet-AI",
                         page_icon: str = "🔮",
                         layout: str = "wide"):
        """Setup Streamlit page configuration (sidebar removed)"""
        st.set_page_config(
            page_title=page_title,
            page_icon=page_icon,
            layout=layout,
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"# {page_title}\nระบบวิเคราะห์พระเครื่องด้วย AI"
            }
        )
        # Apply mobile optimizations
        self.responsive.optimize_for_mobile()
    
    def create_main_layout(self) -> Dict:
        """Create main application layout (sidebar removed)"""
        layout_elements = {}
        # Main content area only
        layout_elements['main_container'] = st.container()
        with layout_elements['main_container']:
            layout_elements['header'] = st.container()
            layout_elements['content'] = st.container()
            layout_elements['footer'] = st.container()
        return layout_elements
    
    def create_comparison_layout(self) -> Tuple:
        """Create layout optimized for image comparison"""
        
        # Mobile: stack vertically
        # Tablet/Desktop: side by side
        col1, col2 = self.responsive.create_responsive_columns([1, 1])
        
        return col1, col2
    
    def create_gallery_layout(self, items_count: int) -> List:
        """Create responsive gallery layout"""
        
        if items_count <= 2:
            columns = self.responsive.create_responsive_columns([1, 1])
        elif items_count <= 4:
            columns = self.responsive.create_responsive_columns([1, 1, 1, 1])
        else:
            # Create multiple rows
            columns = []
            items_per_row = 3
            for i in range(0, items_count, items_per_row):
                row_items = min(items_per_row, items_count - i)
                row_columns = self.responsive.create_responsive_columns([1] * row_items)
                columns.extend(row_columns)
        
        return columns[:items_count]
