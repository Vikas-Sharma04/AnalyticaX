import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="viewerBadge"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #05070a; color: #e2e8f0; }
        .main-title { font-size: 3.5rem; font-weight: 800; margin-bottom: 0; color: #e2e8f0; }
        .main-title span {
            background: linear-gradient(90deg, #00f2ff, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            text-shadow: 0px 0px 15px rgba(0, 242, 255, 0.2);
        }
        .idle-container {
            text-align: center;
            padding: 80px 20px;
            border: 1px dashed #1e293b;
            border-radius: 20px;
            margin: 40px 0;
            background: rgba(15, 18, 26, 0.5);
        }

        .idle-logo {
            font-size: 6rem;
            display: inline-block;
            animation: pulse 2s infinite ease-in-out;
            filter: drop-shadow(0 0 15px rgba(0, 242, 255, 0.4));
            margin-bottom: 20px;
        }

        .idle-title {
            color: #e2e8f0;
            letter-spacing: 4px;
            font-weight: 800;
            margin-top: 0;
        }

        .idle-subtitle {
            color: #64748b;
            font-family: 'Inter', sans-serif;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 0.85rem;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.1); opacity: 1; filter: drop-shadow(0 0 25px rgba(0, 242, 255, 0.7)); }
            100% { transform: scale(1); opacity: 0.8; }
        }
        .data-waves {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 10px;
            height: 80px; /* Slightly taller for better range */
            margin: 0 auto 30px;
        }

        .data-waves span {
            width: 14px;
            height: 60px; /* Base height */
            background: linear-gradient(to top, #0099ff, #00f2ff);
            border-radius: 6px;
            transform-origin: bottom; /* Keeps the bottom fixed so it scales upward */
            animation: wave 1.5s infinite ease-in-out;
            box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
        }

        /* Staggered delays for a fluid "S" curve movement */
        .data-waves span:nth-child(1) { animation-delay: 0.0s; }
        .data-waves span:nth-child(2) { animation-delay: 0.2s; }
        .data-waves span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes wave {
            0%, 100% { 
                transform: scaleY(0.4); 
                opacity: 0.5;
            }
            50% { 
                transform: scaleY(1.1); /* Stretches upward */
                opacity: 1;
                filter: brightness(1.2);
            }
        }
        .sub-text { color: #64748b; letter-spacing: 2px; font-size: 0.8rem; margin-top: -10px; margin-bottom: 30px; }
        section[data-testid="stSidebar"] { background-color: #080a0f; }
        .sidebar-header { color: #00f2ff; font-size: 0.75rem; font-weight: 800; letter-spacing: 2px; margin-bottom: 10px; text-transform: uppercase; }
        div[data-testid="stMetric"] { background: #0f121a; border: 1px solid #1e293b; padding: 20px; border-radius: 15px; }
        div[data-testid="stMetricValue"] > div { color: #00f2ff !important; font-weight: 700; font-size: 2.5rem !important; }
        [data-testid="stMetric"].outlier-card div[data-testid="stMetricValue"] > div { color: #ff007a !important; }
        .stTabs [aria-selected="true"] { color: #00f2ff !important; border-bottom: 2px solid #00f2ff !important; }
        .stDataFrame, .stTable { background: #0f121a; border-radius: 10px; }
        .stTextInput > div > div > input { background-color: #0f121a !important; border: 1px solid #1e293b !important; color: #00f2ff !important; border-radius: 10px !important; }
        .data-note { padding: 10px; border-radius: 5px; background: #1e293b; border-left: 5px solid #00f2ff; margin-bottom: 20px; font-size: 0.9rem; }
        </style>
        """, unsafe_allow_html=True)