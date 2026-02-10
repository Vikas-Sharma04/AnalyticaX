import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from styles import apply_custom_css
from engine import load_data, apply_nlp_filter

# --- Page Config ---
st.set_page_config(page_title="AnalyticaX — Next-Gen Statistical Intelligence", page_icon="📊", layout="wide")

# --- Session State ---
for key in ["filters", "apply_clicked", "main_df", "search_query", "current_filename"]:
    if key not in st.session_state:
        if key == "filters": st.session_state[key] = []
        elif key == "apply_clicked": st.session_state[key] = False
        elif key == "search_query": st.session_state[key] = ""
        else: st.session_state[key] = None

def handle_search():
    st.session_state.search_query = st.session_state.nlp_input_widget
    st.session_state.nlp_input_widget = ""

apply_custom_css()

# --- UI Header ---
head_l, head_r = st.columns([2.5, 1])
with head_l:
    st.markdown('<h1 class="main-title"><span style="-webkit-text-fill-color: initial; background: none;">📊</span> <span>AnalyticaX</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">SMART ANALYTIC ENGINE // NEURAL DATA INTELLIGENCE</p>', unsafe_allow_html=True)

with head_r:
    st.markdown("<br>", unsafe_allow_html=True)
    st.text_input(label="🔍 SMART SEARCH", placeholder="e.g., age >= 30 and salary at least 5000", key="nlp_input_widget", on_change=handle_search)
    if st.session_state.search_query:
        st.caption(f"Active Query: `{st.session_state.search_query}`")

# --- Sidebar ---
st.sidebar.markdown('<p class="sidebar-header">📁 Data Source</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Drop node here", type=["csv", "xlsx", "parquet", "json"], label_visibility="collapsed")

if uploaded_file:
    if st.session_state.current_filename != uploaded_file.name:
        file_content = uploaded_file.read()
        st.session_state.main_df = load_data(file_content, uploaded_file.name)
        st.session_state.current_filename = uploaded_file.name
        st.session_state.filters = []

if st.session_state.main_df is not None:
    df = st.session_state.main_df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Sidebar Tools
    st.sidebar.markdown('<p class="sidebar-header">🛠️ Smart Cleaner</p>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("NULL FIX", use_container_width=True):
        for col in numeric_cols: df[col] = df[col].fillna(df[col].median())
        for col in cat_cols: df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "N/A")
        st.session_state.main_df = df
        st.toast("Intelligence: Repaired.")
    
    if c2.button("CLEAN DUPES", use_container_width=True):
        st.session_state.main_df = df.drop_duplicates()
        st.toast("Intelligence: Purged.")

    st.sidebar.markdown('<p class="sidebar-header">✨ Data Filters</p>', unsafe_allow_html=True)
    col_add, col_clear = st.sidebar.columns(2)
    if col_add.button("➕ ADD LAYER", use_container_width=True):
        st.session_state.filters.append({"col": df.columns[0], "op": "=", "val": 0.0})
        st.rerun()

    if col_clear.button("🗑️ CLEAR FILTERS", use_container_width=True):
        st.session_state.filters = []
        st.session_state.apply_clicked = False
        st.session_state.search_query = ""
        st.rerun()

    # Manual Filters Loop
    for i, f in enumerate(st.session_state.filters):
        with st.sidebar.expander(f"LAYER {i+1}", expanded=True):
            old_col = f['col']
            f['col'] = st.selectbox("Field", df.columns, key=f"c_{i}")
            if f['col'] != old_col:
                f['val'] = 0.0 if np.issubdtype(df[f['col']].dtype, np.number) else []
                st.rerun()

            if np.issubdtype(df[f['col']].dtype, np.number):
                f['op'] = st.selectbox("Op", ["=", ">", "<", ">=", "<=", "Between"], key=f"o_{i}")
                curr = f.get('val', 0.0)
                if isinstance(curr, list) or isinstance(curr, tuple): curr = 0.0
                if f['op'] == "Between":
                    f['val'] = st.slider("Range", float(df[f['col']].min()), float(df[f['col']].max()), 
                                        value=(float(df[f['col']].min()), float(df[f['col']].max())), key=f"v_{i}")
                else:
                    f['val'] = st.number_input("Value", value=float(curr), key=f"v_{i}")
            else:
                f['op'] = "in"
                curr_list = f.get('val', [])
                if not isinstance(curr_list, list): curr_list = []
                f['val'] = st.multiselect("Select", df[f['col']].unique(), default=curr_list, key=f"v_{i}")
            
            if st.button(f"🗑️ Remove Layer {i+1}", key=f"d_{i}", use_container_width=True):
                st.session_state.filters.pop(i)
                st.rerun()

    if st.sidebar.button("⚡ APPLY FILTERS", use_container_width=True, type="primary"):
        st.session_state.apply_clicked = True

    if st.sidebar.button("🚨 RESET ALL", use_container_width=True):
        st.session_state.filters = []; st.session_state.apply_clicked = False; st.session_state.search_query = ""; st.rerun()

    # --- Filtering Logic ---
    filtered_df = df.copy()
    if st.session_state.search_query:
        filtered_df = apply_nlp_filter(filtered_df, st.session_state.search_query)
    if st.session_state.apply_clicked:
        for f in st.session_state.filters:
            try:
                if f['op'] == "=": filtered_df = filtered_df[filtered_df[f['col']] == f['val']]
                elif f['op'] == ">": filtered_df = filtered_df[filtered_df[f['col']] > f['val']]
                elif f['op'] == "<": filtered_df = filtered_df[filtered_df[f['col']] < f['val']]
                elif f['op'] == ">=": filtered_df = filtered_df[filtered_df[f['col']] >= f['val']]
                elif f['op'] == "<=": filtered_df = filtered_df[filtered_df[f['col']] <= f['val']]
                elif f['op'] == "Between": filtered_df = filtered_df[filtered_df[f['col']].between(f['val'][0], f['val'][1])]
                elif f['op'] == "in" and f['val']: filtered_df = filtered_df[filtered_df[f['col']].isin(f['val'])]
            except: pass

    # --- TABS ---
    tab_dash, tab_intel, tab_logic, tab_viz, tab_export = st.tabs(["📈 DASHBOARD", "🔬 SUMMARY", "⚙️ LOGIC", "🎨 VISUALIZER", "📤 EXPORT"])

    with tab_dash:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RECORDS", f"{filtered_df.shape[0]:,}")
        m2.metric("FEATURES", filtered_df.shape[1])
        m3.metric("NULL NODES", filtered_df.isnull().sum().sum())
        density = (1 - filtered_df.isnull().sum().sum()/filtered_df.size)*100 if filtered_df.size > 0 else 0
        m4.metric("DENSITY", f"{density:.1f}%")
        
        l_col, r_col = st.columns([1.5, 1])
        with l_col:
            st.markdown("#### 📡 CORRELATION MATRIX")
            if len(numeric_cols) > 1 and len(filtered_df) > 1:
                fig_corr = px.imshow(filtered_df[numeric_cols].corr(), color_continuous_scale='Viridis', text_auto=".2f")
                fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
                st.plotly_chart(fig_corr, use_container_width=True)
        with r_col:
            st.markdown("#### 💾 DATA PREVIEW")
            st.dataframe(filtered_df.head(50), height=400, use_container_width=True)

    with tab_intel:
        st.markdown("#### 🔬 STATISTICAL SUMMARY")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📋 METADATA OVERVIEW")
        info_df = pd.DataFrame({"Type": df.dtypes.astype(str), "Non-Null": df.count(), "Nulls": df.isnull().sum(), "Unique": df.nunique()})
        st.dataframe(info_df, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📉 OUTLIER DETECTION (IQR)")
        if numeric_cols:
            qc_col = st.selectbox("Analyze Field", numeric_cols)
            Q1, Q3 = df[qc_col].quantile(0.25), df[qc_col].quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = df[(df[qc_col] < low) | (df[qc_col] > high)]
            oc1, oc2 = st.columns([1, 2])
            with oc1:
                st.markdown('<div class="outlier-card">', unsafe_allow_html=True)
                st.metric("DETECTED OUTLIERS", len(outliers))
                st.markdown('</div>', unsafe_allow_html=True)
            with oc2: st.info(f"IQR Fence: {low:.2f} to {high:.2f}")
            if not outliers.empty:
                st.dataframe(outliers.head(10), use_container_width=True)
                st.plotly_chart(px.box(df, y=qc_col, points="all", template="plotly_dark", color_discrete_sequence=['#ff007a']), use_container_width=True)

    with tab_logic:
        st.markdown("#### ⚙️ GROUPED ANALYTICS")
        bypass_l = st.toggle("🔓 Bypass filters", key="bl")
        l_df = df if bypass_l else filtered_df
        lc1, lc2, lc3 = st.columns(3)
        g_cols = lc1.multiselect("Group By", cat_cols)
        v_cols = lc2.multiselect("Value Fields", numeric_cols)
        func = lc3.selectbox("Function", ["mean", "sum", "count", "max", "min"])
        if st.button("🚀 EXECUTE"):
            if g_cols and v_cols:
                res = l_df.groupby(g_cols)[v_cols].agg(func).reset_index()
                r1, r2 = st.columns([1, 1.5])
                r1.dataframe(res, use_container_width=True)
                with r2:
                    if len(g_cols) == 1 and len(v_cols) == 1:
                        st.plotly_chart(px.pie(res, names=g_cols[0], values=v_cols[0], hole=0.4, template="plotly_dark"), use_container_width=True)
                    else:
                        st.plotly_chart(px.bar(res, x=g_cols[0], y=v_cols[0], color=g_cols[-1] if len(g_cols)>1 else None, template="plotly_dark"), use_container_width=True)

    with tab_viz:
        st.markdown("#### 🏗️ CHART BUILDER")
        bypass_v = st.toggle("🔓 Bypass filters for chart", key="bv")
        v_df = df if bypass_v else filtered_df
        v1, v2, v3, v4 = st.columns(4)
        ctype = v1.selectbox("Type", ["Bar", "Line", "Scatter", "Histogram", "Box Plot", "Pie"])
        x = v2.selectbox("X-Axis", df.columns)
        y = v3.selectbox("Y-Axis", [None] + list(df.columns))
        c = v4.selectbox("Legend", [None] + list(df.columns))
        if st.button("🎨 RENDER"):
            try:
                if ctype == "Bar": fig = px.bar(v_df, x=x, y=y, color=c)
                elif ctype == "Line": fig = px.line(v_df, x=x, y=y, color=c)
                elif ctype == "Scatter": fig = px.scatter(v_df, x=x, y=y, color=c)
                elif ctype == "Histogram": fig = px.histogram(v_df, x=x, color=c)
                elif ctype == "Box Plot": fig = px.box(v_df, x=x, y=y, color=c)
                elif ctype == "Pie": fig = px.pie(v_df, names=x, values=y)
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Viz Error: {e}")

    with tab_export:
        st.markdown("#### 📤 EXPORT HUB")
        
        # Create columns for a cleaner layout
        ex1, ex2 = st.columns(2)
        
        with ex1:
            # --- CSV Export ---
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 DOWNLOAD CSV",
                data=csv_data,
                file_name="AnalyticaX_Report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # --- JSON Export ---
            json_data = filtered_df.to_json(orient="records", indent=4).encode('utf-8')
            st.download_button(
                label="🌐 DOWNLOAD JSON",
                data=json_data,
                file_name="AnalyticaX_Report.json",
                mime="application/json",
                use_container_width=True
            )

        with ex2:
            # --- Excel Export ---
            buffer_xlsx = io.BytesIO()
            with pd.ExcelWriter(buffer_xlsx, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='AnalyticaX_Export')
            st.download_button(
                label="📗 DOWNLOAD EXCEL",
                data=buffer_xlsx.getvalue(),
                file_name="AnalyticaX_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # --- Parquet Export ---
            # Note: Parquet is a binary format, perfect for large datasets
            buffer_parquet = io.BytesIO()
            filtered_df.to_parquet(buffer_parquet, index=False)
            st.download_button(
                label="📦 DOWNLOAD PARQUET",
                data=buffer_parquet.getvalue(),
                file_name="AnalyticaX_Report.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )

else:
    st.markdown("""
        <div class="idle-container">
            <div class="idle-logo data-waves">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <h2 class="idle-title">SYSTEM IDLE</h2>
            <p class="idle-subtitle">Upload a dataset to initialize AnalyticaX logic.</p>
        </div>
    """, unsafe_allow_html=True)