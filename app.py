import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import sys
import ast
from styles import apply_custom_css
from engine import load_data, apply_nlp_filter, universal_neural_parser

# --- Page Config ---
st.set_page_config(page_title="AnalyticaX — Next-Gen Statistical Intelligence", page_icon="📊", layout="wide")

# --- Session State Initialization ---
state_keys = {
    "filters": [],
    "viz_gallery": [],
    "apply_clicked": False,
    "search_query": "",
    "main_df": None,
    "current_filename": None,
    "history": [],
    "history_ptr": -1,
    "needs_rerun": False  # Logic to handle instant console updates
}

for key, default in state_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default

def save_to_history(df):
    """Captures a snapshot of the dataframe for the Undo/Redo timeline."""
    if st.session_state.history_ptr < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_ptr + 1]
    
    st.session_state.history.append(df.copy())
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
    
    st.session_state.history_ptr = len(st.session_state.history) - 1

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

# --- Sidebar: Data Source ---
st.sidebar.markdown('<p class="sidebar-header">📁 Data Source</p>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Drop node here", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is None and st.session_state.current_filename is not None:
    for key in state_keys:
        st.session_state[key] = state_keys[key]
    st.rerun()

if uploaded_file:
    if st.session_state.current_filename != uploaded_file.name:
        file_content = uploaded_file.read()
        raw_df = load_data(file_content, uploaded_file.name)
        
        for col in raw_df.columns:
            if raw_df[col].dtype == 'object':
                raw_df[col] = raw_df[col].astype(str).replace('nan', np.nan)
        
        st.session_state.main_df = raw_df
        st.session_state.current_filename = uploaded_file.name
        st.session_state.filters = []
        st.session_state.viz_gallery = []
        st.session_state.history = []
        save_to_history(raw_df)
        st.rerun()

if st.session_state.main_df is not None:
    df = st.session_state.main_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Sidebar: Intelligence Timeline ---
    st.sidebar.markdown('<p class="sidebar-header">⏳ Intelligence Timeline</p>', unsafe_allow_html=True)
    h_col1, h_col2 = st.sidebar.columns(2)
    
    if h_col1.button("⬅️ UNDO", use_container_width=True, disabled=st.session_state.history_ptr <= 0):
        st.session_state.history_ptr -= 1
        st.session_state.main_df = st.session_state.history[st.session_state.history_ptr].copy()
        st.rerun()
        
    if h_col2.button("REDO ➡️", use_container_width=True, disabled=st.session_state.history_ptr >= len(st.session_state.history) - 1):
        st.session_state.history_ptr += 1
        st.session_state.main_df = st.session_state.history[st.session_state.history_ptr].copy()
        st.rerun()
    
    st.sidebar.caption(f"Step {st.session_state.history_ptr + 1} of {len(st.session_state.history)}")

    # --- Sidebar: Smart Cleaner ---
    st.sidebar.markdown('<p class="sidebar-header">🛠️ Smart Cleaner</p>', unsafe_allow_html=True)
    
    with st.sidebar.expander("🩹 NULL REPAIR OPTIONS", expanded=False):
        num_fix = st.selectbox("Numeric Strategy", ["Zero", "Median", "Mean", "FFill"])
        cat_fix = st.selectbox("Categorical Strategy", ["N/A", "Mode", "FFill"])
        
        if st.button("EXECUTE REPAIR", use_container_width=True):
            new_df = df.copy()
            for col in numeric_cols:
                if num_fix == "Zero": new_df[col] = new_df[col].fillna(0)
                elif num_fix == "Median": new_df[col] = new_df[col].fillna(new_df[col].median())
                elif num_fix == "Mean": new_df[col] = new_df[col].fillna(new_df[col].mean())
                elif num_fix == "FFill": new_df[col] = new_df[col].ffill().bfill()
            for col in cat_cols:
                if cat_fix == "N/A": new_df[col] = new_df[col].fillna("N/A")
                elif cat_fix == "Mode": 
                    m = new_df[col].mode()
                    new_df[col] = new_df[col].fillna(m[0] if not m.empty else "N/A")
                elif cat_fix == "FFill": new_df[col] = new_df[col].ffill().bfill()
            st.session_state.main_df = new_df
            save_to_history(new_df)
            st.toast("Intelligence: Repaired")
            st.rerun()

    with st.sidebar.expander("✂️ DROP FEATURES", expanded=False):
        to_drop = st.multiselect("Purge Columns", df.columns.tolist())
        if st.button("DROP SELECTED", use_container_width=True):
            if to_drop:
                new_df = df.drop(columns=to_drop)
                st.session_state.main_df = new_df
                save_to_history(new_df)
                st.toast("Nodes Purged.")
                st.rerun()

    # --- Sidebar: Type Transformer ---
    st.sidebar.markdown('<p class="sidebar-header">🧬 Type Transformer</p>', unsafe_allow_html=True)
    with st.sidebar.expander("🛠️ CONVERT DATA TYPES", expanded=False):
        target_cols = st.multiselect("Select Columns", df.columns, key="trans_cols")
        target_type = st.selectbox("Convert To", ["Numeric (Smart Clean)", "String (Object)", "Datetime"])
        
        if st.button("EXECUTE TRANSFORMATION", use_container_width=True):
            if not target_cols:
                st.warning("Please select at least one column.")
            else:
                try:
                    new_df = df.copy()
                    for col in target_cols:
                        if target_type == "Numeric (Smart Clean)":
                            new_df[col] = new_df[col].apply(universal_neural_parser)
                        elif target_type == "String (Object)":
                            new_df[col] = new_df[col].astype(str)
                        elif target_type == "Datetime":
                            new_df[col] = pd.to_datetime(new_df[col], errors='coerce')

                    st.session_state.main_df = new_df
                    save_to_history(new_df)
                    st.toast(f"Intelligence: {len(target_cols)} nodes re-typed!")
                    st.rerun() 
                except Exception as e:
                    st.error(f"Transformation Error: {e}")
    
    if st.sidebar.button("CLEAN DUPES", use_container_width=True):
        new_df = df.drop_duplicates()
        st.session_state.main_df = new_df
        save_to_history(new_df)
        st.rerun()

    # --- Data Filters UI ---
    st.sidebar.markdown('<p class="sidebar-header">✨ Data Filters</p>', unsafe_allow_html=True)
    col_add, col_clear = st.sidebar.columns(2)
    if col_add.button("➕ ADD LAYER", use_container_width=True):
        st.session_state.filters.append({"col": df.columns[0], "op": "=", "val": 0.0})
        st.rerun()
    if col_clear.button("🗑️ CLEAR FILTERS", use_container_width=True):
        st.session_state.filters = []; st.session_state.apply_clicked = False; st.session_state.search_query = ""; st.rerun()

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
                if isinstance(curr, (list, tuple)): curr = 0.0
                if f['op'] == "Between":
                    f['val'] = st.slider("Range", float(df[f['col']].min()), float(df[f['col']].max()), value=(float(df[f['col']].min()), float(df[f['col']].max())), key=f"v_{i}")
                else:
                    f['val'] = st.number_input("Value", value=float(curr), key=f"v_{i}")
            else:
                f['op'] = "in"; curr_list = f.get('val', [])
                if not isinstance(curr_list, list): curr_list = []
                f['val'] = st.multiselect("Select", df[f['col']].unique(), default=curr_list, key=f"v_{i}")
            if st.button(f"🗑️ Remove Layer {i+1}", key=f"d_{i}", use_container_width=True):
                st.session_state.filters.pop(i); st.rerun()

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

    display_df = filtered_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype == object:
            display_df[col] = display_df[col].astype(str).replace('nan', np.nan)

    # --- Main Tabs ---
    tab_dash, tab_intel, tab_logic, tab_viz, tab_console, tab_export = st.tabs(["📈 DASHBOARD", "🔬 SUMMARY", "⚙️ LOGIC", "🎨 VISUALIZER", "💻 NEURAL CONSOLE", "📤 EXPORT"])

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
            current_num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(current_num_cols) > 1 and len(filtered_df) > 1:
                fig_corr = px.imshow(filtered_df[current_num_cols].corr(), color_continuous_scale='Viridis', text_auto=".2f")
                fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#e2e8f0")
                st.plotly_chart(fig_corr, use_container_width=True)
        with r_col:
            st.markdown("#### 💾 DATA PREVIEW")
            st.dataframe(display_df, height=400, use_container_width=True)

    with tab_intel:
        st.markdown("#### 🔬 STATISTICAL SUMMARY")
        st.dataframe(filtered_df.describe().T.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📋 METADATA OVERVIEW")
        info_df = pd.DataFrame({"Type": filtered_df.dtypes.astype(str), "Non-Null": filtered_df.count(), "Nulls": filtered_df.isnull().sum(), "Unique": filtered_df.nunique()})
        st.dataframe(info_df, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📉 OUTLIER DETECTION (IQR)")
        if numeric_cols:
            qc_col = st.selectbox("Analyze Field", numeric_cols)
            Q1, Q3 = df[qc_col].quantile(0.25), df[qc_col].quantile(0.75)
            IQR = Q3 - Q1; low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
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
        v_cols = lc2.multiselect("Value Fields", current_num_cols)
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
        st.markdown("#### 🏗️ NEURAL VIZ GALLERY")
        with st.expander("➕ CONFIGURE NEW GRAPH", expanded=True):
            v_df = df if st.toggle("🔓 Bypass filters for chart", key="bv") else filtered_df
            v1, v2, v3, v4 = st.columns(4)
            ctype = v1.selectbox("Type", ["Bar", "Line", "Scatter", "Histogram", "Box Plot", "Pie"])
            x = v2.selectbox("X-Axis", df.columns); y = v3.selectbox("Y-Axis", [None] + list(df.columns)); c = v4.selectbox("Legend", [None] + list(df.columns))
            if st.button("🎨 RENDER & ADD TO GALLERY", use_container_width=True):
                try:
                    if ctype == "Bar": fig = px.bar(v_df, x=x, y=y, color=c)
                    elif ctype == "Line": fig = px.line(v_df, x=x, y=y, color=c)
                    elif ctype == "Scatter": fig = px.scatter(v_df, x=x, y=y, color=c)
                    elif ctype == "Histogram": fig = px.histogram(v_df, x=x, color=c)
                    elif ctype == "Box Plot": fig = px.box(v_df, x=x, y=y, color=c)
                    elif ctype == "Pie": fig = px.pie(v_df, names=x, values=y)
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.session_state.viz_gallery.append({"fig": fig, "title": f"{ctype}: {x} vs {y}"})
                except Exception as e: st.error(f"Viz Error: {e}")

        if st.session_state.viz_gallery:
            st.markdown("---")
            if st.button("🗑️ PURGE ALL GRAPHS", use_container_width=True):
                st.session_state.viz_gallery = []; st.rerun()
            for i in range(0, len(st.session_state.viz_gallery), 2):
                row_cols = st.columns(2)
                for j in range(2):
                    if i + j < len(st.session_state.viz_gallery):
                        with row_cols[j]:
                            g_item = st.session_state.viz_gallery[i+j]
                            st.caption(f"NODE {i+j+1} // {g_item['title']}")
                            st.plotly_chart(g_item['fig'], use_container_width=True)
                            if st.button(f"Remove Node {i+j+1}", key=f"del_{i+j}"):
                                st.session_state.viz_gallery.pop(i+j); st.rerun()

    with tab_console:
        # 1. Header with Current File & Features
        current_file = st.session_state.get('current_filename', 'No File Active')
        st.markdown(f"#### 💻 NEURAL CONSOLE <span style='font-size:14px; color:#888;'>// ACTIVE: df = {current_file}</span>", unsafe_allow_html=True)
        
        # Display all features (columns) available in the dataframe
        st.caption(f"**Available Features:** {', '.join(df.columns.tolist())}")

        # Helper to prevent the state modification error
        def load_snippet(snippet):
            st.session_state.neural_code_area = snippet

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)

        # Snippets (kept generic as per previous request)
        if s_col1.button("🏷️ Rename Column", use_container_width=True):
            load_snippet(
                "# Rename specific column (replace 'old' and 'new')\n"
                "df.columns = df.columns.str.strip()\n"
                "df = df.rename(columns={'old_col': 'new_col'})\n"
                "df.head()"
            )

        # 2. FEATURE ENGINEERING (Generic)
        if s_col2.button("🧬 New Feature", use_container_width=True):
            load_snippet("# Create 'new_col' based on 'col_a' and 'col_b'\ndf['new_col'] = df['col_a'] + df['col_b']\ndf.head()")

        # 3. TARGETED OUTLIER REMOVAL (Drops rows)
        if s_col3.button("✂️ Remove Outliers", use_container_width=True):
            load_snippet(
                "# 1. Define target column and thresholds\n"
                "target_col = 'column_name_here'\n"
                "lower_p = 0.05  # 5th percentile\n"
                "upper_p = 0.95  # 95th percentile\n\n"
                "# 2. Calculate thresholds\n"
                "q_low = df[target_col].quantile(lower_p)\n"
                "q_high = df[target_col].quantile(upper_p)\n\n"
                "# 3. Filter the dataframe (Keep rows within range)\n"
                "before = len(df)\n"
                "df = df[(df[target_col] >= q_low) & (df[target_col] <= q_high)]\n"
                "after = len(df)\n\n"
                "f'Deleted {before - after} rows outside [{q_low:.2f}, {q_high:.2f}] range.'"
            )

        # 4. ML ENCODING (Dummy vs Label)
        if s_col4.button("🔢 ML Encode", use_container_width=True):
            load_snippet(
                "# --- CHOOSE YOUR METHOD ---\n"
                "target_col = 'column_name_here'\n"
                "method = 'dummy'  # Options: 'dummy' or 'label'\n\n"
                "if method == 'dummy':\n"
                "    # Creates multiple 0/1 columns (Good for Nominal data)\n"
                "    df = pd.get_dummies(df, columns=[target_col], prefix='is')\n"
                "else:\n"
                "    # Converts text to unique integers: 0, 1, 2... (Good for Ordinal data)\n"
                "    df[target_col] = pd.factorize(df[target_col])[0]\n\n"
                "df.head()"
            )
                    
        # 2. Python Editor with dynamic placeholder
        default_msg = f"# Targeting: {current_file}\n# Write any Python query using 'df'...\n"
        code_input = st.text_area("Python Code Editor", 
                                value=st.session_state.get("neural_code_area", default_msg), 
                                height=250, key="neural_code_area")
        
        col_run, col_clear, _ = st.columns([1, 1, 3])
        
        if col_run.button("▶️ EXECUTE CODE", use_container_width=True, type="primary"):
            import re
            local_context = {"df": df.copy(), "pd": pd, "np": np, "px": px, "st": st, "re": re, "universal_neural_parser": universal_neural_parser}
            try:
                buffer = io.StringIO(); sys.stdout = buffer
                tree = ast.parse(code_input)
                if tree.body:
                    last_node = tree.body[-1]
                    if isinstance(last_node, ast.Expr):
                        exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), '<string>', 'exec'), {}, local_context)
                        result = eval(compile(ast.Expression(body=last_node.value), '<string>', 'eval'), {}, local_context)
                    else:
                        exec(code_input, {}, local_context)
                        result = None
                
                sys.stdout = sys.__stdout__
                
                # --- CRITICAL FIX: Persist results in session state ---
                st.session_state.last_terminal = buffer.getvalue()
                st.session_state.last_result = result

                if not local_context["df"].equals(df):
                    st.session_state.main_df = local_context["df"]
                    save_to_history(local_context["df"])
                    st.toast("Intelligence Updated!", icon="✅")
                    st.session_state.needs_rerun = True 

            except Exception as e:
                sys.stdout = sys.__stdout__
                st.error(f"⚠️ Neural Execution Error: {e}")

        # 3. Persistent Result Display (Shows even after rerun)
        if st.session_state.get("last_terminal"):
            st.info("📤 Terminal Output")
            st.code(st.session_state.last_terminal)
            
        if st.session_state.get("last_result") is not None:
            st.markdown("##### 🏁 Result")
            res = st.session_state.last_result
            if isinstance(res, (pd.DataFrame, pd.Series)): 
                st.dataframe(res, use_container_width=True)
            else: 
                st.write(res)

        if col_clear.button("🧹 RESET"):
            st.session_state.neural_code_area = default_msg
            st.session_state.last_terminal = ""
            st.session_state.last_result = None
            st.rerun()

    with tab_export:
        st.markdown("#### 📤 EXPORT HUB")
        ex1, ex2 = st.columns(2)
        with ex1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📄 DOWNLOAD CSV", data=csv_data, file_name="AnalyticaX_Report.csv", mime="text/csv", use_container_width=True)
        with ex2:
            buffer_xlsx = io.BytesIO()
            with pd.ExcelWriter(buffer_xlsx, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='AnalyticaX_Export')
            st.download_button(label="📗 DOWNLOAD EXCEL", data=buffer_xlsx.getvalue(), file_name="AnalyticaX_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

else:
    st.markdown("""
        <div class="idle-container">
            <div class="idle-logo data-waves"><span></span><span></span><span></span></div>
            <h2 class="idle-title">SYSTEM IDLE</h2>
            <p class="idle-subtitle">Upload a dataset to initialize AnalyticaX logic.</p>
        </div>
    """, unsafe_allow_html=True)

# --- Post-Render Listener ---
# This ensures that if the console updated the data, the rest of the app refreshes to show it.
if st.session_state.needs_rerun:
    st.session_state.needs_rerun = False
    st.rerun()