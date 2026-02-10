import pandas as pd
import numpy as np
import io
import re
import streamlit as st
import time

@st.cache_data(show_spinner=False)
def load_data(file_content, file_name):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        stages = [("Initializing Neural Link...", 0.2), ("Decoding Data Packets...", 0.5), 
                  ("Mapping Feature Vectors...", 0.8), ("Syncing Intelligence Node...", 1.0)]
        for msg, prog in stages:
            status_text.markdown(f"**PROCESS:** `{msg}`")
            progress_bar.progress(prog)
            time.sleep(0.3)
            
        data_stream = io.BytesIO(file_content)
        if file_name.endswith(".csv"):
            df = pd.read_csv(data_stream)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(data_stream)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(data_stream)
        elif file_name.endswith(".json"):
            df = pd.read_json(data_stream)
        else:
            st.error("Format not supported by Neural Engine.")
            return None
        progress_bar.empty()
        status_text.empty()
        return df.loc[:, ~df.columns.str.contains("^Unnamed")]
    except Exception as e:
        st.error(f"Logic Error: {e}")
        return None

def apply_nlp_filter(df, query):
    query = query.lower().strip()
    if not query: return df
    conditions = re.split(r'\s+and\s+', query)
    temp_df = df.copy()
    cols = df.columns.tolist()
    for cond in conditions:
        try:
            target_col = next((c for c in cols if c.lower() in cond), None)
            if not target_col: continue
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", cond)
            val = float(nums[0]) if nums else None
            
            # Logic for >= and <=
            if ">=" in cond or "at least" in cond or "minimum" in cond:
                temp_df = temp_df[temp_df[target_col] >= val]
            elif "<=" in cond or "at most" in cond or "maximum" in cond:
                temp_df = temp_df[temp_df[target_col] <= val]
            elif any(x in cond for x in [">", "greater", "more", "above"]):
                temp_df = temp_df[temp_df[target_col] > val]
            elif any(x in cond for x in ["<", "less", "under", "below"]):
                temp_df = temp_df[temp_df[target_col] < val]
            elif any(x in cond for x in ["is", "equals", "==", "contain", "like"]):
                clean_val = re.split(r'is|equals|==|contain|like', cond)[-1].strip().strip("'\"")
                if temp_df[target_col].dtype in [np.float64, np.int64]:
                    temp_df = temp_df[temp_df[target_col] == float(clean_val)]
                else:
                    temp_df = temp_df[temp_df[target_col].astype(str).str.lower().str.contains(clean_val)]
        except: continue
    return temp_df