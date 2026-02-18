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

def universal_neural_parser(value):
    # 0. Quick Null Check
    if pd.isna(value) or str(value).strip().lower() in ["none", "nan", "", "n/a", "-", "null"]:
        return np.nan
    
    # 1. Cleaning & Fast-Pass
    # Strip spaces and basic currency/commas for a quick numeric check
    val = str(value).lower().strip()
    clean_val = re.sub(r'[$,₹£€,]', '', val) 
    
    # --- ADDED: THE FAST-PASS ---
    try:
        # If it's already "50", "60.5", or "1e3", this returns immediately
        return pd.to_numeric(clean_val)
    except (ValueError, TypeError):
        # If it contains units like "50kg", it will fail and continue to regex
        pass
    # ---------------------------

    # 2. Extract Number (Handles scientific notation like 1e3 and decimals)
    number_match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", val)
    if not number_match:
        return np.nan
    
    number = float(number_match.group(1))
    
    # 3. Comprehensive Multiplier Map
    # Note: 'mb' is the base (1) for storage here
    multipliers = {
        'crore': 10_000_000, 'cr': 10_000_000,
        'lakh': 100_000, 'lakhs': 100_000, 'lac': 100_000, 'lpa': 100_000,
        'billion': 1_000_000_000, 'b': 1_000_000_000,
        'million': 1_000_000, 'm': 1_000_000,
        'k': 1_000, 'thousand': 1_000,
        'tonne': 1_000_000, 'ton': 1_000_000,
        'kg': 1_000, 'kilogram': 1_000,
        'mg': 0.001, 'milligram': 0.001,
        'gram': 1, 'gm': 1,
        'km': 1_000, 'kilometer': 1_000,
        'cm': 0.01, 'centimeter': 0.01,
        'mm': 0.001, 'millimeter': 0.001,
        'meter': 1,
        'hr': 3600, 'hour': 3600, 'hrs': 3600,
        'min': 60, 'minute': 60, 'mins': 60,
        'sec': 1, 'second': 1, 'secs': 1,
        'tb': 1024 * 1024, 'gb': 1024, 'mb': 1, 'kb': 1/1024
    }
    
    # 4. Smart Matching for Units
    for unit, factor in multipliers.items():
        if re.search(rf"{unit}\b", val):
            return number * factor
            
    return number