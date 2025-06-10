import streamlit as st
import pandas as pd
import numpy as np

# --- Load Data ---
df = pd.read_csv('combined_output_merged_input_nanremoved.csv')

# --- Preprocess: Convert date column ---
df['StartDateUTC'] = pd.to_datetime(df['StartDateUTC'])

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Column selection
value_columns = ['MEFOCDeviation', 'PwrDlvrDevOnSpeed', 'SpeedDrop', 'ApparentSlip', 'RPMDeviation', 'ISOSFOCDeviation']  # <-- update this list as needed
selected_column = st.sidebar.selectbox("Select column to display", value_columns)

# # StartDateUTC filter
# min_date = df['StartDateUTC'].min()
# max_date = df['StartDateUTC'].max()
# start_date, end_date = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
# df = df[(df['StartDateUTC'] >= pd.to_datetime(start_date)) & (df['StartDateUTC'] <= pd.to_datetime(end_date))]

# # Beaufort scale filter
# bfscale_options = sorted(df['BFScale'].dropna().unique())
# selected_bfscale = st.sidebar.multiselect("Select BFScale", bfscale_options, default=bfscale_options)
# df = df[df['BFScale'].isin(selected_bfscale)]

# --- Dynamic Advanced Filters ---
operators = {
    '=': lambda a, b: a == b,
    '!=': lambda a, b: a != b,
    '<': lambda a, b: a < b,
    '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b,
    '>=': lambda a, b: a >= b
}

filter_columns = st.sidebar.multiselect("Choose columns to filter", df.columns)

for col in filter_columns:
    if f"{col}_conditions" not in st.session_state:
        st.session_state[f"{col}_conditions"] = 1

    st.sidebar.markdown(f"### Filters for {col}")

    for i in range(st.session_state[f"{col}_conditions"]):
        op = st.sidebar.selectbox(f"{col} - Condition {i+1} Operator", list(operators.keys()), key=f"{col}_op_{i}")

        if pd.api.types.is_numeric_dtype(df[col]):
            val = st.sidebar.number_input(f"{col} - Value {i+1}", key=f"{col}_val_{i}")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            val = st.sidebar.date_input(f"{col} - Date {i+1}", key=f"{col}_val_{i}")
            # Convert date to datetime for comparison
            val = pd.to_datetime(val)
        else:
            val = st.sidebar.text_input(f"{col} - Value {i+1}", key=f"{col}_val_{i}")

        # Apply filter
        try:
            df = df[operators[op](df[col], val)]
        except Exception as e:
            st.warning(f"Error applying filter on {col}: {e}")

    if st.sidebar.button(f"➕ Add another condition for {col}", key=f"{col}_add_btn"):
        st.session_state[f"{col}_conditions"] += 1

# Vessel filter
vessels = df['VesselId'].unique()
selected_vessel = st.sidebar.selectbox("Select Vessel", vessels)
filtered_df = df[df['VesselId'] == selected_vessel].copy()

# --- Create Bins ---
filtered_df['DraftBin'] = np.floor(filtered_df['MeanDraft'] * 2) / 2
filtered_df['DraftBinLabel'] = filtered_df['DraftBin'].astype(str) + "–" + (filtered_df['DraftBin'] + 0.5).astype(str)

filtered_df['SpeedBin'] = np.floor(filtered_df['SpeedOG'])
filtered_df['SpeedBinLabel'] = filtered_df['SpeedBin'].astype(str) + "–" + (filtered_df['SpeedBin'] + 1).astype(str)

# --- Compute Pivot Table ---
pivot = filtered_df.groupby(['DraftBinLabel', 'SpeedBinLabel']).agg(
    avg_value=(selected_column, 'mean'),
    count=(selected_column, 'count')
).reset_index()

# --- Create Labels ---
draft_labels = sorted(pivot['DraftBinLabel'].unique(), key=lambda x: float(x.split("–")[0]))
speed_labels = sorted(pivot['SpeedBinLabel'].unique(), key=lambda x: float(x.split("–")[0]))

# --- Create Matrix ---
matrix = {}
for draft in draft_labels:
    matrix[draft] = {}
    for speed in speed_labels:
        sub = pivot[(pivot['DraftBinLabel'] == draft) & (pivot['SpeedBinLabel'] == speed)]
        if not sub.empty:
            avg = round(sub['avg_value'].values[0], 2)
            cnt = sub['count'].values[0]
            matrix[draft][speed] = (avg, cnt)
        else:
            matrix[draft][speed] = None

# --- Display Matrix ---
# --- Display Matrix ---
st.markdown(f"<h4 style='text-align: center;'>Matrix of {selected_column} (Average + Count)</h4>", unsafe_allow_html=True)

# Header
header_html = "<tr><th style='white-space: nowrap;'>Draft ↓ / Speed →</th>"
for speed in speed_labels:
    header_html += f"<th style='padding: 10px; text-align: center; white-space: nowrap;'>{speed}</th>"
header_html += "</tr>"

# Body
body_html = ""
for draft in draft_labels:
    row = f"<tr><td style='padding: 6px; font-weight: bold; white-space: nowrap;'>{draft}</td>"
    for speed in speed_labels:
        cell = matrix[draft].get(speed)
        if cell:
            avg, cnt = cell
            html = f"""
            <div style='text-align: center;'>
                <div style='font-size: 16px; font-weight: bold;'>{avg}</div>
                <div style='background-color: #007bff; color: white; border-radius: 50%; width: 24px; height: 24px; display: inline-block; line-height: 24px; font-size: 12px;'>{cnt}</div>
            </div>
            """
        else:
            html = "<div style='text-align: center; color: #bbb;'>—</div>"
        row += f"<td style='padding: 8px;'>{html}</td>"
    row += "</tr>"
    body_html += row

# Final Table with centering div
table_html = f"""
<div style='display: flex; justify-content: center; width: 100%;'>
    <table style='border-collapse: collapse; table-layout: auto;'>
        <thead>{header_html}</thead>
        <tbody>{body_html}</tbody>
    </table>
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)
