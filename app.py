import streamlit as st
import pandas as pd
import numpy as np

# --- Load Data ---
df = pd.read_csv('combined_output_merged_input_nanremoved.csv')

df = df[(df["IsSpeedDropValid"]==1) & 
        (df["IsApparentSlipValid"]==1) &
        (df["IsDeltaNpropValid"]==1) &
        (df["IsDeltaPDOnSpeedValid"]==1) &
        (df["IsDeltaFOCMEValid"]==1)]

# --- Preprocess: Convert date column ---
df['StartDateUTC'] = pd.to_datetime(df['StartDateUTC'])

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Mode selection
mode = st.sidebar.radio("Select Mode", ["Normal", "Comparison"])

# Column selection
value_columns = ['MEFOCDeviation', 'PwrDlvrDevOnSpeed', 'SpeedDrop', 'ApparentSlip', 'RPMDeviation', 'ISOSFOCDeviation']
selected_column = st.sidebar.selectbox("Select column to display", value_columns)

df = df[(df[selected_column]>=0) & (df[selected_column]<=100)]  # Filter to keep values between 0 and 100

# Date filters based on mode
min_date = df['StartDateUTC'].min().date()
max_date = df['StartDateUTC'].max().date()

if mode == "Normal":
    # Single date range for normal mode
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    df_filtered = df[(df['StartDateUTC'] >= pd.to_datetime(start_date)) & (df['StartDateUTC'] <= pd.to_datetime(end_date))]
else:
    # Two date ranges for comparison mode
    st.sidebar.markdown("### Date Range 1")
    start_date_1 = st.sidebar.date_input("Start Date 1", min_date, min_value=min_date, max_value=max_date, key="start1")
    end_date_1 = st.sidebar.date_input("End Date 1", max_date, min_value=min_date, max_value=max_date, key="end1")
    
    st.sidebar.markdown("### Date Range 2")
    start_date_2 = st.sidebar.date_input("Start Date 2", min_date, min_value=min_date, max_value=max_date, key="start2")
    end_date_2 = st.sidebar.date_input("End Date 2", max_date, min_value=min_date, max_value=max_date, key="end2")
    
    # Filter data for both date ranges
    df_range1 = df[(df['StartDateUTC'] >= pd.to_datetime(start_date_1)) & (df['StartDateUTC'] <= pd.to_datetime(end_date_1))]
    df_range2 = df[(df['StartDateUTC'] >= pd.to_datetime(start_date_2)) & (df['StartDateUTC'] <= pd.to_datetime(end_date_2))]

# Vessel filter
vessel_names = {
    1023: "MH Perseus",
    1005: "PISCES", 
    1007: "CAPELLA",
    1017: "CETUS",
    1004: "CASSIOPEIA",
    1021: "PYXIS",
    1032: "Cenataurus",
    1016: "CHARA",
    1018: "CARINA"
}

# Get unique vessel IDs and create display options
vessel_ids = df['VesselId'].unique()
vessel_options = [vessel_names.get(vid, f"Unknown ({vid})") for vid in vessel_ids]

# Create selectbox with vessel names
selected_vessel_name = st.sidebar.selectbox("Select Vessel", vessel_options)

# Get the corresponding vessel ID
selected_vessel_id = None
for vid, name in vessel_names.items():
    if name == selected_vessel_name:
        selected_vessel_id = vid
        break

# If it's an unknown vessel, extract ID from the display string
if selected_vessel_id is None:
    if "Unknown (" in selected_vessel_name:
        selected_vessel_id = int(selected_vessel_name.split("(")[1].split(")")[0])

# Filter dataframe by vessel
if mode == "Normal":
    filtered_df = df_filtered[df_filtered['VesselId'] == selected_vessel_id].copy()
else:
    filtered_df_range1 = df_range1[df_range1['VesselId'] == selected_vessel_id].copy()
    filtered_df_range2 = df_range2[df_range2['VesselId'] == selected_vessel_id].copy()

def create_bins(df):
    """Create draft and speed bins for a dataframe"""
    df['DraftBin'] = np.floor(df['MeanDraft'] * 2) / 2
    df['DraftBinLabel'] = df['DraftBin'].astype(str) + "–" + (df['DraftBin'] + 0.5).astype(str)
    
    df['SpeedBin'] = np.floor(df['SpeedOG'])
    df['SpeedBinLabel'] = df['SpeedBin'].astype(str) + "–" + (df['SpeedBin'] + 1).astype(str)
    return df

def create_pivot(df, column):
    """Create pivot table for a dataframe"""
    pivot = df.groupby(['DraftBinLabel', 'SpeedBinLabel']).agg(
        avg_value=(column, 'mean'),
        count=(column, 'count')
    ).reset_index()
    return pivot

def get_all_labels(df1, df2=None):
    """Get all unique draft and speed labels from one or two dataframes"""
    if df2 is not None:
        combined_df = pd.concat([df1, df2])
    else:
        combined_df = df1
    
    draft_labels = sorted(combined_df['DraftBinLabel'].unique(), key=lambda x: float(x.split("–")[0]))
    speed_labels = sorted(combined_df['SpeedBinLabel'].unique(), key=lambda x: float(x.split("–")[0]))
    return draft_labels, speed_labels

def create_matrix(pivot, draft_labels, speed_labels):
    """Create matrix from pivot table"""
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
    return matrix

# --- Process data based on mode ---
if mode == "Normal":
    # Normal mode processing
    filtered_df = create_bins(filtered_df)
    pivot = create_pivot(filtered_df, selected_column)
    draft_labels, speed_labels = get_all_labels(filtered_df)
    matrix = create_matrix(pivot, draft_labels, speed_labels)
    
    # Display normal matrix
    st.markdown(f"<h4 style='text-align: center;'>Matrix of {selected_column} (%)</h4>", unsafe_allow_html=True)
    
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
    
    # Final Table
    table_html = f"""
    <div style='display: flex; justify-content: center; width: 100%;'>
        <table style='border-collapse: collapse; table-layout: auto;'>
            <thead>{header_html}</thead>
            <tbody>{body_html}</tbody>
        </table>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)

else:
    # Comparison mode processing
    filtered_df_range1 = create_bins(filtered_df_range1)
    filtered_df_range2 = create_bins(filtered_df_range2)
    
    pivot1 = create_pivot(filtered_df_range1, selected_column)
    pivot2 = create_pivot(filtered_df_range2, selected_column)
    
    # Get all unique labels from both ranges
    draft_labels, speed_labels = get_all_labels(filtered_df_range1, filtered_df_range2)
    
    matrix1 = create_matrix(pivot1, draft_labels, speed_labels)
    matrix2 = create_matrix(pivot2, draft_labels, speed_labels)
    
    # Display comparison matrix
    st.markdown(f"<h4 style='text-align: center;'>Comparison Matrix of {selected_column} (%)</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Range 1: {start_date_1} to {end_date_1} | Range 2: {start_date_2} to {end_date_2}</p>", unsafe_allow_html=True)
    
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
            cell1 = matrix1[draft].get(speed)
            cell2 = matrix2[draft].get(speed)
            
            # Only show comparison if both ranges have data
            if cell1 and cell2:
                avg1, cnt1 = cell1
                avg2, cnt2 = cell2
                
                # Only calculate difference if both ranges have actual data points
                if cnt1 > 0 and cnt2 > 0:
                    difference = round(avg2 - avg1, 2)
                    
                    # Determine color based on difference
                    color = "green" if difference >= 0 else "red"
                    
                    html = f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 16px; font-weight: bold; color: {color};'>{difference:+.2f}</div>
                        <div style='font-size: 10px; color: #666;'>({avg2} - {avg1})</div>
                        <div style='font-size: 10px; color: #666;'>({cnt2} | {cnt1})</div>
                    </div>
                    """
                else:
                    # One or both ranges have no data points
                    html = "<div style='text-align: center; color: #bbb;'>—</div>"
            else:
                # At least one range has no data for this cell
                html = "<div style='text-align: center; color: #bbb;'>—</div>"
            
            row += f"<td style='padding: 8px;'>{html}</td>"
        row += "</tr>"
        body_html += row
    
    # Final Table
    table_html = f"""
    <div style='display: flex; justify-content: center; width: 100%;'>
        <table style='border-collapse: collapse; table-layout: auto;'>
            <thead>{header_html}</thead>
            <tbody>{body_html}</tbody>
        </table>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Legend
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p><strong>Legend:</strong></p>
        <p><span style='color: green;'>Green</span> = Range 2 > Range 1 (Positive difference)</p>
        <p><span style='color: red;'>Red</span> = Range 2 < Range 1 (Negative difference)</p>
        <p>Format: Difference (Range2 - Range1)<br>Count: (Range2 count | Range1 count)</p>
    </div>
    """, unsafe_allow_html=True)