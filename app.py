import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# --- Load Data ---
df = pd.read_csv('combined_output_merged_input_nanremoved.csv')

df = df[(df["IsSpeedDropValid"]==1) & 
        (df["IsApparentSlipValid"]==1) &
        (df["IsDeltaNpropValid"]==1) &
        (df["IsDeltaPDOnSpeedValid"]==1) &
        (df["IsDeltaFOCMEValid"]==1)]

# --- Preprocess: Convert date column ---
df['StartDateUTC'] = pd.to_datetime(df['StartDateUTC'])

# --- Initialize session state ---
if 'selected_draft' not in st.session_state:
    st.session_state.selected_draft = None
if 'selected_speed' not in st.session_state:
    st.session_state.selected_speed = None
if 'plot_type' not in st.session_state:
    st.session_state.plot_type = "Draft-wise"

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
    start_date_1 = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="start1")
    end_date_1 = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="end1")
    
    st.sidebar.markdown("### Date Range 2")
    start_date_2 = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="start2")
    end_date_2 = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="end2")
    
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

def get_top_cells_to_highlight(matrix, draft_labels, speed_labels, mode="normal"):
    """Determine which cells to highlight based on the specified logic"""
    highlight_cells = set()
    
    for draft in draft_labels:
        row_cells = []
        
        for speed in speed_labels:
            cell = matrix[draft].get(speed)
            if cell:
                if mode == "normal":
                    avg, cnt = cell
                    row_cells.append((speed, cnt))  # Use count instead of average
                elif mode == "comparison":
                    # For comparison mode, we need both counts > 5 and consider absolute difference
                    # This will be handled in the comparison mode processing
                    pass
        
        # Determine how many cells to highlight based on count
        num_cells = len(row_cells)
        if num_cells > 9:
            top_n = 3
        elif 5 <= num_cells <= 9:
            top_n = 2
        elif num_cells < 5 and num_cells > 0:
            top_n = 1
        else:
            top_n = 0
        
        if top_n > 0:
            # Sort by count (descending for normal mode - highest count first)
            row_cells.sort(key=lambda x: x[1], reverse=True)
            for i in range(min(top_n, len(row_cells))):
                highlight_cells.add((draft, row_cells[i][0]))
    
    return highlight_cells

def get_top_cells_to_highlight_comparison(matrix1, matrix2, draft_labels, speed_labels):
    """Determine which cells to highlight in comparison mode"""
    highlight_cells = set()
    
    for draft in draft_labels:
        row_cells = []
        
        for speed in speed_labels:
            cell1 = matrix1[draft].get(speed)
            cell2 = matrix2[draft].get(speed)
            
            if cell1 and cell2:
                avg1, cnt1 = cell1
                avg2, cnt2 = cell2
                
                # Both counts should be greater than 5
                if cnt1 > 5 and cnt2 > 5:
                    abs_diff = abs(cnt2 - cnt1)
                    max_count = max(cnt1, cnt2)
                    row_cells.append((speed, abs_diff, max_count))
        
        # Determine how many cells to highlight based on count
        num_cells = len(row_cells)
        if num_cells > 9:
            top_n = 3
        elif 5 <= num_cells <= 9:
            top_n = 2
        elif num_cells < 5 and num_cells > 0:
            top_n = 1
        else:
            top_n = 0
        
        if top_n > 0:
            # Sort by absolute difference (ascending) then by max count (descending)
            row_cells.sort(key=lambda x: (x[1], -x[2]))
            for i in range(min(top_n, len(row_cells))):
                highlight_cells.add((draft, row_cells[i][0]))
    
    return highlight_cells

def add_trendline_and_stats(fig, x_data, y_data, name_suffix="", color="blue"):
    """Add trendline and average to plot"""
    if len(x_data) > 1:
        # Create trendline using linear regression
        x_clean = np.array(x_data).reshape(-1, 1)
        y_clean = np.array(y_data)
        
        # Remove NaN values
        mask = ~(np.isnan(x_clean.flatten()) | np.isnan(y_clean))
        if np.sum(mask) > 1:
            x_clean = x_clean[mask]
            y_clean = y_clean[mask]

            # Calculate and add average line
            avg_y = np.mean(y_clean)
            
            # model = LinearRegression()
            # model.fit(x_clean, y_clean)
            
            # x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
            # y_trend = model.predict(x_trend.reshape(-1, 1))

            degree = 2  # Use 2 for quadratic, 3 for cubic, etc.
            coeffs = np.polyfit(x_clean.flatten(), y_clean, degree)
            poly = np.poly1d(coeffs)

            x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_trend = poly(x_trend)


            trend_color = color
            if color == "blue":
                trend_color = "green"
            if color == "red":
                trend_color = "orange"
            
            # Add trendline
            fig.add_trace(go.Scatter(
                x=x_trend.flatten(),
                y=y_trend,
                mode='lines',
                name=f'Trend{name_suffix.replace("Range", "")}',
                line=dict(color=trend_color, width=3)
            ))
            
            
            # fig.add_hline(
            #     y=avg_y,
            #     line_dash="dot",
            #     line_color=color,
            #     annotation_text=f"Avg {name_suffix}: {avg_y:.2f}",
            #     annotation_position="top right"
            # )

            # Get current title (if any) and append avg info
            current_title = fig.layout.title.text if fig.layout.title.text else ""
            new_title = f"{current_title}<br> Avg{name_suffix.replace("Range","")}: {avg_y:.2f}"
            fig.update_layout(title=new_title)



def create_speed_wise_scatterplot(df, speed_label, selected_column, mode="normal", df2=None):
    """Create scatterplot for selected speed range (Column vs Draft)"""
    speed_start = float(speed_label.split("–")[0])
    speed_end = float(speed_label.split("–")[1])
    
    # Filter data for the speed range
    df_plot = df[(df['SpeedOG'] >= speed_start) & (df['SpeedOG'] < speed_end)]
    
    if mode == "normal":
        # fig = px.scatter(df_plot, x='MeanDraft', y=selected_column, 
        #                 title=f'{selected_column} vs Draft',
        #                 labels={'MeanDraft': 'Mean Draft', selected_column: selected_column})
        # fig.update_traces(marker=dict(color='blue', size=8))
        
        # # Add trendline and average
        # if len(df_plot) > 1:
        #     add_trendline_and_stats(fig, df_plot['MeanDraft'], df_plot[selected_column], color="blue")
        # Comparison mode
        
        
        fig = go.Figure()
        
        # Add Range 1 data
        fig.add_trace(go.Scatter(
            x=df_plot['MeanDraft'], 
            y=df_plot[selected_column],
            mode='markers',
            name='DataPoints',
            marker=dict(color='blue', size=8)
        ))
        

        fig.update_layout(
            title=f'{selected_column} vs Draft',
            xaxis_title='Mean Draft',
            yaxis_title=selected_column
        )
        
        # Add trendlines and averages for both ranges
        if len(df_plot) > 1:
            add_trendline_and_stats(fig, df_plot['MeanDraft'], df_plot[selected_column], color="blue")
        
        
            
    else:
        # Comparison mode
        df2_plot = df2[(df2['SpeedOG'] >= speed_start) & (df2['SpeedOG'] < speed_end)]
        
        fig = go.Figure()
        
        # Add Range 1 data
        fig.add_trace(go.Scatter(
            x=df_plot['MeanDraft'], 
            y=df_plot[selected_column],
            mode='markers',
            name='Range 1',
            marker=dict(color='blue', size=8)
        ))
        
        # Add Range 2 data
        fig.add_trace(go.Scatter(
            x=df2_plot['MeanDraft'], 
            y=df2_plot[selected_column],
            mode='markers',
            name='Range 2',
            marker=dict(color='red', size=8)
        ))

        fig.update_layout(
            title=f'{selected_column} vs Draft',
            xaxis_title='Mean Draft',
            yaxis_title=selected_column
        )
        
        # Add trendlines and averages for both ranges
        if len(df_plot) > 1:
            add_trendline_and_stats(fig, df_plot['MeanDraft'], df_plot[selected_column], "Range 1", "blue")
        if len(df2_plot) > 1:
            add_trendline_and_stats(fig, df2_plot['MeanDraft'], df2_plot[selected_column], "Range 2", "red")
        
        
    
    return fig

def create_draft_wise_scatterplot(df, draft_label, selected_column, mode="normal", df2=None):
    """Create scatterplot for selected draft range (Column vs Speed)"""
    draft_start = float(draft_label.split("–")[0])
    draft_end = float(draft_label.split("–")[1])
    
    # Filter data for the draft range
    df_plot = df[(df['MeanDraft'] >= draft_start) & (df['MeanDraft'] < draft_end)]
    
    if mode == "normal":
        # fig = px.scatter(df_plot, x='SpeedOG', y=selected_column, 
        #                 title=f'{selected_column} vs Speed',
        #                 labels={'SpeedOG': 'Speed OG', selected_column: selected_column})
        # fig.update_traces(marker=dict(color='blue', size=8))
        
        # # Add trendline and average
        # if len(df_plot) > 1:
        #     add_trendline_and_stats(fig, df_plot['SpeedOG'], df_plot[selected_column], color="blue")

        
        
        fig = go.Figure()
        
        # Add Range 1 data
        fig.add_trace(go.Scatter(
            x=df_plot['SpeedOG'], 
            y=df_plot[selected_column],
            mode='markers',
            name='DataPoints',
            marker=dict(color='blue', size=8)
        ))
        
        
        # Add trendlines and averages for both ranges

        fig.update_layout(
            title=f'{selected_column} vs Speed',
            xaxis_title='Speed OG',
            yaxis_title=selected_column
        )

        if len(df_plot) > 1:
            add_trendline_and_stats(fig, df_plot['SpeedOG'], df_plot[selected_column], color="blue")
        
        
        
            
    else:
        # Comparison mode
        df2_plot = df2[(df2['MeanDraft'] >= draft_start) & (df2['MeanDraft'] < draft_end)]
        
        fig = go.Figure()
        
        # Add Range 1 data
        fig.add_trace(go.Scatter(
            x=df_plot['SpeedOG'], 
            y=df_plot[selected_column],
            mode='markers',
            name='Range 1',
            marker=dict(color='blue', size=8)
        ))
        
        # Add Range 2 data
        fig.add_trace(go.Scatter(
            x=df2_plot['SpeedOG'], 
            y=df2_plot[selected_column],
            mode='markers',
            name='Range 2',
            marker=dict(color='red', size=8)
        ))

        fig.update_layout(
            title=f'{selected_column} vs Speed',
            xaxis_title='Speed OG',
            yaxis_title=selected_column
        )
        
        # Add trendlines and averages for both ranges
        if len(df_plot) > 1:
            add_trendline_and_stats(fig, df_plot['SpeedOG'], df_plot[selected_column], "Range 1", "blue")
        if len(df2_plot) > 1:
            add_trendline_and_stats(fig, df2_plot['SpeedOG'], df2_plot[selected_column], "Range 2", "red")
        
        
    
    return fig

# --- Process data based on mode ---
if mode == "Normal":
    # Normal mode processing
    filtered_df = create_bins(filtered_df)
    pivot = create_pivot(filtered_df, selected_column)
    draft_labels, speed_labels = get_all_labels(filtered_df)
    matrix = create_matrix(pivot, draft_labels, speed_labels)
    
    # Get cells to highlight
    highlight_cells = get_top_cells_to_highlight(matrix, draft_labels, speed_labels, "normal")
    
    # Create main layout with matrix and scatterplot side by side
    col_matrix, col_plot = st.columns([2.5, 1.5])
    
    with col_matrix:
        st.markdown(f"<h5 style='text-align: center;'>Matrix of {selected_column} (%)</h5>", unsafe_allow_html=True)
        
        # Header
        header_html = "<tr><th style='font-size: 10px; padding: 4px; white-space: nowrap;'>Draft ↓ / Speed →</th>"
        for speed in speed_labels:
            header_html += f"<th style='padding: 4px; text-align: center; white-space: nowrap; font-size: 10px;'>{speed}</th>"
        header_html += "</tr>"
        
        # Body
        body_html = ""
        for draft in draft_labels:
            row = f"<tr><td style='padding: 3px; font-weight: bold; white-space: nowrap; background-color: #f0f0f0; font-size: 10px;'>{draft}</td>"
            
            for speed in speed_labels:
                cell = matrix[draft].get(speed)
                cell_style = "padding: 4px; font-size: 9px;"
                
                # Check if this cell should be highlighted
                if (draft, speed) in highlight_cells:
                    cell_style += " background-color: #ffeb3b; border: 2px solid #fbc02d;"
                
                if cell:
                    avg, cnt = cell
                    html = f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 11px; font-weight: bold;'>{avg}</div>
                        <div style='background-color: #007bff; color: white; border-radius: 50%; width: 16px; height: 16px; display: inline-block; line-height: 16px; font-size: 8px;'>{cnt}</div>
                    </div>
                    """
                else:
                    html = "<div style='text-align: center; color: #bbb; font-size: 12px;'>—</div>"
                row += f"<td style='{cell_style}'>{html}</td>"
            row += "</tr>"
            body_html += row
        
        # Final Table
        table_html = f"""
        <div style='display: flex; justify-content: center; width: 100%;'>
            <table style='border-collapse: collapse; table-layout: auto; font-size: 10px;'>
                <thead>{header_html}</thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
    
    with col_plot:
        st.markdown("### 📊 Scatterplot ")
        
        # Plot type selection
        plot_type = st.radio("Select Plot Type:", ["Draft-wise", "Speed-wise"], key="plot_type_normal")
        
        if plot_type == "Draft-wise":
            # Draft selection for speed vs column plot
            draft_options = ["Select a draft range..."] + draft_labels
            selected_draft_index = st.selectbox(
                "Choose draft range:",
                range(len(draft_options)),
                format_func=lambda x: draft_options[x],
                key="draft_selector_normal"
            )
            
            if selected_draft_index > 0:
                selected_draft = draft_options[selected_draft_index]
                fig = create_draft_wise_scatterplot(filtered_df, selected_draft, selected_column, "normal")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a draft range to view scatterplot")
                
        else:  # Speed-wise
            # Speed selection for draft vs column plot
            speed_options = ["Select a speed range..."] + speed_labels
            selected_speed_index = st.selectbox(
                "Choose speed range:",
                range(len(speed_options)),
                format_func=lambda x: speed_options[x],
                key="speed_selector_normal"
            )
            
            if selected_speed_index > 0:
                selected_speed = speed_options[selected_speed_index]
                fig = create_speed_wise_scatterplot(filtered_df, selected_speed, selected_column, "normal")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a speed range to view scatterplot")

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
    
    # Get cells to highlight
    highlight_cells = get_top_cells_to_highlight_comparison(matrix1, matrix2, draft_labels, speed_labels)
    
    # Create main layout with matrix and scatterplot side by side
    col_matrix, col_plot = st.columns([2.5, 1.5])
    
    with col_matrix:
        st.markdown(f"<h5 style='text-align: center;'>Comparison Matrix of {selected_column}</h5>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 10px;'>Range 1: {start_date_1} to {end_date_1}<br>Range 2: {start_date_2} to {end_date_2}</p>", unsafe_allow_html=True)
        
        # Header
        header_html = "<tr><th style='font-size: 10px; padding: 4px; white-space: nowrap;'>Draft ↓ / Speed →</th>"
        for speed in speed_labels:
            header_html += f"<th style='padding: 4px; text-align: center; white-space: nowrap; font-size: 10px;'>{speed}</th>"
        header_html += "</tr>"
        
        # Body
        body_html = ""
        for draft in draft_labels:
            row = f"<tr><td style='padding: 3px; font-weight: bold; white-space: nowrap; background-color: #f0f0f0; font-size: 10px;'>{draft}</td>"
            for speed in speed_labels:
                cell1 = matrix1[draft].get(speed)
                cell2 = matrix2[draft].get(speed)
                
                cell_style = "padding: 4px; font-size: 9px;"
                
                # Check if this cell should be highlighted
                if (draft, speed) in highlight_cells:
                    cell_style += " background-color: #ffeb3b; border: 2px solid #fbc02d;"
                
                # Only show comparison if both ranges have data
                if cell1 and cell2:
                    avg1, cnt1 = cell1
                    avg2, cnt2 = cell2
                    
                    # Only calculate difference if both ranges have actual data points
                    if cnt1 > 0 and cnt2 > 0:
                        difference = round(avg2 - avg1, 2)
                        
                        # Determine color based on difference
                        color = "red" if difference >= 0 else "green"
                        
                        html = f"""
                        <div style='text-align: center;'>
                            <div style='font-size: 11px; font-weight: bold; color: {color};'>{difference:+.2f}</div>
                            <div style='font-size: 7px; color: #666;'>({avg2} - {avg1})</div>
                            <div style='font-size: 7px; color: #666;'>({cnt2} | {cnt1})</div>
                        </div>
                        """
                    else:
                        # One or both ranges have no data points
                        html = "<div style='text-align: center; color: #bbb; font-size: 12px;'>—</div>"
                else:
                    # At least one range has no data for this cell
                    html = "<div style='text-align: center; color: #bbb; font-size: 12px;'>—</div>"
                
                row += f"<td style='{cell_style}'>{html}</td>"
            row += "</tr>"
            body_html += row
        
        # Final Table
        table_html = f"""
        <div style='display: flex; justify-content: center; width: 100%;'>
            <table style='border-collapse: collapse; table-layout: auto; font-size: 10px;'>
                <thead>{header_html}</thead>
                <tbody>{body_html}</tbody>
            </table>
        </div>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Legend
        st.markdown("""
        <div style='text-align: center; margin-top: 10px; font-size: 10px;'>
            <p><strong>Legend:</strong></p>
            <p><span style='color: red;'>Red</span> = R2 > R1 | <span style='color: green;'>Green</span> = R2 < R1</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_plot:
        st.markdown("### 📊 Comparison Scatterplot ")
        
        # Plot type selection
        plot_type = st.radio("Select Plot Type:", ["Draft-wise", "Speed-wise"], key="plot_type_comparison")
        
        if plot_type == "Draft-wise":
            # Draft selection for speed vs column plot
            draft_options = ["Select a draft range..."] + draft_labels
            selected_draft_index = st.selectbox(
                "Choose draft range:",
                range(len(draft_options)),
                format_func=lambda x: draft_options[x],
                key="draft_selector_comparison"
            )
            
            if selected_draft_index > 0:
                selected_draft = draft_options[selected_draft_index]
                fig = create_draft_wise_scatterplot(filtered_df_range1, selected_draft, selected_column, "comparison", filtered_df_range2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a draft range to view comparison scatterplot")
                
        else:  # Speed-wise
            # Speed selection for draft vs column plot
            speed_options = ["Select a speed range..."] + speed_labels
            selected_speed_index = st.selectbox(
                "Choose speed range:",
                range(len(speed_options)),
                format_func=lambda x: speed_options[x],
                key="speed_selector_comparison"
            )
            
            if selected_speed_index > 0:
                selected_speed = speed_options[selected_speed_index]
                fig = create_speed_wise_scatterplot(filtered_df_range1, selected_speed, selected_column, "comparison", filtered_df_range2)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select a speed range to view comparison scatterplot")