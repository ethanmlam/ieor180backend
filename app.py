import os
os.environ["STREAMLIT_HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
from functools import reduce
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import base64

st.set_page_config(layout="wide")
st.title("Masters of Analytics: Enrollment & Preferences Explorer")

# Initialize session state for demo files if not already present
if 'demo_loaded' not in st.session_state:
    st.session_state.demo_loaded = False

# Function to create sample data files for download
def get_sample_data():
    # Sample enrollment data
    enrollment_data = """Subject,Catalog Nbr,Section
INDENG,221,1
INDENG,221,1
INDENG,221,1
INDENG,221,1
INDENG,221,1
INDENG,231,1
INDENG,231,1
INDENG,231,1
INDENG,242B,1
INDENG,242B,1
INDENG,242B,1
INDENG,242B,1
INDENG,242B,1
INDENG,242B,1
INDENG,C253,1
INDENG,C253,1
INDENG,C253,1"""

    # Sample preferences data
    preferences_data = """Timestamp,Email Address,Preferences
2022-01-15,student1@example.com,"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management"
2022-01-16,student2@example.com,"INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II"
2022-01-17,student3@example.com,"INDENG 242B Machine Learning and Data Analytics II, INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation"
2022-01-18,student4@example.com,"INDENG C253 Supply Chain and Logistics Management, INDENG 242B Machine Learning and Data Analytics II, INDENG 221 Introduction to Financial Engineering"
2022-01-19,student5@example.com,"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG C253 Supply Chain and Logistics Management, INDENG 221 Introduction to Financial Engineering"
2022-01-20,student6@example.com,"INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation"
2022-01-21,student7@example.com,"INDENG 221 Introduction to Financial Engineering, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management"
2022-01-22,student8@example.com,"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 221 Introduction to Financial Engineering, INDENG 242B Machine Learning and Data Analytics II"
2022-01-23,student9@example.com,"INDENG C253 Supply Chain and Logistics Management, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 221 Introduction to Financial Engineering"
2022-01-24,student10@example.com,"INDENG 242B Machine Learning and Data Analytics II, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG C253 Supply Chain and Logistics Management"
"""
    
    return enrollment_data, preferences_data

# Add info button explaining data formats
with st.expander("ℹ️ Data Format Information"):
    st.markdown("""
    ### Expected Data Formats
    
    #### Preferences CSV Format
    Preferences files should contain comma-separated course preferences like:
    ```
    INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management
    
    INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II
    ```
    
    #### Enrollment CSV Format
    Enrollment files should have the following structure:
    ```
    Subject    Catalog Nbr    Section
    INDENG     221            1
    INDENG     221            1
    INDENG     221            1
    ```
    
    Each row represents one enrolled student. The app will count them to determine total enrollment per course.
    """)

# Sidebar for display options
st.sidebar.header("Display Options")
show_processing_tables = st.sidebar.checkbox("Show Processing Tables", value=False)
show_merged_enrollment = st.sidebar.checkbox("Show Merged Enrollment Table", value=True)
show_enrollment_overview = st.sidebar.checkbox("Show Enrollment Data Overview", value=True)

show_merged_preferences = st.sidebar.checkbox("Show Merged Preferences Table", value=True)
show_preference_chart = st.sidebar.checkbox("Show Preference Chart", value=True)

# Add download option for processed data
st.sidebar.header("Download Options")
download_format = st.sidebar.selectbox("Select Format", ["CSV", "Excel"], index=0)

# Add course filter option
st.sidebar.header("Filter Options")
min_enrollment = st.sidebar.number_input("Min Enrollment (if applicable)", min_value=0, value=0)
min_preferences = st.sidebar.number_input("Min Preferences (if applicable)", min_value=0, value=0)

# Function to extract year label from filename
def extract_label_from_filename(filename):
    match = re.search(r"SP(\d+)", filename.upper())
    if match:
        return f"Sp 20{match.group(1)[-2:]}"
    return "Unknown"

# Function to validate file names for preferences
def is_valid_preferences_file(filename):
    return "Form Responses" in filename

# Function to validate file names for enrollment
def is_valid_enrollment_file(filename):
    return "Enrollment" in filename

# Function to process enrollment files
def process_enrollment_file(uploaded_file, label):
    try:
        df = pd.read_csv(uploaded_file)
        if 'Catalog Nbr' in df.columns:
            df = df.drop(columns=[col for col in ['Subject', 'Unnamed: 3', 'Unnamed: 4'] if col in df.columns], errors='ignore')
            df = df.groupby('Catalog Nbr').sum(numeric_only=True).rename(columns={'Section': f'Enrolled {label}'})
            df = df[[f'Enrolled {label}']].reset_index()
            return df
        else:
            st.warning(f"Could not find 'Catalog Nbr' column in {uploaded_file.name}")
            return None
    except Exception as e:
        st.error(f"Error reading enrollment file '{uploaded_file.name}': {str(e)}")
        return None

# Function to check if a file is empty or has valid data
def is_valid_csv(file):
    # Reset file pointer to the beginning
    file.seek(0)
    
    # Read a sample of the file to check if it has content
    sample = file.read(min(1024, file.size))
    
    # Reset file pointer again
    file.seek(0)
    
    # Check if there's any content or at least a header
    return len(sample.strip()) > 0 and b',' in sample

# Function to process preferences files by splitting comma-separated entries
def process_preference_file(uploaded_file, label):
    try:
        # Check if the file is empty
        if not is_valid_csv(uploaded_file):
            st.warning(f"File '{uploaded_file.name}' appears to be empty or not a valid CSV. Skipping...")
            return None
        
        # Try to read with different approaches
        try:
            df = pd.read_csv(uploaded_file)
        except pd.errors.EmptyDataError:
            st.warning(f"Empty data in file '{uploaded_file.name}'. Skipping...")
            return None
        except Exception as e:
            # Try with different delimiter if comma doesn't work
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, sep='\t')  # Try tab delimiter
            except Exception:
                st.error(f"Could not parse '{uploaded_file.name}' with common delimiters.")
                return None
        
        # Verify we have some data
        if df.empty or len(df.columns) == 0:
            st.warning(f"No usable data found in '{uploaded_file.name}'. Skipping...")
            return None
            
        # Process preferences by rank (focusing on 1st to 4th)
        preference_counts = defaultdict(lambda: defaultdict(int))
        
        # Handle different data formats
        for _, row in df.iterrows():
            # Find columns that might contain preference data
            pref_data = None
            
            # Try to find a column with comma-separated preferences
            for col in row.index:
                if pd.notna(row[col]) and isinstance(row[col], str) and ',' in row[col] and 'INDENG' in row[col]:
                    pref_data = row[col]
                    break
            
            # If we found a column with comma-separated preferences
            if pref_data:
                # Split by comma to get individual preferences
                preferences = [p.strip() for p in pref_data.split(',') if p.strip()]
                
                # Process each preference by its position in the list (rank)
                for idx, entry in enumerate(preferences):
                    if idx < 4:  # Only count the first 4 preferences
                        # Extract course number (e.g., INDENG 221)
                        match = re.search(r"INDENG\s+([A-Z]?\d+[A-Z]?)", entry)
                        if match:
                            catalog = match.group(1)
                            rank = f"{idx+1}st" if idx == 0 else f"{idx+1}nd" if idx == 1 else f"{idx+1}rd" if idx == 2 else f"{idx+1}th"
                            preference_counts[catalog][rank] += 1
            else:
                # Try to handle individual columns for each preference
                for idx, col in enumerate(df.columns):
                    if idx < 4 and pd.notna(row[col]) and isinstance(row[col], str) and 'INDENG' in row[col]:
                        match = re.search(r"INDENG\s+([A-Z]?\d+[A-Z]?)", row[col])
                        if match:
                            catalog = match.group(1)
                            rank = f"{idx+1}st" if idx == 0 else f"{idx+1}nd" if idx == 1 else f"{idx+1}rd" if idx == 2 else f"{idx+1}th"
                            preference_counts[catalog][rank] += 1
        
        # If we didn't find any preferences, return None
        if not preference_counts:
            st.warning(f"No course preferences found in '{uploaded_file.name}'. Skipping...")
            return None
            
        # Create DataFrame with standardized columns for 1st to 4th preferences
        all_catalogs = sorted(preference_counts.keys())
        standard_ranks = ['1st', '2nd', '3rd', '4th']
        
        pref_df = pd.DataFrame(index=all_catalogs, columns=standard_ranks).fillna(0)
        
        for catalog in preference_counts:
            for rank in preference_counts[catalog]:
                if rank[:3] in standard_ranks:  # Only include 1st, 2nd, 3rd, 4th preferences
                    pref_df.at[catalog, rank[:3]] = preference_counts[catalog][rank]
        
        pref_df = pref_df.astype(int)
        pref_df["Total"] = pref_df.sum(axis=1)
        
        # Add year label only to the total column
        pref_df = pref_df.rename(columns={"Total": f"Total {label}"})
        pref_df.index.name = 'Course'
        
        return pref_df.reset_index()
        
    except Exception as e:
        st.error(f"Error processing preference file '{uploaded_file.name}': {str(e)}")
        return None

# Upload multiple enrollment and preferences CSVs
col1, col2 = st.columns(2)
with col1:
    uploaded_enrolls = st.file_uploader("Upload Enrollment CSV(s) - Must contain 'Enrollment' in filename", 
                                       type="csv", key="enroll", accept_multiple_files=True)
    
    # Check if any uploaded files don't match the required format
    if uploaded_enrolls:
        invalid_files = [f.name for f in uploaded_enrolls if not is_valid_enrollment_file(f.name)]
        if invalid_files:
            st.error(f"The following files are not valid enrollment files and will be ignored: {', '.join(invalid_files)}")
            # Filter out invalid files
            uploaded_enrolls = [f for f in uploaded_enrolls if is_valid_enrollment_file(f.name)]
            
with col2:
    uploaded_prefs = st.file_uploader("Upload Preferences CSV(s) - Must contain 'Form Responses' in filename", 
                                     type=["csv"], key="prefs", accept_multiple_files=True)
    
    # Check if any uploaded files don't match the required format
    if uploaded_prefs:
        invalid_files = [f.name for f in uploaded_prefs if not is_valid_preferences_file(f.name)]
        if invalid_files:
            st.error(f"The following files are not valid preference files and will be ignored: {', '.join(invalid_files)}")
            # Filter out invalid files
            uploaded_prefs = [f for f in uploaded_prefs if is_valid_preferences_file(f.name)]

if uploaded_enrolls and show_enrollment_overview:
    st.subheader("📊 Enrollment Data Overview")
    try:
        # Process enrollment files
        all_enrollment_dfs = []
        
        for uploaded_file in uploaded_enrolls:
            year_label = extract_label_from_filename(uploaded_file.name)
            df_processed = process_enrollment_file(uploaded_file, year_label)
            
            if df_processed is not None:
                all_enrollment_dfs.append(df_processed)
                if show_processing_tables:
                    st.dataframe(df_processed.head())
        
        # Merge all enrollment DataFrames
        if all_enrollment_dfs:
            merged_df = reduce(lambda left, right: pd.merge(left, right, on='Catalog Nbr', how='outer'), all_enrollment_dfs)
            merged_df = merged_df.fillna(0)
            
            # Convert to integers to remove decimal points
            for col in merged_df.columns:
                if col != 'Catalog Nbr':
                    merged_df[col] = merged_df[col].astype(int)
            
            # Sort year columns chronologically
            enrollment_cols = sorted(
                [col for col in merged_df.columns if col.startswith("Enrolled Sp")],
                key=lambda x: int(x.split()[-1])
            )
            merged_df = merged_df[['Catalog Nbr'] + enrollment_cols]
            
            # Filter by minimum enrollment if specified
            if min_enrollment > 0:
                # Create a column with the maximum enrollment across years
                merged_df['Max_Enrolled'] = merged_df[[col for col in merged_df.columns if col.startswith('Enrolled')]].max(axis=1)
                merged_df = merged_df[merged_df['Max_Enrolled'] >= min_enrollment]
                merged_df = merged_df.drop(columns=['Max_Enrolled'])
            
            if show_merged_enrollment:
                st.write("Merged Enrollment Data Across Years:")
                st.dataframe(merged_df)
                
                # Add download button for enrollment data
                if not merged_df.empty:
                    buffer = io.BytesIO()
                    if download_format == "CSV":
                        merged_df.to_csv(buffer, index=False)
                        file_ext = "csv"
                    else:  # Excel
                        merged_df.to_excel(buffer, index=False)
                        file_ext = "xlsx"
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download Enrollment Data",
                        data=buffer,
                        file_name=f"enrollment_data.{file_ext}",
                        mime="application/octet-stream"
                    )
            
            # Create visualization of enrollment trends using Plotly for interactive hover
            plot_data = merged_df.reset_index().melt(
                id_vars=['Catalog Nbr'],
                value_vars=enrollment_cols,
                var_name='Year',
                value_name='Enrollment'
            )
            
            # Create interactive bar chart with Plotly
            fig = px.bar(
                plot_data, 
                x='Catalog Nbr', 
                y='Enrollment', 
                color='Year',
                title="Enrollment Trends by Course and Year",
                labels={'Catalog Nbr': 'Course Number', 'Enrollment': 'Enrollment Count'},
                hover_data=['Catalog Nbr', 'Year', 'Enrollment']
            )
            
            fig.update_layout(
                xaxis_title="Course Number",
                yaxis_title="Enrollment Count",
                legend_title="Year",
                barmode='group',
                xaxis={'type': 'category', 'categoryorder': 'category ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error processing enrollment files: {e}")
        st.exception(e)
        st.info("Make sure your enrollment files match the expected format shown in the information section.")

if uploaded_prefs:
    st.subheader("📈 Preferences Data Overview")
    try:
        # Process preference files
        all_preference_dfs = []
        year_labels = []
        
        for uploaded_file in uploaded_prefs:
            year_label = extract_label_from_filename(uploaded_file.name)
            year_labels.append(year_label)
            
            df_processed = process_preference_file(uploaded_file, year_label)
            
            if df_processed is not None:
                all_preference_dfs.append(df_processed)
                if show_processing_tables:
                    st.write(f"Processed preference counts for {uploaded_file.name}:")
                    st.dataframe(df_processed.head())
        
        # Merge all preference DataFrames
        if all_preference_dfs:
            # Merge the preference dataframes on 'Course' column
            merged_prefs_df = reduce(lambda left, right: pd.merge(left, right, on='Course', how='outer'), all_preference_dfs)
            merged_prefs_df = merged_prefs_df.fillna(0)
            
            # Convert to integers to remove decimal points
            for col in merged_prefs_df.columns:
                if col != 'Course':
                    merged_prefs_df[col] = merged_prefs_df[col].astype(int)
            
            # Create a custom tidy dataframe for display
            # Start with a copy of the Course column
            tidy_df = pd.DataFrame({'Course': merged_prefs_df['Course']})
            
            # Add preference rank columns (1st, 2nd, 3rd, 4th) with summed values across years
            for rank in ['1st', '2nd', '3rd', '4th']:
                # Sum all columns containing this rank
                rank_cols = [col for col in merged_prefs_df.columns if rank in col]
                if rank_cols:
                    tidy_df[rank] = merged_prefs_df[rank_cols].sum(axis=1)
                else:
                    tidy_df[rank] = 0
            
            # Add total by year columns
            for label in year_labels:
                total_col = f"Total {label}"
                if total_col in merged_prefs_df.columns:
                    tidy_df[total_col] = merged_prefs_df[total_col]
            
            # Add Grand Total column
            tidy_df["Grand Total"] = tidy_df[['1st', '2nd', '3rd', '4th']].sum(axis=1)
            
            # Sort by Grand Total descending
            tidy_df = tidy_df.sort_values("Grand Total", ascending=False)
            
            # Filter by minimum preferences if specified
            if min_preferences > 0:
                tidy_df = tidy_df[tidy_df["Grand Total"] >= min_preferences]
            
            if show_merged_preferences:
                st.write("Preferences Summary by Course (Across All Years):")
                st.dataframe(tidy_df)
                
                # Add download button for preferences data
                if not tidy_df.empty:
                    buffer = io.BytesIO()
                    if download_format == "CSV":
                        tidy_df.to_csv(buffer, index=False)
                        file_ext = "csv"
                    else:  # Excel
                        tidy_df.to_excel(buffer, index=False)
                        file_ext = "xlsx"
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download Preferences Data",
                        data=buffer,
                        file_name=f"preferences_data.{file_ext}",
                        mime="application/octet-stream"
                    )
            
            # Create visualization of preference totals if enabled
            if show_preference_chart and not tidy_df.empty:
                # Prepare data for plotting with Plotly
                plot_data = tidy_df.copy()
                
                # Create a Plotly figure for stacked bar chart
                fig = go.Figure()
                
                # Add each preference rank as a separate bar in the stack
                for rank, color in zip(['1st', '2nd', '3rd', '4th'], 
                                      ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
                    fig.add_trace(go.Bar(
                        x=plot_data['Course'],
                        y=plot_data[rank],
                        name=rank,
                        marker_color=color,
                        hovertemplate='Course: %{x}<br>' +
                                      'Rank: ' + rank + '<br>' +
                                      'Count: %{y}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Preference Distribution by Course",
                    xaxis_title="Course",
                    yaxis_title="Number of Preferences",
                    legend_title="Preference Rank",
                    barmode='stack',
                    hovermode='closest',
                    xaxis={'type': 'category', 'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif not tidy_df.empty:
                st.info("Preference chart display is disabled. Enable it in the sidebar to see the visualization.")
            else:
                st.warning("No preference data to display in chart.")
                
    except Exception as e:
        st.error(f"Error processing preferences files: {e}")
        st.exception(e)
        st.info("Make sure your preferences files match the expected format shown in the information section.")

# If no data is uploaded, show a welcome message and sample data download options
if not uploaded_enrolls and not uploaded_prefs:
    st.info("Please upload Enrollment and/or Preferences CSV files to get started.")
    
    # Add sample data download section
    st.subheader("🧪 Sample Data Files")
    st.write("Download these sample files to test the functionality of the app:")
    
    # Get sample data
    sample_enrollment, sample_preferences = get_sample_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create download button for sample enrollment data
        enrollment_buffer = io.BytesIO(sample_enrollment.encode())
        st.download_button(
            label="Download Sample Enrollment Data",
            data=enrollment_buffer,
            file_name="Enrollment_SP22.csv",
            mime="text/csv"
        )
    
    with col2:
        # Create download button for sample preferences data
        preferences_buffer = io.BytesIO(sample_preferences.encode())
        st.download_button(
            label="Download Sample Preferences Data",
            data=preferences_buffer,
            file_name="Form Responses SP22.csv",
            mime="text/csv"
        )
