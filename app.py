import os
os.environ["STREAMLIT_HOME"] = "/tmp"

import streamlit as st
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
hide_spaces_header = """
    <style>
    /* Hide Hugging Face top bar */
    .st-emotion-cache-6qob1r.e1tzin5v2 {
        display: none !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_spaces_header, unsafe_allow_html=True)
# Function to create sample data files for demo
def create_sample_data():
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
    preferences_data = """Preferences
"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management"
"INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II"
"INDENG 242B Machine Learning and Data Analytics II, INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation"
"INDENG C253 Supply Chain and Logistics Management, INDENG 242B Machine Learning and Data Analytics II, INDENG 221 Introduction to Financial Engineering"
"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG C253 Supply Chain and Logistics Management, INDENG 221 Introduction to Financial Engineering"
"INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation"
"INDENG 221 Introduction to Financial Engineering, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management"
"INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 221 Introduction to Financial Engineering, INDENG 242B Machine Learning and Data Analytics II"
"INDENG C253 Supply Chain and Logistics Management, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 221 Introduction to Financial Engineering"
"INDENG 242B Machine Learning and Data Analytics II, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG C253 Supply Chain and Logistics Management"
"""

    return enrollment_data, preferences_data

# Add info button explaining data formats
with st.expander("ℹ️ Data Format Information"):
    st.markdown("""
    ### CSV Upload Requirements
    - Enrollment files must contain 'Enrollment' and SemesterYear (ie FA25, SP24) in the filename
    - Preferences files must contain 'Form Responses' and SemesterYear (ie FA25, SP24) in the filename
    
    ### Expected Data Formats
    
    #### Preferences CSV Format
    Preferences files should contain comma-separated course preferences like:
    ```
    INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II, INDENG C253 Supply Chain and Logistics Management
    
    INDENG 221 Introduction to Financial Engineering, INDENG 231 Introduction to Data Modeling, Statistics, and System Simulation, INDENG 242B Machine Learning and Data Analytics II
    ```
    
    #### Enrollment CSV Format
    Enrollment files must have the following structure:
    ```
    Subject    Catalog Nbr    Section
    INDENG     221            1
    INDENG     221            1
    INDENG     221            1
    ```
    For best results, Enrollment files should contain only INDENG courses. Section column should be 1.
    Each row represents one enrolled student. The app will count them to determine total enrollment per course.
    The app can handle all files of this format including MEng files.
    """)

# Add demo files section toggle in sidebar
st.sidebar.header("Example Files")
show_demo_files = st.sidebar.checkbox("Show Example Files Section", value=False)

# Sidebar for display options - convert to expander to save space
with st.sidebar.expander("Display Options", expanded=False):
    show_processing_tables = st.checkbox("Show Processing Tables", value=False)
    show_merged_enrollment = st.checkbox("Show Merged Enrollment Table", value=True)
    show_enrollment_overview = st.checkbox("Show Enrollment Data Overview", value=True)
    show_merged_preferences = st.checkbox("Show Merged Preferences Table", value=True)
    show_preference_chart = st.checkbox("Show Preference Chart", value=True)

# Add download option for processed data
with st.sidebar.expander("Download Options", expanded=False):
    download_format = st.selectbox("Select Format", ["CSV", "Excel"], index=0)

# Filter options - give more prominence with dedicated section
st.sidebar.header("Filter Options")
min_enrollment = st.sidebar.number_input("Min Enrollment", min_value=0, value=0)
max_enrollment = st.sidebar.number_input("Max Enrollment", min_value=0, value=0, 
                                         help="Set to 0 for no maximum limit")
min_preferences = st.sidebar.number_input("Min Preferences (if applicable)", min_value=0, value=0)

# Create a permanent subject filter with default values
# We'll use a session state variable to store and update available subjects
if 'subject_options' not in st.session_state:
    st.session_state.subject_options = ["All Subjects", "INDENG"]
    st.session_state.selected_subjects = ["INDENG"]  # Pre-select INDENG by default

# Display the permanent filter
selected_subjects = st.sidebar.multiselect(
    "Filter by Subject:",
    st.session_state.subject_options,
    default=st.session_state.selected_subjects,
    help="Select one or more subjects. Use 'All Subjects' to see everything."
)

# Function to sort course numbers with numeric courses first, then alphanumeric
def sort_course_numbers(course_list):
    # Separate numeric and alphanumeric courses
    numeric_courses = []
    alphanumeric_courses = []
    
    for course in course_list:
        # Convert any numeric types to string first
        course_str = str(course)
        
        # Check if the course number is purely numeric
        if course_str.isdigit():
            numeric_courses.append(course_str)
        else:
            alphanumeric_courses.append(course_str)
    
    # Sort each group separately
    numeric_courses.sort(key=int)  # Sort numeric courses as integers
    alphanumeric_courses.sort()    # Sort alphanumeric courses regularly
    
    # Return numeric courses first, then alphanumeric
    return numeric_courses + alphanumeric_courses

# Function to extract year label from filename
def extract_label_from_filename(filename):
    # Check for Spring semester (SP)
    sp_match = re.search(r"SP(\d+)", filename.upper())
    if sp_match:
        return f"Sp 20{sp_match.group(1)[-2:]}"
    
    # Check for Fall semester (FA)
    fa_match = re.search(r"FA(\d+)", filename.upper())
    if fa_match:
        return f"Fa 20{fa_match.group(1)[-2:]}"
    
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
        # Read CSV with handling for blank rows
        df = pd.read_csv(uploaded_file)
        # st.dataframe(df.head(10)) 
        # Drop completely empty rows
        df = df.dropna(how='all')
        # For string columns, skip rows with only whitespace
        for col in df.columns:
            if df[col].dtype == object:  # Only check string columns
                df = df[~(df[col].astype(str).str.isspace())]
        
        if 'Catalog Nbr' in df.columns:
            # Include Subject in the groupby if it exists
            if 'Subject' in df.columns:
                # Count occurrences of each Subject + Catalog Nbr combination
                enrollment_counts = df.groupby(['Subject', 'Catalog Nbr']).size().reset_index(name=f'Enrolled {label}')
                
                # Create a combined Subject+CourseNum column for display
                enrollment_counts['Course_Full'] = enrollment_counts['Subject'] + ' ' + enrollment_counts['Catalog Nbr'].astype(str)
            else:
                # Count occurrences of each course number only
                enrollment_counts = df.groupby('Catalog Nbr').size().reset_index(name=f'Enrolled {label}')
                enrollment_counts['Subject'] = 'UNKNOWN'  # Add a placeholder Subject
                enrollment_counts['Course_Full'] = enrollment_counts['Catalog Nbr'].astype(str)
            
            # Create a single expander for file details
            with st.expander(f"File details: {uploaded_file.name}", expanded=False):
                # Display success message first
                st.success(f"Successfully processed {len(df)} rows from {uploaded_file.name}, found {len(enrollment_counts)} unique courses")
                
                # Add horizontal rule as a separator
                st.markdown("---")
                
                # Add sample data heading
                st.markdown("#### Sample Data")
                st.dataframe(df, height=400)  # Show all data but limit visible height
                
                # Show subject information if available
                if 'Subject' in df.columns:
                    st.markdown("#### Top Subjects")
                    subject_counts = df['Subject'].value_counts().head(5)
                    st.info(f"{', '.join(subject_counts.index.tolist())}")
            
            return enrollment_counts
        else:
            missing_cols = []
            if 'Catalog Nbr' not in df.columns:
                missing_cols.append('Catalog Nbr')
            st.warning(f"Could not find {', '.join(missing_cols)} column(s) in {uploaded_file.name}")
            st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            return None
    except Exception as e:
        st.error(f"Error reading enrollment file '{uploaded_file.name}': {str(e)}")
        st.exception(e)  # Show the full exception trace
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
        
        # For Form Responses files, we expect only one column with preferences
        # If there are multiple columns, we'll look for the one with INDENG in it
        preference_column = None
        if len(df.columns) == 1:
            # If there's only one column, use it regardless of name
            preference_column = df.columns[0]
        else:
            # Try to find a column with preferences
            for col in df.columns:
                sample_data = df[col].dropna().astype(str).head(5).str.contains('INDENG').sum()
                if sample_data > 0:
                    preference_column = col
                    break
        
        if preference_column is None:
            st.warning(f"No column with preference data found in '{uploaded_file.name}'. Skipping...")
            return None
            
        # Process preferences by rank (focusing on 1st to 4th)
        preference_counts = defaultdict(lambda: defaultdict(int))
        
        # Only process the identified preference column
        for _, row in df.iterrows():
            pref_text = row[preference_column]
            
            # Skip empty cells
            if pd.isna(pref_text) or not isinstance(pref_text, str):
                continue
                
            # Find all INDENG course numbers in the text
            course_numbers = re.findall(r'INDENG\s+([A-Z]?\d+[A-Z]?)', pref_text)
            
            # Process each preference by its position in the list (rank)
            for idx, catalog in enumerate(course_numbers):
                if idx < 4:  # Only count the first 4 preferences
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
        
        # Add year label to every column to identify the source year
        labeled_columns = {}
        for col in pref_df.columns:
            labeled_columns[col] = f"{col} {label}"
        pref_df = pref_df.rename(columns=labeled_columns)
        
        pref_df.index.name = 'Course'
        result_df = pref_df.reset_index()
        
        # Create a single expander for file details
        with st.expander(f"File details: {uploaded_file.name}", expanded=False):
            # Display success message first
            st.success(f"Successfully processed {len(df)} rows from {uploaded_file.name}, found {len(result_df)} unique courses")
            
            # Add horizontal rule as a separator
            st.markdown("---")
            
            # Add sample data heading
            st.markdown("#### Sample Data")
            st.dataframe(df, height=400)  # Show all data but limit visible height
            
            # Add detected courses section
            st.markdown("#### Detected Courses")
            st.write(", ".join(result_df['Course'].tolist()))
        
        return result_df
        
    except Exception as e:
        st.error(f"Error processing preference file '{uploaded_file.name}': {str(e)}")
        st.exception(e)  # Show detailed error for debugging
        return None

# Upload multiple enrollment and preferences CSVs
col1, col2 = st.columns(2)
with col1:
    uploaded_enrolls = st.file_uploader("Upload Enrollment CSV(s)", 
                                       type="csv", key="enroll", accept_multiple_files=True)
    
    # Check if any uploaded files don't match the required format
    if uploaded_enrolls:
        invalid_files = [f.name for f in uploaded_enrolls if not is_valid_enrollment_file(f.name)]
        if invalid_files:
            st.error(f"The following files are not valid enrollment files and will be ignored: {', '.join(invalid_files)}")
            # Filter out invalid files
            uploaded_enrolls = [f for f in uploaded_enrolls if is_valid_enrollment_file(f.name)]
            
with col2:
    uploaded_prefs = st.file_uploader("Upload Preferences CSV(s)", 
                                     type=["csv"], key="prefs", accept_multiple_files=True)
    
    # Check if any uploaded files don't match the required format
    if uploaded_prefs:
        invalid_files = [f.name for f in uploaded_prefs if not is_valid_preferences_file(f.name)]
        if invalid_files:
            st.error(f"The following files are not valid preference files and will be ignored: {', '.join(invalid_files)}")
            # Filter out invalid files
            uploaded_prefs = [f for f in uploaded_prefs if is_valid_preferences_file(f.name)]

# Demo Files Section (conditionally displayed)
if show_demo_files:
    st.subheader("Demo Files")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        enrollment_data, _ = create_sample_data()
        st.download_button(
            label="Download Enrollment Example",
            data=enrollment_data,
            file_name="Example_Enrollment_SP22.csv",
            mime="text/csv"
        )
        
    with demo_col2:
        _, preferences_data = create_sample_data()
        st.download_button(
            label="Download Preferences Example",
            data=preferences_data,
            file_name="Example Form Responses SP22.csv",
            mime="text/csv"
        )

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
                    with st.expander(f"Processing details for {uploaded_file.name}", expanded=False):
                        st.markdown("#### Processed Data Summary")
                        st.dataframe(df_processed, height=400)  # Show all data but limit visible height
        
        # Merge all enrollment DataFrames
        if all_enrollment_dfs:
            # Merge on both Subject and Catalog Nbr columns
            merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Subject', 'Catalog Nbr', 'Course_Full'], how='outer'), all_enrollment_dfs)
            merged_df = merged_df.fillna(0)
            
            # Convert to integers to remove decimal points
            for col in merged_df.columns:
                if col not in ['Subject', 'Catalog Nbr', 'Course_Full']:
                    merged_df[col] = merged_df[col].astype(int)
            
            # Get unique subjects for the filter and update session state
            unique_subjects = sorted(merged_df['Subject'].unique())
            
            # Update the options in session state
            st.session_state.subject_options = ["All Subjects"] + unique_subjects
            
            # Make sure selected_subjects contains valid choices
            if not any(subj in st.session_state.subject_options for subj in selected_subjects):
                # If no valid subjects selected, prioritize selecting INDENG if available
                if "INDENG" in unique_subjects:
                    st.session_state.selected_subjects = ["INDENG"]
                else:
                    # Default to "All Subjects" if INDENG is not available
                    st.session_state.selected_subjects = ["All Subjects"]
            else:
                # Store current selection in session state for persistence
                st.session_state.selected_subjects = selected_subjects
            
            # Apply subject filter if specific subjects are selected
            filtered_df = merged_df.copy()
            if selected_subjects and "All Subjects" not in selected_subjects:
                filtered_df = merged_df[merged_df['Subject'].isin(selected_subjects)]
                st.info(f"Filtering to show only: {', '.join(selected_subjects)}")
            
            # Sort year columns chronologically
            enrollment_cols = sorted(
                [col for col in filtered_df.columns if col.startswith("Enrolled")],
                key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0
            )
            
            # Reorder columns with Subject and Catalog Nbr first, then Course_Full, then enrollment data
            filtered_df = filtered_df[['Subject', 'Catalog Nbr', 'Course_Full'] + enrollment_cols]
            
            # Filter by minimum and maximum enrollment if specified
            if min_enrollment > 0 or (max_enrollment > 0):
                # Create a column with the maximum enrollment across years
                filtered_df['Max_Enrolled'] = filtered_df[[col for col in filtered_df.columns if col.startswith('Enrolled')]].max(axis=1)
                
                # Apply minimum enrollment filter
                if min_enrollment > 0:
                    filtered_df = filtered_df[filtered_df['Max_Enrolled'] >= min_enrollment]
                    
                # Apply maximum enrollment filter if specified
                if max_enrollment > 0:
                    filtered_df = filtered_df[filtered_df['Max_Enrolled'] <= max_enrollment]
                
                # Remove the helper column
                filtered_df = filtered_df.drop(columns=['Max_Enrolled'])
            
            if show_merged_enrollment:
                st.write("Merged Enrollment Data Across Years:")
                st.dataframe(filtered_df)
                
                # Add download button for enrollment data
                if not filtered_df.empty:
                    buffer = io.BytesIO()
                    if download_format == "CSV":
                        filtered_df.to_csv(buffer, index=False)
                        file_ext = "csv"
                    else:  # Excel
                        filtered_df.to_excel(buffer, index=False)
                        file_ext = "xlsx"
                    
                    buffer.seek(0)
                    st.download_button(
                        label="Download Enrollment Data",
                        data=buffer,
                        file_name=f"enrollment_data.{file_ext}",
                        mime="application/octet-stream"
                    )
            
            # Create visualization of enrollment trends using Plotly for interactive hover
            plot_data = filtered_df.reset_index().melt(
                id_vars=['Subject', 'Catalog Nbr', 'Course_Full'],
                value_vars=enrollment_cols,
                var_name='Year',
                value_name='Enrollment'
            )
            
            # Use custom sort function for Course_Full
            sorted_courses = sort_course_numbers(plot_data['Course_Full'].unique())
            
            # Create interactive bar chart with Plotly
            fig = px.bar(
                plot_data, 
                x='Course_Full',  # Use the combined Subject+CourseNum column
                y='Enrollment', 
                color='Year',
                title="Enrollment Trends by Course and Year",
                labels={'Course_Full': 'Course', 'Enrollment': 'Enrollment Count'},
                hover_data=['Subject', 'Catalog Nbr', 'Year', 'Enrollment'],
                category_orders={"Course_Full": sorted_courses}
            )
            
            # Calculate a reasonable maximum for y-axis
            max_enrollment = plot_data['Enrollment'].max()
            y_max = max(20, int(max_enrollment * 1.2))  # At least 20 or 20% above max
            
            # Set explicit height and configure layout completely
            fig.update_layout(
                height=600,  # Fixed height in pixels
                autosize=True,  # Allow horizontal resizing 
                xaxis_title="Course",
                yaxis_title="Enrollment Count",
                legend_title="Year",
                barmode='group',
                margin=dict(l=50, r=50, t=80, b=50),  # Add some margin
                xaxis=dict(
                    type='category',
                    tickmode='array',
                    tickvals=sorted_courses,
                    tickangle=-45  # Angle labels for better readability
                ),
                yaxis=dict(
                    range=[0, y_max],  # Explicit range
                    fixedrange=False  # Allow zooming on y-axis
                )
            )
            
            # Use streamlit's plotly_chart with use_container_width
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
            
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
                    with st.expander(f"Processing details for {uploaded_file.name}", expanded=False):
                        st.markdown("#### Processed Preference Counts")
                        st.dataframe(df_processed, height=400)  # Show all data but limit visible height
        
        # Merge all preference DataFrames
        if all_preference_dfs:
            # Simple year filter
            year_filter_options = ["All Years"] + sorted(year_labels)
            selected_year_option = st.selectbox("Filter by Year:", year_filter_options)
            
            # Merge the preference dataframes on 'Course' column
            try:
                merged_prefs_df = reduce(lambda left, right: pd.merge(left, right, on='Course', how='outer'), all_preference_dfs)
                merged_prefs_df = merged_prefs_df.fillna(0)
                
                # Convert to integers to remove decimal points
                for col in merged_prefs_df.columns:
                    if col != 'Course':
                        merged_prefs_df[col] = merged_prefs_df[col].astype(int)
                
                # Create a custom tidy dataframe for display
                tidy_df = pd.DataFrame({'Course': merged_prefs_df['Course']})
                
                # Get the standard rank labels
                standard_ranks = ['1st', '2nd', '3rd', '4th']
                
                # Add columns based on filter selection
                if selected_year_option == "All Years":
                    # For each standard rank, sum all columns from all years
                    for rank in standard_ranks:
                        # Find all columns for this rank across all years
                        rank_cols = [col for col in merged_prefs_df.columns if rank in col]
                        if rank_cols:
                            tidy_df[rank] = merged_prefs_df[rank_cols].sum(axis=1)
                        else:
                            tidy_df[rank] = 0
                    
                    # Add total columns for each year
                    for year in year_labels:
                        total_col = f"Total {year}"
                        if any(total_col in col for col in merged_prefs_df.columns):
                            # Find the exact column name
                            matching_col = [col for col in merged_prefs_df.columns if total_col in col]
                            if matching_col:
                                tidy_df[total_col] = merged_prefs_df[matching_col[0]]
                else:
                    # Only include columns for the selected year
                    for rank in standard_ranks:
                        # Find rank columns for this specific year 
                        rank_col = f"{rank} {selected_year_option}"
                        matching_cols = [col for col in merged_prefs_df.columns if rank_col in col]
                        
                        if matching_cols:
                            # Use the original rank name in the output dataframe
                            tidy_df[rank] = merged_prefs_df[matching_cols[0]]
                        else:
                            tidy_df[rank] = 0
                    
                    # Add total for the selected year
                    total_col = f"Total {selected_year_option}"
                    matching_total = [col for col in merged_prefs_df.columns if total_col in col]
                    if matching_total:
                        tidy_df[total_col] = merged_prefs_df[matching_total[0]]
                
                # Add Grand Total column
                tidy_df["Grand Total"] = tidy_df[standard_ranks].sum(axis=1)
                
                # Sort by Grand Total descending
                tidy_df = tidy_df.sort_values("Grand Total", ascending=False)
                
                # Filter by minimum preferences if specified
                if min_preferences > 0:
                    tidy_df = tidy_df[tidy_df["Grand Total"] >= min_preferences]
                
                if show_merged_preferences:
                    st.write(f"Preferences Summary by Course ({selected_year_option}):")
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
                    
                    # Sort courses using custom function (numeric first, then alphanumeric)
                    sorted_courses = sort_course_numbers(plot_data['Course'].unique())
                    plot_data['Course'] = pd.Categorical(plot_data['Course'], categories=sorted_courses, ordered=True)
                    plot_data = plot_data.sort_values('Course')
                    
                    # Create a Plotly figure for stacked bar chart with vertical bars
                    fig = go.Figure()
                    
                    # Add each preference rank as a separate bar in the stack
                    for rank, color in zip(standard_ranks, 
                                          ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']):
                        fig.add_trace(go.Bar(
                            x=plot_data['Course'],  # Use x for Course for vertical bars
                            y=plot_data[rank],      # Use y for values for vertical bars
                            name=rank,
                            marker_color=color,
                            hovertemplate='Course: %{x}<br>' +
                                          'Rank: ' + rank + '<br>' +
                                          'Count: %{y}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f"Preference Distribution by Course ({selected_year_option})",
                        xaxis_title="Course",
                        yaxis_title="Number of Preferences",
                        legend_title="Preference Rank",
                        barmode='stack',
                        hovermode='closest',
                        xaxis=dict(
                            type='category',
                            tickmode='array',
                            tickvals=sorted_courses,
                            tickangle=-45  # Angle the course labels for better readability
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif not tidy_df.empty:
                    st.info("Preference chart display is disabled. Enable it in the sidebar to see the visualization.")
                else:
                    st.warning("No preference data to display in chart.")
                
            except Exception as e:
                st.error(f"Error merging preference dataframes: {str(e)}")
                st.exception(e)  # Show detailed error for debugging
                all_preference_dfs = []  # Empty the list so the next condition will be false
                
    except Exception as e:
        st.error(f"Error processing preferences files: {e}")
        st.exception(e)
        st.info("Make sure your preferences files match the expected format shown in the information section.")

# Remove the demo data loading functionality at the bottom
if not uploaded_enrolls and not uploaded_prefs:
    st.info("Please upload Enrollment and/or Preferences CSV files to get started.")
