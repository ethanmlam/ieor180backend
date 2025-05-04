# Masters of Analytics: Enrollment & Preferences Explorer
## Demo Script

Hello everyone! Today I'm excited to showcase our new "Masters of Analytics: Enrollment & Preferences Explorer" - a powerful analytics tool built with Streamlit that helps program administrators visualize and understand student enrollment patterns and course preferences over time.

## Introduction (0:30)

This application was designed to solve several key challenges for our Masters program:
- Tracking historical enrollment trends across multiple semesters
- Understanding student course preferences and demand
- Making data-driven decisions about course offerings
- Simplifying the complex data processing workflow for program staff

Let me walk you through the main features and how they address these needs.

## Interface Overview (1:00)

As you can see, the interface is clean and intuitive with:
- A wide layout optimized for data visualization
- Clear file upload sections for both enrollment and preference data
- Comprehensive sidebar options for customizing your analysis
- Interactive charts with hover details for deep exploration
- Data export capabilities for further analysis

## Data Format Information (1:30)

The application includes built-in documentation through this information expander. This explains exactly what file formats are expected:

For enrollment files:
- Files must contain "Enrollment" and semester/year in the filename (like "Enrollment_SP24")
- CSV format with columns for Subject, Catalog Number, and Section
- Each row represents one enrolled student

For preferences files:
- Files must contain "Form Responses" and semester/year in the filename
- CSV format with student course preferences in a specific format
- The app automatically extracts and ranks preferences from 1st to 4th

## Demo Data Functionality (2:00)

For new users, we've included sample data files that can be downloaded directly from the app:
- Toggle the "Show Demo Files Section" in the sidebar
- Download example enrollment and preference files
- Use these to understand the expected format
- Upload them back to see how the visualizations work

## Uploading and Processing Files (2:30)

Now let me demonstrate the file upload process:
1. Upload enrollment CSVs from multiple semesters
2. Upload preference survey results from multiple semesters  
3. The app automatically:
   - Validates filenames and formats
   - Extracts semester/year information
   - Processes and merges data across semesters
   - Handles various CSV formats and edge cases

## Enrollment Data Analysis (3:30)

Let's look at the enrollment visualization features:
- Merged enrollment table showing trends across semesters
- Interactive bar chart displaying enrollment by course and semester
- Ability to hover over bars to see exact enrollment counts
- Option to download the processed data in CSV or Excel format

Key features include:
- Automatic integer conversion for clean data display
- Intelligent sorting of course numbers (numeric first, then alphanumeric)
- Chronological ordering of semester data
- Minimum enrollment filters to focus on popular courses

## Course Preferences Analysis (4:30)

The preferences visualization is even more powerful:
- Summarized preference data showing 1st through 4th choices
- Year-based filtering to view specific semesters or all combined
- Stacked bar chart showing preference distribution by rank
- Color-coded preference ranks for immediate visual understanding

Advanced features include:
- Regular expression parsing to handle various preference formats
- Robust data handling for different CSV structures
- Grand total calculations to identify highest demand courses
- Minimum preference filters to identify popular courses

## Display Customization (5:30)

The sidebar offers extensive customization options:
- Toggle individual visualization components on/off
- Show processing tables to understand data transformations
- Set minimum thresholds for enrollment or preferences
- Choose download format (CSV or Excel)
- Filter data by specific years

## Data Export (6:00)

For further analysis, all processed data can be exported:
- Enrollment data with counts by semester
- Preference data with detailed rank breakdowns
- Choose between CSV or Excel formats
- Use the downloaded data in other analytics tools

## Error Handling (6:30)

The application includes robust error handling:
- Validation of file formats and naming conventions
- Helpful error messages for troubleshooting
- Graceful handling of empty or malformed files
- Multiple fallback parsing strategies for different data formats

## Conclusion (7:00)

In summary, this application transforms complex enrollment and preference data into actionable insights:
- Track historical course demand across semesters
- Identify trends and shifts in student preferences
- Make informed decisions about course offerings
- Save hours of manual data processing and analysis

By centralizing this analysis in a user-friendly web application, we've made these insights accessible to all stakeholders in the Masters program without requiring specialized data science skills.

Thank you for watching this demonstration. Are there any questions about specific features you'd like to explore further? 