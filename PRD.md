# Product Requirements Document (PRD)
# Masters of Analytics: Enrollment & Preferences Explorer

## Overview

The **Masters of Analytics: Enrollment & Preferences Explorer** is a data visualization application designed to help academic administrators analyze and track enrollment patterns and student preferences for INDENG courses in the Masters of Analytics program. This interactive dashboard provides insights into historical enrollment trends and student course preferences to support data-driven decisions in course scheduling and resource allocation.

## Business Objectives

- Enable administrators to identify popular courses based on historical enrollment data
- Provide insights into student course preferences to optimize course offerings
- Support resource allocation decisions through data visualization
- Track enrollment trends across multiple semesters
- Consolidate and standardize enrollment and preference data in one tool

## Target Users

- Academic administrators and program coordinators
- Department chairs and faculty involved in curriculum planning
- Enrollment management teams
- Academic advisors

## Features

### Data Management

1. **File Upload**
   - Support for multiple CSV file uploads
   - Separate upload areas for enrollment and preference data
   - Automated file validation based on naming convention and required fields
   - Error handling for malformed or empty files

2. **Data Formats**
   - Enrollment CSV format standardization
   - Preference data parsing and normalization
   - Auto-detection of semester/year from filenames using regex

3. **Demo Files**
   - Sample data generation for testing
   - Downloadable example files for reference

### Data Processing

1. **Enrollment Analysis**
   - Aggregation of enrollment numbers by course
   - Cross-semester comparison of enrollment trends
   - Automatic detection and labeling of semester/year information

2. **Preference Processing**
   - Parsing of comma-separated course preferences
   - Ranking categorization (1st, 2nd, 3rd, 4th choice)
   - Support for different preference data formats

### Visualization

1. **Enrollment Visualization**
   - Interactive bar charts showing enrollment by course
   - Multi-year comparison view
   - Customizable display options

2. **Preference Visualization**
   - Stacked bar charts for preference distribution by course
   - Color-coded preference rankings
   - Interactive tooltips with detailed information

### User Interface

1. **Layout**
   - Wide layout for comprehensive data display
   - Sidebar for configuration options
   - Expandable sections for optional information

2. **Display Options**
   - Toggles for showing/hiding data tables
   - Visualization display controls
   - Year/semester filtering capabilities

3. **Filtering**
   - Minimum enrollment threshold filtering
   - Minimum preference count filtering
   - Year-specific data views

4. **Export**
   - Download processed data as CSV or Excel
   - Separate download options for enrollment and preference data

## Technical Requirements

### Platform

- **Deployed on**: Hugging Face Spaces
- **Framework**: Streamlit 1.32.0
- **Language**: Python 3.8+

### Dependencies

- **Data Handling**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **UI**: streamlit
- **Utilities**: re, io, functools, collections

### System Architecture

- **Application Type**: Web-based dashboard
- **Deployment Model**: Cloud-hosted application
- **Data Storage**: File-based (no persistent database)
- **Preprocessing Pipeline**: File upload → Validation → Processing → Visualization

## Performance Requirements

- Support for CSV files up to 10MB in size
- Responsive UI with minimal processing delays
- Support for multiple concurrent users
- Efficient memory usage for data processing

## Security & Privacy

- No persistent storage of uploaded data
- No collection of personally identifiable information
- Session-based data handling (data cleared on session end)

## Out of Scope

- User authentication and role-based access
- Real-time data synchronization with student information systems
- Predictive analytics for future enrollment trends
- Course scheduling automation
- Student-facing interfaces

## Future Enhancements

- Integration with institutional databases
- Enhanced data visualization options
- Export to additional formats (PDF, PowerPoint)
- Predictive analytics for enrollment forecasting
- Custom reporting capabilities
- Course dependency and prerequisite visualization
- Student demographic analysis

## Release Plan

### Version 1.0 (Current)
- Basic data upload and visualization functionality
- Support for enrollment and preference data
- Interactive visualizations for both data types
- Filtering and export capabilities

### Version 2.0 (Planned)
- Enhanced data integration capabilities
- Advanced analytics features
- Improved UI/UX
- Custom reporting

## Maintenance Plan

- Quarterly updates for bug fixes
- Annual feature enhancements
- Ongoing compatibility with Streamlit framework updates
- Regular dependency updates

---

Document Version: 1.0  
Last Updated: May 2023  
Author: IEOR 180 Team 