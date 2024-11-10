# Student-Metrics-Analysis-With-Python
Welcome to the Student Performance Analysis project! This repository contains a comprehensive Python script designed to analyze and visualize various factors affecting student performance. Leveraging statistical methods and machine learning techniques, this analysis provides insights into how study habits, sleep patterns, attendance, stress levels, and other factors correlate with academic outcomes.

Features
Correlation Analysis: Examine relationships between study hours, sleep patterns, attendance, and performance metrics.
Data Visualization: Generate insightful plots and charts to visualize data distributions and correlations.
Clustering: Segment students into performance-based clusters using K-Means clustering.
Statistical Analysis: Perform ANOVA and other statistical tests to understand the impact of different factors.
Performance Comparison: Compare top and bottom performers across various metrics.
Dataset
The analysis uses a synthetically generated dataset representing student performance metrics, created by the generate_fake_data function. The dataset includes the following features:

Study Hours Per Week (study_hours_per_week): Number of hours a student dedicates to studying each week.
Quiz Average (quiz_average): Average score (%) across quizzes.
Sleep Hours (sleep_hours): Average hours of sleep per night.
Attendance Rate (attendance_rate): Percentage of classes attended.
Project Score (project_score): Score (%) achieved in projects.
Participation Score (participation_score): Score (%) based on class participation.
Extracurricular Hours (extracurricular_hours): Hours spent on extracurricular activities weekly.
Assignments Completed (assignments_completed): Number of assignments completed.
Stress Level (stress_level): Self-reported stress level on a scale (e.g., 1-10).
Usage
This script will:

Generate synthetic student data.
Perform various analyses (correlation, clustering, statistical tests).
Display plots and visualizations for each analysis task.
Print summaries and interpretations of the results.
Analysis Tasks
The script performs the following analysis tasks:

Study Hours and Quiz Performance Correlation
Sleep Pattern Impact Analysis
Attendance and Project Performance Analysis
Stress Level Impact Analysis
Time Management Analysis
Student Performance Cluster Analysis
Assignment Completion Patterns
Participation and Attendance Analysis
Study Efficiency Analysis
Top vs Bottom Performers Comparison
Each task includes both statistical computations and visualizations to provide a comprehensive understanding of the factors influencing student performance.

Visualizations
The analysis generates various plots to visualize the data and results:

Scatter Plots: Showing correlations between variables.
Box Plots: Comparing distributions across categories.
Heatmaps: Displaying correlation matrices.
Violin Plots: Illustrating the distribution of scores.
KDE Plots: Showing density distributions.
Radar Charts: Comparing performance profiles.
Visualizations are displayed using Matplotlib and Seaborn, with interactive plots available via Plotly.

Technical Details
This project uses a variety of Python libraries to perform data generation, analysis, and visualization:

Pandas: For data manipulation and analysis.
NumPy: To handle numerical operations, particularly for random data generation.
Seaborn: Provides aesthetic statistical plots, used for box plots, scatter plots, and more.
Matplotlib: Core library for plotting, enabling a wide range of customization options.
Scipy: For statistical analysis, including correlation and ANOVA.
Scikit-Learn: Includes KMeans clustering and StandardScaler for machine learning tasks.
Plotly: Used to create interactive visualizations, enhancing exploration and understanding of data.
Each section of the analysis code highlights the specific libraries and functions used, making it easy to follow along and understand each step of the process.

License
This project is licensed under the MIT License.

Contact
For any questions or suggestions, feel free to reach out:

Email: amiranikvariani@gmail.com
