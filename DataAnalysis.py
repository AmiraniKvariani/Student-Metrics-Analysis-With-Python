import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
from scipy.stats import f_oneway



def generate_fake_data(n_students=100):
    np.random.seed(42)

    data = {

        'student_id': range(1, n_students + 1),
        'attendance_rate': np.random.uniform(0.7, 1.0, n_students),
        'study_hours_per_week': np.random.normal(15, 5, n_students),
        'assignments_completed': np.random.randint(8, 15, n_students),
        'quiz_average': np.random.normal(75, 15, n_students),
        'project_score': np.random.normal(82, 10, n_students),
        'participation_score': np.random.normal(70, 20, n_students),
        'stress_level': np.random.randint(1, 11, n_students),
        'sleep_hours': np.random.normal(7, 1.5, n_students),
        'extracurricular_hours': np.random.normal(5, 2, n_students)
    }

    df = pd.DataFrame(data)
    df['attendance_rate'] = df['attendance_rate'].clip(0, 1)
    df['quiz_average'] = df['quiz_average'].clip(0, 100)
    df['project_score'] = df['project_score'].clip(0, 100)
    df['participation_score'] = df['participation_score'].clip(0, 100)
    df['sleep_hours'] = df['sleep_hours'].clip(4, 12)
    df['study_hours_per_week'] = df['study_hours_per_week'].clip(0, 40)
    df['extracurricular_hours'] = df['extracurricular_hours'].clip(0, 15)
    return df




student_data = generate_fake_data(n_students=100)

print("\n=== Student Performance Analysis Report ===\n")

# Study Hours and Quiz Performance Correlation
print("1. STUDY HOURS AND QUIZ PERFORMANCE ANALYSIS")
study_quiz_correlation = student_data["study_hours_per_week"].corr(
    student_data["quiz_average"]
)
print(
    f"Correlation coefficient between study hours and quiz scores: {study_quiz_correlation:.2f}"
)
print(
    f"Interpretation: {'Strong' if abs(study_quiz_correlation) > 0.7 else 'Moderate' if abs(study_quiz_correlation) > 0.4 else 'Weak'} correlation"
)

plt.figure(figsize=(10, 6))
sns.regplot(data=student_data, x="study_hours_per_week", y="quiz_average")
plt.title("Impact of Study Hours on Quiz Performance")
plt.xlabel("Weekly Study Hours")
plt.ylabel("Quiz Score (%)")
plt.show()

# Sleep Pattern Analysis
print("\n2. SLEEP PATTERN IMPACT ANALYSIS")
student_data["sleep_category"] = pd.cut(
    student_data["sleep_hours"],
    bins=[0, 6, 8, 12],
    labels=["Insufficient (<6h)", "Optimal (6-8h)", "Extended (>8h)"],
)

sleep_performance_stats = student_data.groupby("sleep_category")["quiz_average"].agg(
    ["mean", "std", "count"]
)
sleep_performance_stats.columns = [
    "Average Quiz Score",
    "Standard Deviation",
    "Number of Students",
]
print("\nPerformance Statistics by Sleep Duration:")
print(sleep_performance_stats.round(2))

plt.figure(figsize=(10, 6))
sns.boxplot(data=student_data, x="sleep_category", y="quiz_average")
plt.title("Quiz Performance Distribution by Sleep Duration")
plt.xlabel("Sleep Duration Category")
plt.ylabel("Quiz Score (%)")
plt.show()

# Attendance and Project Performance
print("\n3. ATTENDANCE AND PROJECT PERFORMANCE ANALYSIS")
attendance_project_corr = student_data["attendance_rate"].corr(
    student_data["project_score"]
)
print(
    f"Correlation between attendance and project scores: {attendance_project_corr:.2f}"
)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    student_data["attendance_rate"],
    student_data["project_score"],
    c=student_data["participation_score"],
    cmap="viridis",
)
plt.colorbar(scatter, label="Participation Level (%)")
plt.xlabel("Attendance Rate (%)")
plt.ylabel("Project Score (%)")
plt.title("Relationship: Attendance Rate vs Project Performance")
plt.show()

# Stress Impact Analysis
print("\n4. STRESS LEVEL IMPACT ANALYSIS")
performance_metrics = ["quiz_average", "project_score", "participation_score"]
stress_correlation = student_data[["stress_level"] + performance_metrics].corr()[
    "stress_level"
]
print("\nCorrelation between stress and performance metrics:")
for metric in performance_metrics:
    print(
        f"- Stress vs {metric.replace('_', ' ').title()}: {stress_correlation[metric]:.2f}"
    )

plt.figure(figsize=(10, 6))
sns.heatmap(stress_correlation.to_frame(), annot=True, cmap="RdYlBu")
plt.title("Stress Level Impact on Performance Metrics")
plt.show()

# Time Management Analysis
print("\n5. TIME MANAGEMENT ANALYSIS")
student_data["total_activity_hours"] = (
    student_data["study_hours_per_week"] + student_data["extracurricular_hours"]
)
print("\nTime allocation statistics:")
print(f"Average weekly study hours: {student_data['study_hours_per_week'].mean():.1f}")
print(
    f"Average extracurricular hours: {student_data['extracurricular_hours'].mean():.1f}"
)
print(
    f"Average total activity hours: {student_data['total_activity_hours'].mean():.1f}"
)

plt.figure(figsize=(12, 8))
plt.scatter(
    student_data["study_hours_per_week"],
    student_data["extracurricular_hours"],
    s=student_data["quiz_average"] * 2,
    alpha=0.6,
)
plt.xlabel("Weekly Study Hours")
plt.ylabel("Weekly Extracurricular Hours")
plt.title("Time Allocation Analysis (Bubble Size = Quiz Performance)")
plt.show()

# Performance Clustering Analysis
print("\n6. STUDENT PERFORMANCE CLUSTER ANALYSIS")
performance_features = ["quiz_average", "project_score", "participation_score"]
scaler = StandardScaler()
normalized_features = scaler.fit_transform(student_data[performance_features])

inertia_values = []
cluster_range = range(1, 10)
for k in cluster_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(normalized_features)
    inertia_values.append(kmeans_model.inertia_)

# Apply final clustering
optimal_clusters = 3
final_clustering = KMeans(n_clusters=optimal_clusters, random_state=42)
student_data["performance_cluster"] = final_clustering.fit_predict(normalized_features)

# Calculate cluster profiles
cluster_profiles = student_data.groupby("performance_cluster")[
    performance_features
].mean()
print("\nCluster Profiles (Average Scores):")
print(cluster_profiles.round(2))

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=student_data,
    x="quiz_average",
    y="project_score",
    hue="performance_cluster",
    palette="deep",
)
plt.title("Student Performance Clusters")
plt.show()

# Assignment Completion Patterns
print("\n7. ASSIGNMENT COMPLETION ANALYSIS")
student_data["performance_quartile"] = pd.qcut(
    student_data["quiz_average"],
    q=4,
    labels=["Bottom 25%", "Lower Mid 25%", "Upper Mid 25%", "Top 25%"],
)

completion_stats = student_data.groupby("performance_quartile")[
    "assignments_completed"
].agg(["mean", "min", "max"])
print("\nAssignment Completion Statistics by Performance Level:")
print(completion_stats.round(2))

plt.figure(figsize=(10, 6))
sns.histplot(
    data=student_data,
    x="assignments_completed",
    hue="performance_quartile",
    multiple="stack",
)
plt.title("Assignment Completion Distribution by Performance Level")
plt.show()

# Participation Analysis
print("\n8. PARTICIPATION AND ATTENDANCE ANALYSIS")
student_data["attendance_level"] = pd.qcut(
    student_data["attendance_rate"],
    q=4,
    labels=["Low", "Medium-Low", "Medium-High", "High"],
)

participation_by_attendance = student_data.groupby("attendance_level")[
    "participation_score"
].describe()
print("\nParticipation Statistics by Attendance Level:")
print(participation_by_attendance.round(2))

plt.figure(figsize=(12, 6))
sns.violinplot(data=student_data, x="attendance_level", y="participation_score")
plt.title("Participation Score Distribution by Attendance Level")
plt.xticks(rotation=45)
plt.show()

# Study Efficiency Analysis
print("\n9. STUDY EFFICIENCY ANALYSIS")
student_data["study_efficiency"] = (
    student_data["quiz_average"] / student_data["study_hours_per_week"]
)

efficiency_stats = student_data["study_efficiency"].describe()
print("\nStudy Efficiency Statistics:")
print(f"Average score per study hour: {efficiency_stats['mean']:.2f}")
print(f"Most efficient student: {efficiency_stats['max']:.2f} points per hour")
print(f"Least efficient student: {efficiency_stats['min']:.2f} points per hour")

plt.figure(figsize=(10, 6))
sns.kdeplot(data=student_data, x="study_efficiency")
plt.title("Study Efficiency Distribution")
plt.xlabel("Quiz Points per Study Hour")
plt.show()

# Top vs Bottom Performers Analysis
print("\n10. TOP VS BOTTOM PERFORMERS COMPARISON")
top_performers = student_data.nlargest(10, "quiz_average")
bottom_performers = student_data.nsmallest(10, "quiz_average")

comparison_metrics = [
    "attendance_rate",
    "study_hours_per_week",
    "assignments_completed",
    "project_score",
    "participation_score",
    "sleep_hours",
]

comparison_stats = pd.DataFrame(
    {
        "Top 10% Average": top_performers[comparison_metrics].mean(),
        "Bottom 10% Average": bottom_performers[comparison_metrics].mean(),
    }
)

print("\nTop vs Bottom Performers Comparison:")
print(comparison_stats.round(2))
print("\nKey Differences (Top - Bottom):")
differences = (
    comparison_stats["Top 10% Average"] - comparison_stats["Bottom 10% Average"]
).round(2)
for metric in comparison_metrics:
    print(f"{metric.replace('_', ' ').title()}: {differences[metric]:+.2f}")

angles = np.linspace(0, 2 * np.pi, len(comparison_metrics), endpoint=False)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="polar")

for group, style in [("Top 10%", "-"), ("Bottom 10%", "--")]:
    values = comparison_stats[f"{group} Average"].values
    values = np.concatenate((values, [values[0]]))
    angles_plot = np.concatenate((angles, [angles[0]]))
    ax.plot(angles_plot, values, style, label=group)

ax.set_xticks(angles)
ax.set_xticklabels([m.replace("_", " ").title() for m in comparison_metrics])
plt.title("Performance Profile Comparison: Top vs Bottom Students")
plt.legend(loc="upper right")
plt.show()