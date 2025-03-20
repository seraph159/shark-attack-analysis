import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for our plots
# Use default style with grid
plt.rcParams['axes.grid'] = True
sns.set_palette('viridis')

# Display settings for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Load the data
data_path = 'GSAF5.xls.csv'
df = pd.read_csv(data_path)

# Create a directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Data Understanding
print("\n===== DATASET INFORMATION =====")
print(f"Dataset Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nColumn Info:")
df.info()

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe(include='all'))

# 2. Data Cleaning
print("\n===== DATA CLEANING =====")

# Identify columns with many unnamed columns
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
print(f"\nNumber of unnamed columns: {len(unnamed_cols)}")

# Create a cleaned dataframe with only the useful columns
useful_cols = [col for col in df.columns if 'Unnamed' not in col]
df_clean = df[useful_cols].copy()

print(f"\nCleaned dataframe shape: {df_clean.shape}")
print("\nCleaned dataframe columns:")
print(df_clean.columns.tolist())

# Check for duplicates
print(f"\nNumber of duplicate rows: {df_clean.duplicated().sum()}")

# Convert Year to numeric, handling errors
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')

# Save the cleaned dataframe to a CSV file
clean_data_path = 'shark_attacks_cleaned.csv'
df_clean.to_csv(clean_data_path, index=False)
print(f"\nCleaned data saved to: {clean_data_path}")

# 3. Exploratory Analysis
print("\n===== EXPLORATORY ANALYSIS =====")

# 3.1 Univariate Analysis
print("\n----- UNIVARIATE ANALYSIS -----")

# Numerical Variables Analysis
print("\nNumerical Variables Summary:")

# Year distribution
print("\nYear Distribution:")
df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
print(df_clean['Year'].describe())

plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Year'].dropna(), bins=30, kde=True)
plt.title('Distribution of Shark Attacks by Year')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/year_distribution.png')
plt.close()

# Age distribution
print("\nAge Distribution:")
df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
print(df_clean['Age'].describe())

plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Age'].dropna(), bins=20, kde=True)
plt.title('Age Distribution of Shark Attack Victims')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/age_distribution.png')
plt.close()

# Categorical Variables Analysis
print("\nCategorical Variables Summary:")

# Country analysis
print("\nCountry Distribution:")
country_counts = df_clean['Country'].value_counts().head(10)
print(country_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=country_counts.index, y=country_counts.values)
plt.title('Top 10 Countries with Most Shark Attacks')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/country_distribution.png')
plt.close()

# Activity analysis
print("\nActivity Distribution:")
activity_counts = df_clean['Activity'].value_counts().head(10)
print(activity_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=activity_counts.index, y=activity_counts.values)
plt.title('Top 10 Activities During Shark Attacks')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/activity_distribution.png')
plt.close()

# Species analysis
print("\nSpecies Distribution:")
species_counts = df_clean['Species '].value_counts().head(10)
print(species_counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=species_counts.index, y=species_counts.values)
plt.title('Top 10 Shark Species Involved in Attacks')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/species_distribution.png')
plt.close()

# Fatality analysis
print("\nFatality Distribution:")
fatality_counts = df_clean['Fatal (Y/N)'].value_counts()
print(fatality_counts)

plt.figure(figsize=(8, 8))
sns.countplot(x='Fatal (Y/N)', data=df_clean)
plt.title('Fatality Distribution in Shark Attacks')
plt.xlabel('Fatal (Y/N)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/fatality_distribution.png')
plt.close()

# Attack type analysis
print("\nAttack Type Distribution:")
type_counts = df_clean['Type'].value_counts()
print(type_counts)

plt.figure(figsize=(10, 6))
sns.countplot(x='Type', data=df_clean)
plt.title('Types of Shark Attacks')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/attack_type_distribution.png')
plt.close()

# 3.2 Bivariate Analysis
print("\n----- BIVARIATE ANALYSIS -----")

# Create a numeric fatality column for analysis
print("\nCreating numeric fatality column...")
df_clean['Fatal_Numeric'] = df_clean['Fatal (Y/N)'].map(lambda x: 1 if x == 'Y' else 0 if x == 'N' else None)

# Age vs Fatality
print("\nAge vs Fatality:")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fatal (Y/N)', y='Age', data=df_clean)
plt.title('Age Distribution by Fatality')
plt.xlabel('Fatal (Y/N)')
plt.ylabel('Age')
plt.tight_layout()
plt.savefig('plots/age_vs_fatality.png')
plt.close()

# Year vs Fatality (trend over time)
print("\nFatality Rate Over Time:")
# Group by year and calculate fatality rate
df_year_fatal = df_clean.groupby('Year')['Fatal_Numeric'].mean().reset_index()
df_year_fatal = df_year_fatal[df_year_fatal['Year'] >= 1900]  # Filter for more recent years

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Fatal_Numeric', data=df_year_fatal)
plt.title('Fatality Rate Over Time')
plt.xlabel('Year')
plt.ylabel('Fatality Rate')
plt.tight_layout()
plt.savefig('plots/fatality_rate_over_time.png')
plt.close()

# Activity vs Fatality
print("\nActivity vs Fatality:")
# Create a cross-tabulation
activity_fatal_crosstab = pd.crosstab(df_clean['Activity'], df_clean['Fatal (Y/N)'])
print(activity_fatal_crosstab.head(10))

# Calculate fatality rate for top activities
top_activities = df_clean['Activity'].value_counts().head(10).index
df_activity_fatal = df_clean[df_clean['Activity'].isin(top_activities)]

plt.figure(figsize=(12, 8))
sns.countplot(x='Activity', hue='Fatal (Y/N)', data=df_activity_fatal)
plt.title('Fatality by Activity')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/activity_vs_fatality.png')
plt.close()

# Country vs Fatality
print("\nCountry vs Fatality:")
# Calculate fatality rate for top countries
top_countries = df_clean['Country'].value_counts().head(10).index
df_country_fatal = df_clean[df_clean['Country'].isin(top_countries)]

plt.figure(figsize=(12, 8))
sns.countplot(x='Country', hue='Fatal (Y/N)', data=df_country_fatal)
plt.title('Fatality by Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/country_vs_fatality.png')
plt.close()

# 3.3 Multivariate Analysis
print("\n----- MULTIVARIATE ANALYSIS -----")

# Convert categorical variables to numeric for correlation analysis
print("\nPreparing data for multivariate analysis...")
df_clean['Fatal_Numeric'] = df_clean['Fatal (Y/N)'].map({'Y': 1, 'N': 0})

# Select only numeric columns for correlation analysis
numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
print(f"\nNumeric columns used for correlation: {numeric_cols}")

# Create correlation matrix
corr_matrix = df_clean[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Country, Activity, and Fatality analysis
print("\nCountry, Activity, and Fatality Analysis:")
# Group by country and activity, calculate fatality rate
group_analysis = df_clean.groupby(['Country', 'Activity'])['Fatal_Numeric'].agg(['mean', 'count']).reset_index()
group_analysis = group_analysis.rename(columns={'mean': 'fatality_rate', 'count': 'number_of_attacks'})
group_analysis = group_analysis.sort_values('number_of_attacks', ascending=False).head(20)
print(group_analysis)

# 3.4 Domain-Specific Analysis
print("\n----- DOMAIN-SPECIFIC ANALYSIS -----")

# Temporal Analysis - Attacks by Year
print("\nAttacks by Year (last 20 years):")
year_counts = df_clean['Year'].value_counts().sort_index()
recent_years = year_counts[year_counts.index >= 2000]
print(recent_years)

plt.figure(figsize=(12, 6))
recent_years.plot(kind='line')
plt.title('Shark Attacks by Year (2000 onwards)')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.savefig('plots/attacks_by_year.png')
plt.close()

# Geographical Analysis - Top 10 Countries
print("\nTop 10 Countries with Most Shark Attacks:")
country_counts = df_clean['Country'].value_counts().head(10)
print(country_counts)

plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar')
plt.title('Top 10 Countries with Most Shark Attacks')
plt.xlabel('Country')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.savefig('plots/attacks_by_country.png')
plt.close()

# Activity Analysis
print("\nTop 10 Activities During Shark Attacks:")
activity_counts = df_clean['Activity'].value_counts().head(10)
print(activity_counts)

plt.figure(figsize=(12, 6))
activity_counts.plot(kind='bar')
plt.title('Top 10 Activities During Shark Attacks')
plt.xlabel('Activity')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.savefig('plots/attacks_by_activity.png')
plt.close()

# Fatality Analysis
print("\nFatality Rate:")
fatality_counts = df_clean['Fatal (Y/N)'].value_counts()
print(fatality_counts)

plt.figure(figsize=(8, 8))
fatality_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Fatality Rate in Shark Attacks')
plt.ylabel('')
plt.tight_layout()
plt.savefig('plots/fatality_rate.png')
plt.close()

# Species Analysis
print("\nTop 10 Shark Species Involved in Attacks:")
species_counts = df_clean['Species '].value_counts().head(10)
print(species_counts)

plt.figure(figsize=(12, 6))
species_counts.plot(kind='bar')
plt.title('Top 10 Shark Species Involved in Attacks')
plt.xlabel('Species')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.savefig('plots/attacks_by_species.png')
plt.close()

# 4. Additional Analysis - Attack Types
print("\nTypes of Shark Attacks:")
type_counts = df_clean['Type'].value_counts()
print(type_counts)

plt.figure(figsize=(10, 6))
type_counts.plot(kind='bar')
plt.title('Types of Shark Attacks')
plt.xlabel('Attack Type')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.savefig('plots/attack_types.png')
plt.close()

# 5. Time of Day Analysis (if data is available)
if 'Time' in df_clean.columns:
    print("\nAttacks by Time of Day:")
    # This would require parsing the time data which might be in various formats
    # For simplicity, we'll just show the raw counts
    time_counts = df_clean['Time'].value_counts().head(10)
    print(time_counts)

# 6. Age and Gender Analysis
print("\nAge Distribution of Shark Attack Victims:")
# Convert Age to numeric, handling errors
df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
print(df_clean['Age'].describe())

plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Age'].dropna(), bins=20, kde=True)
plt.title('Age Distribution of Shark Attack Victims')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/age_distribution.png')
plt.close()

# Gender analysis
if 'Unnamed: 9' in df.columns:  # This might be the gender column based on the data preview
    df_clean['Gender'] = df['Unnamed: 9']
    print("\nGender Distribution:")
    gender_counts = df_clean['Gender'].value_counts()
    print(gender_counts)
    
    plt.figure(figsize=(8, 8))
    gender_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Gender Distribution of Shark Attack Victims')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('plots/gender_distribution.png')
    plt.close()

# 7. Correlation Analysis
print("\n===== CORRELATION ANALYSIS =====")
print("\nAnalyzing correlations between numeric variables:")

# Convert 'Fatal (Y/N)' to numeric for correlation analysis
df_clean['Fatal_Numeric'] = df_clean['Fatal (Y/N)'].map({'Y': 1, 'N': 0})

# Select only numeric columns for correlation analysis
numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
print(f"\nNumeric columns used for correlation: {numeric_cols}")

# Create correlation matrix
corr_matrix = df_clean[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# Interpret some key correlations
print("\nKey correlation insights:")
print("- Correlation between Age and Fatality (if any)")
print("- Correlation between Year and other variables (if any)")
print("- Any other significant correlations found in the data")

print("\n===== EDA COMPLETE =====")
print("Plots have been saved to the 'plots' directory.")

# Run this script to perform the EDA
if __name__ == "__main__":
    print("Exploratory Data Analysis of Shark Attack Dataset")
    print("Check the plots directory for visualizations")