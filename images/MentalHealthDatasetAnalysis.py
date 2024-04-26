#!/usr/bin/env python
# coding: utf-8

# In[45]:


##########################################Visualization for Data Analytics-Assessment 2#########################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px
from matplotlib.patches import Patch
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, sum, round
import time


# In[46]:


##################################################Dataset Description###########################################################

# Load the dataset into a pandas df
mHealth_df = pd.read_csv("Mental Health Dataset.csv")


# In[47]:


# Get the total number of rows in the dataset
total_rows = len(mHealth_df)
print("Total number of rows in the dataset:", total_rows)


# In[48]:


# Display basic information and the first few rows of the dataset
mHealth_df.info(), mHealth_df.head(5)


# In[49]:


############################################Dataset Prepprocessing#############################################################

# Check for duplicates in the dataset
duplicates =mHealth_df.duplicated().sum()
print("Number of duplicate rows:", duplicates)


# In[50]:


# Drop duplicate rows and create a copy of the "mHealth_df" DataFrame
mHealth_df_cleaned = mHealth_df.drop_duplicates().copy()


# In[51]:


# Verify that the duplicates have been dropped
duplicates =mHealth_df_cleaned.duplicated().sum()
print("Number of duplicate rows:", duplicates)


# In[52]:


# Count the number of null values in each column and display them
null_counts = mHealth_df_cleaned.isnull().sum()
print("Number of null values in each column:")
print(null_counts)


# In[53]:


# Create a summary statistics table for the 'self_employed' column from the DataFrame 'mHealth_df'
distribution_assessment =mHealth_df_cleaned[["self_employed"]].describe()
print(distribution_assessment)


# In[54]:


# Fill missing values with the mode
self_employed_mode = mHealth_df_cleaned['self_employed'].mode()[0]
mHealth_df_cleaned.loc[:, 'self_employed'] = mHealth_df_cleaned['self_employed'].fillna(self_employed_mode)


# In[55]:


# Verifying the dataset for null values
null_counts = mHealth_df_cleaned.isnull().sum()
print("Number of null values in each column:")
print(null_counts)


# In[56]:


# Retrieve unique categories present in the columns
mHealth_df_cleaned["Growing_Stress"].unique()


# In[57]:


# Convert 'Timestamp' from string to date-time format
mHealth_df_cleaned['Timestamp'] = pd.to_datetime(mHealth_df['Timestamp'])

mHealth_df_cleaned['Day'] = mHealth_df_cleaned['Timestamp'].dt.day_name()


# In[58]:


# Find the earliest and latest timestamps
start_period = mHealth_df_cleaned['Timestamp'].min()
end_period = mHealth_df_cleaned['Timestamp'].max()

print("The dataset covers the period from", start_period, "to", end_period)


# In[59]:


# Define the seasons based on month
def get_season(month):
    if month in (3, 4, 5):
        return 'Spring'
    elif month in (6, 7, 8):
        return 'Summer'
    elif month in (9, 10, 11):
        return 'Autumn'
    else:
        return 'Winter'


# In[60]:


# Step 4: Create a new column 'Season' based on the month
mHealth_df_cleaned['Season'] = mHealth_df_cleaned['Timestamp'].dt.month.apply(get_season)


# In[61]:


# Display the first five rows of the cleaned DataFrame to verify the current structure and data
mHealth_df_cleaned.head()


# In[62]:


# Define columns with binary responses to encode as numeric values
columns_to_encode = ["self_employed","family_history", "treatment","Coping_Struggles"]


# In[63]:


# Loop through each column and map 'Yes' to 1 and 'No' to 0 for binary encoding
for column in columns_to_encode:
    mHealth_df_cleaned[column] = mHealth_df_cleaned[column].map({'Yes': 1, 'No': 0})


# In[64]:


# Encode 'Gender' column: map 'Male' to 1 and 'Female' to 0 for binary gender representation
mHealth_df_cleaned['Gender'] = mHealth_df_cleaned['Gender'].map({'Male': 1, 'Female': 0})


# In[65]:


# Identify columns with multiple categorical responses
columns_to_encode = ["Growing_Stress", "Changes_Habits", "Mental_Health_History", "Work_Interest", "Social_Weakness"]


# In[66]:


# Predefined mapping of categories to numeric codes
fixed_mapping = {"No": 1, "Maybe": 2, "Yes": 3}

# Loop through each column to convert categorical text data to the specified numeric codes
for column in columns_to_encode:
    # Replace the text data in the original column with predefined numeric codes
    mHealth_df_cleaned[column] = mHealth_df_cleaned[column].map(fixed_mapping)


# In[67]:


# Define the order for ordinal columns using a dictionary to establish their numeric equivalents
ordinal_mappings = {
    "Days_Indoors": {"1-14 days": 1, "15-30 days": 2, "31-60 days": 3, "More than 2 months": 4, "Go out Every day": 5 },
    "Mood_Swings": {"Low": 1, "Medium": 2, "High": 3},
    "mental_health_interview": {"No": 1, "Maybe": 2, "Yes": 3},
    "care_options": {"No": 1, "Not sure": 2, "Yes": 3}
}


# In[68]:


# Loop through each defined mapping and convert the categorical data in each column to their respective ordinal values
for column, mapping in ordinal_mappings.items():
    mHealth_df_cleaned[column] = mHealth_df_cleaned[column].map(mapping)


# In[69]:


# Display the first five rows of the cleaned DataFrame to verify the current structure and data
mHealth_df_cleaned.head()


# In[70]:


# Save the cleaned and preprocessed DataFrame to a new CSV file for future use without the index column
mHealth_df_cleaned.to_csv("Mental Health Dataset(preprocessed).csv", index=False)


# In[ ]:





# In[71]:


#######################################Mental Health Dataset Analysis-Visualisations############################################


# In[72]:


## Visualisation of the Interaction between work interest and family history on Treatment.

# Combine 'Work_Interest' and 'family_history' into a single interaction column
mHealth_df_cleaned['Work_interest & family_history Interaction'] = mHealth_df_cleaned.apply(
    lambda x: f"{'No Interest' if x['Work_Interest'] == 1 else 'Maybe Interest' if x['Work_Interest'] == 2 else 'Interest'} & "
              f"{'Family History' if x['family_history'] == 1 else 'No Family History'}",
    axis=1)

# Aggregate 'treatment' by the interaction column and calculate mean and standard error
interaction_treatment = mHealth_df_cleaned.groupby('Work_interest & family_history Interaction')['treatment'].agg(['mean', 'sem']).reset_index()

# Sort by mean treatment rate for clearer visualization
interaction_treatment = interaction_treatment.sort_values(by='mean', ascending=False)

# Create bar chart with error bars representing the standard error
plt.figure(figsize=(12, 8))
sns.barplot(x='Work_interest & family_history Interaction', y='mean', data=interaction_treatment,
            palette=sns.diverging_palette(220, 20, n=7),
            yerr=interaction_treatment['sem']*1.96)
plt.title('Impact of Work Interest and Family History on Treatment Seeking')
plt.xlabel('Interaction')
plt.ylabel('Proportion Seeking Treatment')
plt.xticks(rotation=65)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[73]:


## Visualisation of the Impact of Family History on Mental Health Outcomes

# Group by 'family_history' to find average values of 'Coping_Struggles', 'Mood_Swings', and 'treatment'
outcome_data = mHealth_df_cleaned.groupby('family_history')[['Coping_Struggles', 'Mood_Swings', 'treatment']].mean()

# Renaming index for clarity in the plot
outcome_data.index = ['No Family History', 'Family History']

# Plot each mental health outcome as a separate bar
colors = ['purple', 'skyblue', 'gold']  # Colors for visual distinction
outcome_data.plot(kind='bar', color=colors)

# Customize plot appearance
plt.title('Impact of Family History on Mental Health Outcomes')
plt.xlabel('Family History')
plt.ylabel('Proportion')
plt.ylim(0, 1)  # Proportion values range from 0 to 1
plt.xticks(rotation=0)  # Horizontal x-axis labels
plt.legend(title='Outcomes')
plt.tight_layout()  # Fit plot neatly within the figure area
plt.show()


# In[ ]:





# In[ ]:





# In[74]:


## Visualisation of the Gender Differences in Treatment Seeking

# Calculate mean, count, and standard deviation of treatment by gender
gender_treatment = mHealth_df_cleaned.groupby('Gender')['treatment'].agg(['mean', 'count', 'std'])
# Compute standard error of the mean (SEM)
gender_treatment['sem'] = gender_treatment['std'] / np.sqrt(gender_treatment['count'])

# Rename index to more descriptive gender labels
gender_treatment.index = ['Female', 'Male']

# Create a contingency table for a chi-squared test if 'treatment' is binary (0 or 1)
contingency_table = mHealth_df_cleaned.pivot_table(index='Gender', columns='treatment', aggfunc='size')
# Conduct the chi-squared test
chi2, p_value, _, _ = chi2_contingency(contingency_table)

# Plot proportion of individuals seeking treatment by gender with standard error bars
sns.barplot(x=gender_treatment.index, y=gender_treatment['mean'], yerr=gender_treatment['sem'], palette=['pink', 'blue'])
plt.title('Gender Differences in Seeking Treatment')
plt.ylabel('Proportion Seeking Treatment')
plt.xlabel('Gender')
# Display the p-value in the plot
plt.text(1, gender_treatment['mean'].max(), f'p-value: {p_value:.4f}', ha='center')
plt.show()


# In[ ]:





# In[ ]:





# In[75]:


## Visualisation of the Days Indoors vs. Mental Health Outcomes

# Encoding 'Mental_Health_History' before grouping
mHealth_df_cleaned['Mental_Health_History'] = mHealth_df_cleaned['Mental_Health_History'].replace({"No": 1, "Maybe": 2, "Yes": 3})

# Grouping by 'Days_Indoors' and calculating the mean and standard error of selected columns
days_indoors_outcomes = mHealth_df_cleaned.groupby('Days_Indoors')[['Growing_Stress', 'treatment', 'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles']].agg(['mean', 'std'])
n_days = days_indoors_outcomes.shape[0]

# Converting 'Days_Indoors' back to the original string labels for visualization
days_labels = ["1-14 days", "15-30 days", "31-60 days", "More than 2 months", "Go out Every day"]
days_indoors_outcomes.index = days_labels

# Calculate the standard error of the mean (SEM) for error bars
for column in days_indoors_outcomes.columns.levels[0]:
    days_indoors_outcomes[(column, 'sem')] = days_indoors_outcomes[(column, 'std')] / np.sqrt(n_days)

# Define custom color palette for each characteristic
colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'lightsteelblue']

# Define the columns to plot based on your grouped DataFrame
columns_to_plot = days_indoors_outcomes.columns.levels[0]

# Plotting each column with custom color and error bars
fig, axes = plt.subplots(3, 2, figsize=(14, 18))  # Adjust subplot layout based on count
axes = axes.flatten()

for i, column in enumerate(columns_to_plot):
    ax = axes[i]  # Select the appropriate subplot
    sns.barplot(ax=ax, x=days_indoors_outcomes.index, y=days_indoors_outcomes[(column, 'mean')], 
                yerr=days_indoors_outcomes[(column, 'sem')], color=colors[i % len(colors)])
    ax.set_title(f'Average {column.replace("_", " ")} by Days Indoors')  # Changed to 'Average'
    ax.set_ylabel('Average')  # Changed from 'Proportion' to 'Average'
    ax.set_xlabel('Days Indoors')
    ax.tick_params(axis='x', rotation=45)

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), textcoords='offset points')

# Remove the empty subplot (if any) in the last row if the number of variables is odd
if len(columns_to_plot) % 2 != 0:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


# In[76]:


## Visualisation of the Global Self-Reported Mental Health History

# Group data by 'Country' and 'Mental_Health_History' and count entries in each category, filling missing data with zero
mental_health_categories = mHealth_df_cleaned.groupby(['Country', 'Mental_Health_History']).size().unstack(fill_value=0)

# Ensure all expected categories (1, 2, 3) are present in the DataFrame, adding missing columns as zero
for i in [1, 2, 3]:
    if i not in mental_health_categories.columns:
        mental_health_categories[i] = 0

# Add a 'Total' column to sum up counts across all mental health history categories for each country
mental_health_categories['Total'] = mental_health_categories.sum(axis=1)

# Calculate the percentage representation of each mental health history category and round to two decimal places for better readability
mental_health_categories['No_Percentage'] = (mental_health_categories[1] / mental_health_categories['Total'] * 100).round(2)
mental_health_categories['Maybe_Percentage'] = (mental_health_categories[2] / mental_health_categories['Total'] * 100).round(2)
mental_health_categories['Yes_Percentage'] = (mental_health_categories[3] / mental_health_categories['Total'] * 100).round(2)

# Flatten the DataFrame for easier plotting and merging with other data sources
mental_health_categories.reset_index(inplace=True)

# Visualize the data using a choropleth map to show the percentage of respondents with 'Yes' in mental health history by country
fig = px.choropleth(mental_health_categories, locations="Country",
                    locationmode='country names',
                    color="Yes_Percentage",
                    hover_name="Country",
                    hover_data={
                        "No_Count": mental_health_categories[1],
                        "Maybe_Count": mental_health_categories[2],
                        "Yes_Count": mental_health_categories[3],
                        "No_Percentage": True,
                        "Maybe_Percentage": True,
                        "Yes_Percentage": True
                    },
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Percentage of Respondents by Mental Health History Category")
fig.update_geos(projection_type="natural earth")
fig.show()


# Note: You can change the 'color' parameter to 'Maybe_Percentage' or 'No_Percentage' to view different aspects of the data.


# In[ ]:





# In[77]:


## Percentage of Treatments per season 
season_treatment_counts = mHealth_df_cleaned.groupby(['Season', 'treatment']).size().unstack(fill_value=0)

custom_order = ['Spring', 'Summer', 'Autumn', 'Winter']

season_treatment_counts = season_treatment_counts.reindex(custom_order)

season_treatment_percentages = season_treatment_counts.div(season_treatment_counts.sum(axis=1), axis=0) * 100

# Plots Bar Chart
ax = season_treatment_percentages.plot(kind='bar', stacked=False, figsize=(10, 7))
ax.set_title('Seasonal Variation in Mental Health Treatment')
ax.set_xlabel('Season')
ax.set_ylabel('Percentage (%)')
ax.legend(title='Treatment', labels=['Not Receiving Treatment', 'Receiving Treatment'])

# Adding Values to Each Section of Bars by Calculating Centre
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center')


# In[78]:


mHealth_df_cleaned.head()


# In[79]:


## Heatmap Comparing Days Indoors by Season

encoding = {1: '1-14 Days', 5: 'Go out Everyday', 4: 'More than 2 Months', 
            2: '15-30 Days', 3: '31-60 Days'}
mHealth_df_cleaned['Days_Indoors'] = mHealth_df_cleaned['Days_Indoors'].map(encoding)

# Setting Order of Variables
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
days_indoors_order = ['Go out Everyday', '1-14 Days', '15-30 Days', '31-60 Days', 'More than 2 Months']

# Create the Cross-Tabulation of the Two Sets of Variables
season_indoors_cross = pd.crosstab(mHealth_df_cleaned['Season'], mHealth_df_cleaned['Days_Indoors'])[days_indoors_order]

# Normalises tCross-Tabulation by Rows to get Percentages
season_indoors_percentage = season_indoors_cross.div(season_indoors_cross.sum(axis=1), axis=0) * 100

# Reordering the Rows According to the Season Order Given Earlier
season_indoors_percentage = season_indoors_percentage.reindex(season_order)

# Plots/Creates the Heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(season_indoors_percentage, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Percentage'})
plt.title('Time Spent Indoors per Season')
plt.xlabel('Days Indoors')
plt.ylabel('Season')


# In[80]:


mHealth_df_cleaned.head()


# In[ ]:





# In[81]:


## Percentage of Gender per Occupation

# Groups Data by 'Occupation' and 'Gender' Then Calculates Counts
occupation_gender_counts = mHealth_df_cleaned.groupby(['Occupation', 'Gender']).size().unstack()

# Calculate Percentage of each Gender per Occupation
occupation_gender_percentage = occupation_gender_counts.div(occupation_gender_counts.sum(axis=1), axis=0) * 100

# Plots the Bar Chart, Including Legend Titles
fig, ax = plt.subplots(figsize=(12, 8))
occupation_gender_percentage.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Gender Distribution per Occupation')
ax.set_xlabel('Occupation')
ax.set_ylabel('Percentage')
ax.legend(title='Gender', labels=['Female', 'Male', 'Other']) 
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Adds Annotations for Each Chunk of Bar
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    if height > 0:  # Only annotate non-zero values
        ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center')


# In[ ]:





# In[ ]:





# In[82]:


## Work Interest vs Occupation

# Maps Integer Values to Actual Variables
interest_mapping = {1: 'No', 2: 'Maybe', 3: 'Yes'}
mHealth_df_cleaned['Work_Interest'] = mHealth_df_cleaned['Work_Interest'].map(interest_mapping)

# Groups Data by 'Occupation' and 'Work_Interest' Then Calculates Counts
occupation_interest_counts = mHealth_df_cleaned.groupby(['Occupation', 'Work_Interest']).size().unstack()

# Calculate the Percentage of Occupation Interest per Occupation
occupation_interest_percentage = occupation_interest_counts.div(occupation_interest_counts.sum(axis=1), axis=0) * 100

# Plots Bar Chart
fig, ax = plt.subplots(figsize=(12, 8))
occupation_interest_percentage.plot(kind='bar', stacked=True, ax=ax)
ax.set_title("Employee's Job Interest per Sector")
ax.set_xlabel('Occupation')
ax.set_ylabel('Percentage')
ax.legend(title='Work Interest')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Annotations at Midpoint
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    if height > 0: 
        ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center')

plt.show()


# In[83]:


mHealth_df_cleaned.head()


# In[ ]:





# In[84]:


## Treatments by Country 

# Load your preprocessed dataset
visualisations = pd.read_csv("Mental Health Dataset(preprocessed).csv")

# Mapping integer codes to descriptive text
treatment_mapping = {0: 'No', 1: 'Yes'}
visualisations['treatment'] = visualisations['treatment'].map(treatment_mapping)

# Calculate percentage of treatments by country
country_treatment_counts = visualisations.groupby(['Country', 'treatment']).size().unstack(fill_value=0)
country_treatment_percentages = country_treatment_counts.div(country_treatment_counts.sum(axis=1), axis=0) * 100

# Reset index to ensure 'Country' becomes a column for plotting
country_treatment_percentages = country_treatment_percentages.reset_index()

# Plotting the bar chart for treatments by country
plt.figure(figsize=(12, 8))
sns.barplot(data=country_treatment_percentages, x='Country', y='Yes', color='skyblue')
plt.title('Percentage of Treatments by Country')
plt.xlabel('Country')
plt.ylabel('Percentage of Yes (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:





# In[85]:


## Growing Stress with Self Employment 

# Load your preprocessed dataset
visualisations = pd.read_csv("Mental Health Dataset(preprocessed).csv")

growing_stress_mapping = {2: 'Maybe', 1: 'No', 3 : 'Yes'}
visualisations['Growing_Stress'] = visualisations['Growing_Stress'].map(growing_stress_mapping)

self_employed_mapping = {0: 'No', 1: 'Yes'}
visualisations['self_employed'] = visualisations['self_employed'].map(self_employed_mapping)
self_employed_counts = visualisations.groupby(['self_employed', 'Growing_Stress']).size().unstack()

# Calculate the percentage of each work interest within each occupation
self_employed_stress_percentage = self_employed_counts.div(self_employed_counts.sum(axis=1), axis=0) * 100

# Plotting the bar chart
fig, ax = plt.subplots(figsize=(12, 8))
self_employed_stress_percentage.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Growing Stress Percentage Split Self Employed')
ax.set_xlabel('Self Employed')
ax.set_ylabel('Percentage')
plt.xticks(rotation=45)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Add annotations for each bar in the stacked bar chart
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    if height > 0:  # Only annotate non-zero values
        ax.annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center')

plt.show()


# In[ ]:





# In[86]:


## Mood Swings and Social Weakness

# Load your preprocessed dataset
visualisations = pd.read_csv("Mental Health Dataset(preprocessed).csv")

# Mapping integer codes to descriptive text
social_weakness_mapping = {2: 'Maybe', 1: 'No', 3: 'Yes'}
visualisations['Social_Weakness'] = visualisations['Social_Weakness'].map(social_weakness_mapping)

mood_swings_mapping = {1: 'Low', 2: 'Medium', 3: 'High'}
visualisations['Mood_Swings'] = visualisations['Mood_Swings'].map(mood_swings_mapping)

# Calculate percentage of treatments by Mood Swings, Social Weakness, and treatment
mood_swings_counts = visualisations.groupby(['Mood_Swings', 'Social_Weakness']).size().unstack(fill_value=0)
mood_swings_counts_social_weakness_percentages = mood_swings_counts.div(mood_swings_counts.sum(axis=1), axis=0) * 100

# Reset index to convert Mood Swings & Social Weakness to columns
mood_swings_counts_social_weakness_percentages.reset_index(inplace=True)

# Melt the dataframe to prepare for plotting
mood_swings_counts_social_weakness_melted = pd.melt(mood_swings_counts_social_weakness_percentages, id_vars=['Mood_Swings'],
                                    var_name='Social_Weakness', value_name='Percentage')

# Plotting the bar chart for treatments by Mood Swings & Social Weakness
plt.figure(figsize=(12, 8))
sns.barplot(data=mood_swings_counts_social_weakness_melted, x='Mood_Swings', y='Percentage', hue='Social_Weakness', 
            palette='coolwarm', errorbar=None)
plt.title('Percentage of Treatments by Mood Swings & Social Weakness')
plt.xlabel('Mood Swings')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.legend(title='Social Weakness', loc='upper right')
plt.show()


# In[ ]:





# In[87]:


## Coping Struggles with mental health

# Load your preprocessed dataset
visualisations = pd.read_csv("Mental Health Dataset(preprocessed).csv")

coping_struggles_mapping = {0: 'No', 1: 'Yes'}
visualisations['Coping_Struggles'] = visualisations['Coping_Struggles'].map(coping_struggles_mapping)

## Work Interest vs Occupation
# Mapping integer codes to descriptive text
treatment_mapping = {0: 'No', 1: 'Yes'}
visualisations['treatment'] = visualisations['treatment'].map(treatment_mapping)

# Calculate percentage of treatments by Coping_Struggles and treatment
coping_struggle_treatment_counts = visualisations.groupby(['Coping_Struggles', 'treatment']).size().unstack(fill_value=0)
couping_struggle_treatment_percentages = coping_struggle_treatment_counts.div(coping_struggle_treatment_counts.sum(axis=1), axis=0) * 100

# Reset index to ensure 'Coping_Struggles' becomes a column for plotting
couping_treatment_percentages = couping_struggle_treatment_percentages.reset_index()

# Plotting the bar chart for treatments by Coping_Struggles
plt.figure(figsize=(12, 8))
sns.barplot(data=couping_treatment_percentages, x='Coping_Struggles', y='Yes', color='skyblue')
plt.title('Percentage of Treatments by Coping Struggles')
plt.xlabel('Coping Struggles')
plt.ylabel('Percentage of Yes (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:





# In[ ]:





# In[88]:


###########################################Big Data Technologies-PySpark########################################################

## Gender Distribution Analysis Across Occupations Using PySpark

# Create Session for Spark
spark = SparkSession.builder \
    .appName("Detailed Data Analysis with PySpark") \
    .getOrCreate()

# Record start time
start_time = time.time()

# Load the CSV file 
df = spark.read.csv("Mental Health Dataset(preprocessed).csv", header=True, inferSchema=True)

df = df.withColumn("Gender", when(df.Gender == 0, "female").otherwise("male"))

# Check for necessary columns
if 'Occupation' in df.columns and 'Gender' in df.columns:
    # Count the total and gender-specific counts within each occupation
    occupation_stats = df.groupBy("Occupation") \
                         .pivot("Gender", ["female", "male"]) \
                         .count()

    # Add a total count column for each occupation
    occupation_stats = occupation_stats.withColumn("Total", col("female") + col("male"))

    # Calculate gender percentages
    occupation_stats = occupation_stats.withColumn("Female Percentage", round((col("female") / col("Total")) * 100, 2)) \
                                       .withColumn("Male Percentage", round((col("male") / col("Total")) * 100, 2))

    print("Gender Distribution Analysis Across Occupations\n")
    # Show Results
    occupation_stats.show()
else:
    print("The necessary columns for analysis are not present in the DataFrame.")
    
# Record end time
end_time = time.time()

# Calculate duration
duration = end_time - start_time

# Print duration
print("Job duration:", duration, "seconds")

# Closes the SparkSession when done
spark.stop()


# In[ ]:




