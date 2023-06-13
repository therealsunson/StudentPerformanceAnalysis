#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

#Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

#Research Question 1 "What is the relationship between the parental level of education and student's average scores across all subjects?"
print("-----RESEARCH QUESTION 1: What is the relationship between the parental level of education and student's average scores across all subjects?")
print('\n')

#Calculate the average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

#Convert the parental level of education to numerical values
le = LabelEncoder()
df['parental level of education'] = le.fit_transform(df['parental level of education'])

#Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='parental level of education', y='average_score', data=df)

#Calculate the correlation coefficient
correlation, _ = pearsonr(df['parental level of education'], df['average_score'])
print('Pearsons correlation: %.3f' % correlation)

#Fit a linear regression model
model = LinearRegression()
model.fit(df[['parental level of education']], df['average_score'])

#Print the coefficient of the model
print('Coefficient of determination: %.3f' % model.score(df[['parental level of education']], df['average_score']))
print('\n')

#Explaination
print('-The Pearsons correlation coefficient is the measure of the strength and direction of association between two continuous variables. The closer the score is to 0, the weaker the two variables are in correlation.')
print('-The coefficient of determination, also known as R-squared is used to measure the proportion of the variance for a dependent variable. In this case, the coefficient is near 0, which further supports the Pearsons correlation coefficient.')
print('\n')


#Research Question 2: "Do students who complete the test preparation course have better average scores than those who do not?"
print('-----RESEARCH QUESTION 2: Do students who complete the test preparation course have better average scores than those who do not?')
print('\n')

#Import additional necessary library
from scipy import stats

#Group the data by test preparation course completion status
groups = df.groupby('test preparation course')

#Calculate the t-test for the means of two independent samples
t_stat, p_value = stats.ttest_ind(groups.get_group('completed')['average_score'], groups.get_group('none')['average_score'])

print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')
print('\n')

#Explanation
print('-The T-statistic measures the difference between the two means relative to the variations in the data. The 8.39 suggests a large difference between the students who completed the test preparation course and those who did not.')
print('-The P-value is a threshold for statistical significance in social sciences. When the value is above 0.05, that means there is statistical significance. In this case, the P-value is higher than 0.05, suggesting that completing the test preparation \n course appears to be associated with better academic performance.')
print('\n')

#Research Question 3: "How does the type of lunch (standard vs free/reduced) affect the average student scores?"
print('-----RESEARCH QUESTION 3: How does the type of lunch (standard vs free/reduced) affect the average student scores?')
print('\n')

#Group the data by type of lunch
lunch_groups = df.groupby('lunch')

#Calculate the t-test for the means of two independent samples
t_stat_lunch, p_value_lunch = stats.ttest_ind(lunch_groups.get_group('standard')['average_score'], lunch_groups.get_group('free/reduced')['average_score'])

print(f'T-statistic: {t_stat_lunch}')
print(f'P-value: {p_value_lunch}')
print('\n')

#Explanation
print('-The 9.575 T-Statistic suggests a large difference between the students with standard lunch and the students with free/reduced lunch')
print('-The P-value surpasses 0.05 which suggests a statistical significance')
print('\n')

#Research Question 4: "Are there differences in average student scores based on gender?"
print('-----RESEARCH QUESTION 4: Are there differences in average student scores based on gender?')
print('\n')

#Group the data by gender
gender_groups = df.groupby('gender')

#Calculate the t-test for the means of two independent samples
t_stat_gender, p_value_gender = stats.ttest_ind(gender_groups.get_group('female')['average_score'], gender_groups.get_group('male')['average_score'])

print(f'T-statistic: {t_stat_gender}')
print(f'P-value: {p_value_gender}')
print('\n')

#Explanation
print('-The 4.169 T-statistic suggests a large difference between tests scores based on gender')
print('-The P-value surpasses 0.05 which suggests a statistical significance')
print('\n')

#Research Question 5: "Does race/ethnicity have an impact on student scores across all subjects?"
print('-----RESEARCH QUESTION 5: Does race/ethnicity have an impact on student scores across all subjects?')
print('\n')

#Import the necessary function for ANOVA
from scipy.stats import f_oneway

#Get the unique groups in 'race/ethnicity'
race_groups = df['race/ethnicity'].unique()

#Create a list of average score series for each group
score_groups = [df[df['race/ethnicity'] == group]['average_score'] for group in race_groups]

#Perform one-way ANOVA
f_stat, p_value = f_oneway(*score_groups)

print(f'F-statistic: {f_stat}')
print(f'P-value: {p_value}')
print('\n')

#Explanation
print('-The F-statistic gives a value ratio on whether the means between two populations are significantly diffferent. The F-statistic of 9.10 \n says that there is a significant difference in average scores among the groups.')
print('-The P-value surpasses 0.05 which suggests a statistical significance. This also supports the F-statistic ratio findings.')
print('\n')
print('\n')
print('\n')