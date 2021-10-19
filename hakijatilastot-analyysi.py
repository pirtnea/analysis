#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency


# In[ ]:


students = pd.read_csv('dummy_students.csv', dtype = {'student': str})
courses = pd.read_csv('dummy_course_records.csv', dtype = {'student': str})


# In[ ]:

print("Students: ")
print(students)


# In[ ]:

print("Courses: ")
print(courses)


# In[ ]:


# [RQ2.] How do students accepted through the DEFA project perform in their studies compared to students
# accepted through other intake mechanisms?

# RQ2.1. Credits per year
# RQ2.1.1. Credits per year per subject (cs, math, other)
# RQ2.2. GPA per year
# RQ2.2.1. GPA per year per subject (cs, math, other)


# In[ ]:


# RQ2.1. Credits per year

credit_acs = courses[['student', 'credits', 'study_year']].groupby(['student', 'study_year'], as_index = False).sum()


# In[ ]:


intake_credits = credit_acs.merge(students, on = 'student', how = 'left')


# In[ ]:

print("Credits per year: ")
print(intake_credits[['intake', 'study_year', 'credits']].groupby(['intake', 'study_year'], as_index = False).mean())


# In[ ]:


# n = number of students per group (intake/study year combination)

intake_credits_n = intake_credits[['intake', 'study_year', 'credits']].groupby(['intake', 'study_year'], as_index = False).count()
intake_credits_n.rename(columns={'credits':'n'}, inplace = True)
print("Number of students per group: ")
print(intake_credits_n)


# In[ ]:


# RQ2.1.1. Credits per year per subject (cs, math, other)

credit_acs_s = courses[['student', 'credits', 'study_year', 'subject']].groupby(['student', 'study_year', 'subject'], as_index = False).sum()


# In[ ]:


intake_credits_s = credit_acs_s.merge(students, on = 'student', how = 'left')


# In[ ]:

print("Credits per year per subject: ")
print(intake_credits_s[['intake', 'study_year', 'credits', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).mean())


# In[ ]:


# n = number of students per group (intake/study year/subject combination)

intake_credits_s_n = intake_credits_s[['intake', 'study_year', 'credits', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).count()
intake_credits_s_n.rename(columns = {'credits' : 'n'}, inplace = True)
print("Number of students per group: ")
print(intake_credits_s_n)


# In[ ]:


# RQ2.2. GPA per year

gpas = courses[['student', 'grade', 'study_year']]
gpas['grade'] = pd.to_numeric(gpas['grade'], errors = 'coerce')
gpas = gpas.groupby(['student', 'study_year'], as_index = False).mean()


# In[ ]:


intake_gpas = gpas.merge(students, on = 'student', how = 'left')


# In[ ]:

print("GPA per year: ")
print(intake_gpas[['intake', 'study_year', 'grade']].groupby(['intake', 'study_year'], as_index = False).mean())


# In[ ]:


# n = number of students per group (intake/study year combination)

intake_gpas_n = intake_gpas[['intake', 'study_year', 'grade']].groupby(['intake', 'study_year'], as_index = False).count()
intake_gpas_n.rename(columns = {'grade' : 'n'}, inplace = True)
print("Number of students per group: ")
print(intake_gpas_n)


# In[ ]:


# RQ2.2.1. GPA per year per subject (cs, math, other)

gpas_s = courses[['student', 'grade', 'study_year', 'subject']]
gpas_s['grade'] = pd.to_numeric(gpas_s['grade'], errors = 'coerce')
gpas_s = gpas_s.groupby(['student', 'study_year', 'subject'], as_index = False).mean()


# In[ ]:


intake_gpas_s = gpas_s.merge(students, on = 'student', how = 'left')


# In[ ]:

print("GPA per year per subject: ")
print(intake_gpas_s[['intake', 'study_year', 'grade', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).mean())


# In[ ]:


# n = number of students per group (intake/study year/subject combination)

intake_gpas_s_n = intake_gpas_s[['intake', 'study_year', 'grade', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).count()
intake_gpas_s_n.rename(columns = {'grade' : 'n'}, inplace = True)
print("Number of students per group: ")
print(intake_gpas_s_n)


# In[ ]:


# [RQ3.] Do the demographics of students accepted through the Open Doors differ from the general student population?

# RQ3.1. Average age by intake
# RQ3.2. Gender distribution by intake
# RQ3.3. Gender by age by intake?


# In[ ]:


# RQ3.1. Average age by intake

print("Average age by intake: ")
print(students[['intake', 'age_at_intake']].groupby('intake', as_index = False).mean())


# In[ ]:


# RQ3.2. Gender distribution by intake

students_s = students.replace({'Mies': 0, 'Nainen': 1})


# In[ ]:


students_g = students_s[['intake', 'gender']].groupby('intake', as_index = False).sum()
students_g.columns = ['intake', 'women']


# In[ ]:


students_c = students_s[['intake', 'gender']].groupby('intake', as_index = False).count()
students_c.columns = ['intake', 'count']


# In[ ]:


students_genders = students_g.merge(students_c, on = 'intake')


# In[ ]:


students_genders['men'] = students_genders['count'] - students_genders['women']
column_names = ['intake', 'women', 'men', 'count']

students_genders = students_genders.reindex(columns=column_names)
print("Gender distribution by intake: ")
print(students_genders)


# In[ ]:


# RQ3.3. Gender by age by intake

print("Gender by age and intake: ")
print(students[['intake', 'age_at_intake', 'gender']].groupby(['intake', 'gender'], as_index = False).mean())


# In[ ]:


# age_by_intake = students[['intake', 'age_at_intake']]

# varmista hakuväylien järjestys kuvaajassa, ja siisti sen jälkeen kuvaajan selitys ottamalla kommentointi pois solun viimeisestä rivistä
# labels-listan järjestystä saattaa joutua muuttamaan, jotta uudet labelit osuvat oikeisiin sisäänpääsyväylädatoihin
# g = sns.kdeplot(data = age_by_intake, x = 'age_at_intake', hue = 'intake', multiple = 'fill', bw_adjust = 0.5)
# g.set(xlim = (age_by_intake['age_at_intake'].min(), age_by_intake['age_at_intake'].max()), xlabel = 'Age at intake', ylabel = 'Distribution of ages')
# g.legend(title = 'Intake', labels = ['DEFA accept', 'Open uni.', 'DEFA tried', 'Main'])
# plt.show()

age_by_intake = students[['intake', 'age_at_intake']]

# varmista hakuväylien järjestys kuvaajassa, ja siisti sen jälkeen kuvaajan selitys ottamalla kommentointi pois solun viimeisestä rivistä
# labels-listan järjestystä saattaa joutua muuttamaan, jotta uudet labelit osuvat oikeisiin sisäänpääsyväylädatoihin
g = sns.kdeplot(data = age_by_intake, x = 'age_at_intake', hue = 'intake', multiple = 'layer', bw_adjust = 0.5)
g.set(xlim = (age_by_intake['age_at_intake'].min(), age_by_intake['age_at_intake'].max()), xlabel = 'Age at intake', ylabel = 'Distribution of ages')
g.legend(title = 'Intake', labels = ['DEFA accept', 'Open uni.', 'Main'])
plt.show()

# age_by_intake = students[['intake', 'age_at_intake']]

# varmista hakuväylien järjestys kuvaajassa, ja siisti sen jälkeen kuvaajan selitys ottamalla kommentointi pois solun viimeisestä rivistä
# labels-listan järjestystä saattaa joutua muuttamaan, jotta uudet labelit osuvat oikeisiin sisäänpääsyväylädatoihin
# g = sns.kdeplot(data = age_by_intake, x = 'age_at_intake', hue = 'intake', multiple = 'stack', bw_adjust = 0.5)
# g.set(xlim = (age_by_intake['age_at_intake'].min(), age_by_intake['age_at_intake'].max()), xlabel = 'Age at intake', ylabel = 'Distribution of ages')
# g.legend(title = 'Intake', labels = ['DEFA accept', 'Open uni.', 'DEFA tried', 'Main'])
# plt.show()

# age_by_intake = students[['intake', 'age_at_intake']]

# sns.set(rc={'figure.figsize':(10,10)})
# g = sns.countplot(data = age_by_intake, x = 'age_at_intake', hue = 'intake')
# plt.show()

# Khi^2 testi (onko tilastollisesti merkitsevää eroa siinä, mitä väylää miehet ja naiset suosivat)
# Testin outputissa toinen luku on p-value. Muiden arvojen selitykset: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html

obs = np.array([students_genders['women'], students_genders['men']])
print("Khi2: ")
print(chi2_contingency(obs, correction = False))

# In[ ]:

comparison_df = intake_credits[['intake', 'study_year', 'credits']]
grades_df = intake_gpas[['intake', 'study_year', 'grade']]

print('Credits between intakes, total')

print(kruskal(comparison_df[comparison_df['intake'] == 'defa_accept']['credits'],
        comparison_df[comparison_df['intake'] == 'main']['credits']))

print('DEFA:', np.mean(comparison_df[comparison_df['intake'] == 'defa_accept']['credits']))
print('Main:', np.mean(comparison_df[comparison_df['intake'] == 'main']['credits']))
print(mannwhitneyu(comparison_df[comparison_df['intake'] == 'defa_accept']['credits'], comparison_df[comparison_df['intake'] == 'main']['credits']))

print('GPAs between intakes, total')

print(kruskal(grades_df[grades_df['intake'] == 'defa_accept']['grade'],
        grades_df[grades_df['intake'] == 'main']['grade']))

print('DEFA:', np.mean(grades_df[grades_df['intake'] == 'defa_accept']['grade']))
print('Main:', np.mean(grades_df[grades_df['intake'] == 'main']['grade']))
print(mannwhitneyu(grades_df[grades_df['intake'] == 'defa_accept']['grade'], grades_df[grades_df['intake'] == 'main']['grade']))

print('Credits between intakes, first year')

print(kruskal(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits']))

print('DEFA:', np.mean(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits']))
print('Main:', np.mean(comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits']))
print(mannwhitneyu(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits'], comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits']))

print('GPAs between intakes, first year')

print(kruskal(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 1.0)]['grade'],
        grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 1.0)]['grade']))

print('DEFA:', np.mean(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 1.0)]['grade']))
print('Main:', np.mean(grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 1.0)]['grade']))
print(mannwhitneyu(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 1.0)]['grade'], grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 1.0)]['grade']))

print('Credits between intakes, second year')

print(kruskal(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 2.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 2.0)]['credits']))

print('DEFA:', np.mean(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 2.0)]['credits']))
print('Main:', np.mean(comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 2.0)]['credits']))
print(mannwhitneyu(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 2.0)]['credits'], comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 2.0)]['credits']))

print('GPAs between intakes, second year')

print(kruskal(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 2.0)]['grade'],
        grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 2.0)]['grade']))

print('DEFA:', np.mean(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 2.0)]['grade']))
print('Main:', np.mean(grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 2.0)]['grade']))
print(mannwhitneyu(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 2.0)]['grade'], grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 2.0)]['grade']))

print('Credits between intakes, third year')

print(kruskal(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 3.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 3.0)]['credits']))

print('DEFA:', np.mean(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 3.0)]['credits']))
print('Main:', np.mean(comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 3.0)]['credits']))
print(mannwhitneyu(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 3.0)]['credits'], comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 3.0)]['credits']))

print('GPAs between intakes, third year')

print(kruskal(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 3.0)]['grade'],
        grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 3.0)]['grade']))

print('DEFA:', np.mean(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 3.0)]['grade']))
print('Main:', np.mean(grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 3.0)]['grade']))
print(mannwhitneyu(grades_df[(grades_df['intake'] == 'defa_accept') & (grades_df['study_year'] == 3.0)]['grade'], grades_df[(grades_df['intake'] == 'main') & (grades_df['study_year'] == 3.0)]['grade']))
