#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal


# In[ ]:


students = pd.read_csv('students.csv', dtype = {'student': str})
courses = pd.read_csv('course_records.csv', dtype = {'student': str})


# In[ ]:


students


# In[ ]:


courses


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


intake_credits[['intake', 'study_year', 'credits']].groupby(['intake', 'study_year'], as_index = False).mean()


# In[ ]:


# n = number of students per group (intake/study year combination)

intake_credits_n = intake_credits[['intake', 'study_year', 'credits']].groupby(['intake', 'study_year'], as_index = False).count()
intake_credits_n.rename(columns={'credits':'n'}, inplace = True)
intake_credits_n


# In[ ]:


# RQ2.1.1. Credits per year per subject (cs, math, other)

credit_acs_s = courses[['student', 'credits', 'study_year', 'subject']].groupby(['student', 'study_year', 'subject'], as_index = False).sum()


# In[ ]:


intake_credits_s = credit_acs_s.merge(students, on = 'student', how = 'left')


# In[ ]:


intake_credits_s[['intake', 'study_year', 'credits', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).mean()


# In[ ]:


# n = number of students per group (intake/study year/subject combination)

intake_credits_s_n = intake_credits_s[['intake', 'study_year', 'credits', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).count()
intake_credits_s_n.rename(columns = {'credits' : 'n'}, inplace = True)
intake_credits_s_n


# In[ ]:


# RQ2.2. GPA per year

gpas = courses[['student', 'grade', 'study_year']]
gpas['grade'] = pd.to_numeric(gpas['grade'], errors = 'coerce')
gpas = gpas.groupby(['student', 'study_year'], as_index = False).mean()


# In[ ]:


intake_gpas = gpas.merge(students, on = 'student', how = 'left')


# In[ ]:


intake_gpas[['intake', 'study_year', 'grade']].groupby(['intake', 'study_year'], as_index = False).mean()


# In[ ]:


# n = number of students per group (intake/study year combination)

intake_gpas_n = intake_gpas[['intake', 'study_year', 'grade']].groupby(['intake', 'study_year'], as_index = False).count()
intake_gpas_n.rename(columns = {'grade' : 'n'}, inplace = True)
intake_gpas_n


# In[ ]:


# RQ2.2.1. GPA per year per subject (cs, math, other)

gpas_s = courses[['student', 'grade', 'study_year', 'subject']]
gpas_s['grade'] = pd.to_numeric(gpas_s['grade'], errors = 'coerce')
gpas_s = gpas_s.groupby(['student', 'study_year', 'subject'], as_index = False).mean()


# In[ ]:


intake_gpas_s = gpas_s.merge(students, on = 'student', how = 'left')


# In[ ]:


intake_gpas_s[['intake', 'study_year', 'grade', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).mean()


# In[ ]:


# n = number of students per group (intake/study year/subject combination)

intake_gpas_s_n = intake_gpas_s[['intake', 'study_year', 'grade', 'subject']].groupby(['intake', 'study_year', 'subject'], as_index = False).count()
intake_gpas_s_n.rename(columns = {'grade' : 'n'}, inplace = True)
intake_gpas_s_n


# In[ ]:


# [RQ3.] Do the demographics of students accepted through the Open Doors differ from the general student population?

# RQ3.1. Average age by intake
# RQ3.2. Gender distribution by intake
# RQ3.3. Gender by age by intake?


# In[ ]:


# RQ3.1. Average age by intake

students[['intake', 'age_at_intake']].groupby('intake', as_index = False).mean()


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
students_genders


# In[ ]:


# RQ3.3. Gender by age by intake

students[['intake', 'age_at_intake', 'gender']].groupby(['intake', 'gender'], as_index = False).mean()


# In[ ]:


age_by_intake = students[['intake', 'age_at_intake']]

# varmista hakuväylien järjestys kuvaajassa, ja siisti sen jälkeen kuvaajan selitys ottamalla kommentointi pois solun viimeisestä rivistä
# labels-listan järjestystä saattaa joutua muuttamaan, jotta uudet labelit osuvat oikeisiin sisäänpääsyväylädatoihin
g = sns.kdeplot(data = age_by_intake, x = 'age_at_intake', hue = 'intake', multiple = 'fill', bw_adjust = 0.5)
g.set(xlim = (age_by_intake['age_at_intake'].min(), age_by_intake['age_at_intake'].max()), xlabel = 'Age at intake', ylabel = 'Distribution of ages')
# g.legend(title = 'Intake', labels = ['DEFA accept', 'Open uni.', 'DEFA tried', 'Main'])


# In[ ]:


# Example Kruskal-Wallis test
# Credits between intakes, first year

comparison_df = intake_credits[['intake', 'study_year', 'credits']]

# Aja kaikille muuttujille, joita haluat tutkia omista sisäänpääsyväylistä
# Alla ajettu ensimmäisen vuoden opintopisteille -> jos haluaa katsoa esim. toisen vuoden nopat, pitää ajaa erikseen study_year == 2.0 jne.
# Samat myös keskiarvoille
kruskal(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'ay') & (comparison_df['study_year'] == 1.0)]['credits'],
        comparison_df[(comparison_df['intake'] == 'defa_tried') & (comparison_df['study_year'] == 1.0)]['credits'])


# In[ ]:


# Example Mann-Whitney U test
# Credits between DEFA and main, first year

print('Credits')
print('DEFA:', np.mean(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits']))
print('Main:', np.mean(comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits']))
mannwhitneyu(comparison_df[(comparison_df['intake'] == 'defa_accept') & (comparison_df['study_year'] == 1.0)]['credits'], comparison_df[(comparison_df['intake'] == 'main') & (comparison_df['study_year'] == 1.0)]['credits'])

