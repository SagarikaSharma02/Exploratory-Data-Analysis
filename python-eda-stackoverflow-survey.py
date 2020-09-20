#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis using Python - A Case Study
# 
# *Analyzing responses from the Stack Overflow Annual Developer Survey 2020*
# 
# 
# 

# In[1]:


get_ipython().system('pip install opendatasets --upgrade')


# In[2]:


import opendatasets as od


# In[3]:


od.download('stackoverflow-developer-survey-2020')


# Let's verify that the dataset was downloaded into the directory `stackoverflow-developer-survey-2020`, and retrieve the list of files in the dataset.

# In[84]:


import os


# In[85]:


os.listdir('stackoverflow-developer-survey-2020')


# In[6]:


import pandas as pd


# In[7]:


survey_raw_df = pd.read_csv('stackoverflow-developer-survey-2020/survey_results_public.csv')


# In[8]:


survey_raw_df


# 
# 
# Let's view the list of columns in the data frame. 

# In[86]:


survey_raw_df.columns


# It appears that short codes for questions are used as column names. 
# 
# We can refer to the schema file to see the full text of each question. The schema file contains only two columns: `Column` and `QuestionText`, so we can load it as Pandas Series with `Column` as the index and the  `QuestionText` as the value.

# In[87]:


schema_fname = 'stackoverflow-developer-survey-2020/survey_results_schema.csv'
schema_raw = pd.read_csv(schema_fname, index_col='Column').QuestionText


# In[88]:


schema_raw


# **Saving the Notebook**

# 
# We can now use `schema_raw` to retrieve the full question text for any column in `survey_raw_df`.

# In[91]:


# Select a project name
project='python-eda-stackoverflow-survey'


# In[92]:


# Install the Jovian library
get_ipython().system('pip install jovian --upgrade --quiet')


# In[93]:


import jovian


# In[94]:


jovian.commit(project=project)


# `jovian.commit` uploads the notebook to [Jovian.ml](https://jovian.ml) account.I can use this link to share my work and let anyone (including me) run my notebooks and reproduce my work.

# ## Data Preparation & Cleaning
# 
# 
# 
# Let's select a subset of columns with the relevant data for our analysis.

# In[95]:


selected_columns = [
    # Demographics
    'Country',
    'Age',
    'Gender',
    'EdLevel',
    'UndergradMajor',
    # Programming experience
    'Hobbyist',
    'Age1stCode',
    'YearsCode',
    'YearsCodePro',
    'LanguageWorkedWith',
    'LanguageDesireNextYear',
    'NEWLearn',
    'NEWStuck',
    # Employment
    'Employment',
    'DevType',
    'WorkWeekHrs',
    'JobSat',
    'JobFactors',
    'NEWOvertime',
    'NEWEdImpt'
]


# In[96]:


len(selected_columns)


# ### Let's extract a copy of the data from these columns into a new data frame `survey_df`, which we can continue to modify further without affecting the original data frame.
# 

# In[97]:


survey_df = survey_raw_df[selected_columns].copy()


# In[98]:


schema = schema_raw[selected_columns]


# Let's view some basic information about the data frame.

# In[99]:


survey_df.shape


# In[100]:


survey_df.info()


# In[101]:


schema.Age1stCode


# In[102]:


survey_df.Age1stCode.unique()


# In[103]:


survey_df['Age1stCode'] = pd.to_numeric(survey_df.Age1stCode, errors='coerce')
survey_df['YearsCode'] = pd.to_numeric(survey_df.YearsCode, errors='coerce')#converting integer to float and string to NAN
survey_df['YearsCodePro'] = pd.to_numeric(survey_df.YearsCodePro, errors='coerce')


# In[104]:


survey_df['Age1stCode']


# In[105]:


survey_df['YearsCode']


# In[106]:


survey_df['YearsCodePro']


# Let's now view some basic statistics about the the numeric columns.

# In[107]:


survey_df.describe()


# There seems to be a problem with the age column, as the minimum value is 1 and max value is 279. This is a common issues with surveys: responses may contain invalid values due to accidental or intentional errors while responding. A simple fix would be ignore the rows where the value in the age column is higher than 100 years or lower than 10 years as invalid survey responses. This can be done using the `.drop` method.

# In[108]:


survey_df.drop(survey_df[survey_df.Age < 10].index, inplace=True)
survey_df.drop(survey_df[survey_df.Age > 100].index, inplace=True)


# The same hold true for `WorkWeekHrs`. Let's ignore entries where the value for the column is higher than 140 hours. (~20 hours per day).

# In[109]:


survey_df.drop(survey_df[survey_df.WorkWeekHrs > 140].index, inplace=True)


# The gender column also allows picking multiple options, but to simplify our analysis, we'll remove values containing more than option.

# In[110]:


survey_df['Gender'].value_counts()


# In[111]:


import numpy as np


# In[112]:


survey_df.where(~(survey_df.Gender.str.contains(';', na=False)), np.nan, inplace=True)


# In[115]:


survey_df['Gender'].value_counts()


# In[116]:


survey_df


# We've now cleaned up and prepared the dataset for analysis. Let's take a look at sample of rows from the data frame.

# In[117]:


survey_df.sample(10)


# Let's save and commit our work before continuing.

# In[118]:


import jovian


# In[119]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# Before we can ask interesting questions about the survey responses, it would help to understand what the demographics i.e. country, age, gender, education level, employment level etc. of the respondents look like. It's important to explore these variable in order to understand how representative the survey is of the worldwide programming community, as a survey of this scale generally tends to have some selection bias
# 
# Let's begin by importing `matplotlib.pyplot` and `seaborn`.

# In[120]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)#Each time Matplotlib loads, 
#it defines a runtime configuration (rc) containing the default styles for every plot element you create.
#This configuration can be adjusted at any time using the plt.
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ### Country
# 
# Let's look at the number of countries from which there are responses in the survey, and plot the 10 countries with the highest number of responses.

# In[121]:


schema.Country


# In[122]:


survey_df.Country.value_counts()


# In[123]:


survey_df.Country.nunique()


# We can identify the countries with the highest number of respondents using the `value_counts` method.

# In[124]:


top_countries = survey_df.Country.value_counts().head(15)
top_countries


# We can visualize this information using a bar chart.

# In[125]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title(schema.Country)
sns.barplot(top_countries.index, top_countries);


# ### Age
# 
# The distribution of the age of respondents is another important factor to look at, and we can use a histogram to visualize it. 

# In[126]:


plt.figure(figsize=(12, 6))
plt.title(schema.Age)
plt.xlabel('Age')
plt.ylabel('Number of respondents')

plt.hist(survey_df.Age, bins=np.arange(10,80,5), color='purple');


# It appears that a large percentage of respondents are in the age range of 20-45, which is somewhat representative of the programming community in general, as a lot of young people have taken up computer as their field of study or profession in the last 20 years.
# 
# 

# ### Gender
# 
# Let's look at the distribution of responses for the Gender. It's a well known fact that women and non-binary genders are underrepresented in the programming community, so we might expect to see a skewed distribution here.

# In[127]:


schema.Gender


# In[128]:


gender_counts = survey_df.Gender.value_counts(dropna = False)
gender_counts


# A pie chart would be a good way to visualize the distribution.

# In[129]:


plt.figure(figsize=(12,6))
plt.title(schema.Gender)
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=180);


# Only about 8% of survey respondents who have answered the question identify as women or non-binary. This number is lower than the overall percentage of women & non-binary genders in the programming community - which is estimated to be around 12%. 
# 

# 
# ### Education Level
# 
# Formal education in computer science is often considered an important requirement of becoming a programmer. Let's see if this indeed the case, especially since there are many free resources & tutorials available online to learn programming. We'll use a horizontal bar plot to compare education levels of respondents.

# In[130]:


sns.countplot(y=survey_df.EdLevel)
plt.xticks(rotation=75);
plt.title(schema['EdLevel'])
plt.ylabel(None);


# It appears that well over half of the respondents hold a bachelor's or master's degree, so most programmers definitely seem to have some college education, although it's not clear from this graph alone if they hold a degree in computer science.
# 
#  

# Let's also plot undergraduate majors, but this time we'll convert the numbers into percentages, and sort by percentage values to make it easier to visualize the order.

# In[131]:


schema.UndergradMajor


# In[132]:


survey_df.UndergradMajor.count()


# In[133]:


survey_df.UndergradMajor.value_counts()


# In[134]:


undergrad_pct = survey_df.UndergradMajor.value_counts() * 100 / survey_df.UndergradMajor.count()

sns.barplot(undergrad_pct, undergrad_pct.index)

plt.title(schema.UndergradMajor)
plt.ylabel(None);
plt.xlabel('Percentage');


# It turns that 40% of programmers holding a college degree have a field of study other than computer science - which is very encouraging. This seems to suggest that while college education is helpful in general, you do not need to pursue a major in computer science to become a successful programmer.
# 
# 
# 

# ### Employment
# 
# Freelancing or contract work is a common choice among programmer, so it would be interesting to compare the breakdown between full time, part time & freelance work. Let's visualize the data from `Employment` column.

# In[135]:


schema.Employment


# In[136]:


(survey_df.Employment.value_counts(normalize=True, ascending=True)*100).plot(kind='barh', color='g')
plt.title(schema.Employment)
plt.xlabel('Percentage');


# It appears that close to 10% of respondents are employed part time or as freelancers.
# 
# 
# 
# 

# The `DevType` field contains information about the roles held by respondents. Since the question allows multiple answers, the column contains lists of values separated by `;`, which makes it a bit harder to analyze directly.

# In[137]:


schema.DevType


# In[138]:


survey_df.DevType.value_counts()


# Let's define a helper function which turns a column containing lists of values (like `survey_df.DevType`) into a data frame with one column for each possible option.

# In[139]:


def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    # Iterate over the column
    for idx, value  in col_series[col_series.notnull()].iteritems():
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as True
            result_df.at[idx, option] = True
    return result_df[options]


# In[140]:


dev_type_df = split_multicolumn(survey_df.DevType)


# In[141]:



dev_type_df


# The `dev_type_df` has one column for each option that can be selected as a response. If a responded has selected the option, the value in the column is `True`, otherwise it is false.
# 
# We can now use the column-wise totals to identify the most common roles.

# In[142]:


dev_type_totals = dev_type_df.sum().sort_values(ascending=False)
dev_type_totals


# As one might expect, the most common roles include "Developer" in the name. 
# 

# ## Asking and Answering Questions
# 
# We've already gained several insights about the respondents and the programming community in general, simply by exploring individual columns of the dataset. Let's ask some specific questions, and try to answer them using data frame operations and interesting visualizations.

# #### Q: Which were the most popular programming languages in 2020? 
# 
# To answer, this we can use the `LanguageWorkedWith` column. Similar to `DevType` respondents were allowed to choose multiple options here.

# In[143]:


survey_df.LanguageWorkedWith


# First, we'll split this column into a data frame containing a column of each languages listed in the options.

# In[144]:


languages_worked_df = split_multicolumn(survey_df.LanguageWorkedWith)


# In[145]:


languages_worked_df


# It appears that a total of 25 languages were included among the options. Let's aggregate these to identify the percentage of respondents who selected each language.

# In[146]:


languages_worked_percentages = languages_worked_df.mean().sort_values(ascending=False) * 100
languages_worked_percentages


# We can plot this information using a horizontal bar chart.

# In[147]:


plt.figure(figsize=(12, 12))
sns.barplot(languages_worked_percentages, languages_worked_percentages.index)
plt.title("Languages used in the past year");
plt.xlabel('count');


# Perhaps not surprisingly, Javascript & HTML/CSS comes out at the top as web development is one of the most sought skills today and it's also happens to be one of the easiest to get started with. SQL is necessary for working with relational databases, so it's no surprise that most programmers work with SQL on a regular basis. For other forms of development, Python seems be the popular choice, beating out Java, which was the industry standard for server & application development for over 2 decades.
# 
# 

# #### Q: Which languages are the most people interested to learn over the next year?
# 
# For this we can can use the `LanguageDesireNextYear` column, with similar processing as the previous one.

# In[148]:


languages_interested_df = split_multicolumn(survey_df.LanguageDesireNextYear)
languages_interested_percentages = languages_interested_df.mean().sort_values(ascending=False) * 100
languages_interested_percentages


# In[149]:


plt.figure(figsize=(12, 12))
sns.barplot(languages_interested_percentages, languages_interested_percentages.index)
plt.title("Languages people are intersted in learning over the next year");
plt.xlabel('count');


# Once again, it's not surprising that Python is the language most people are interested in learning - since it is an easy-to-learn general purpose programming language well suited for a variety of domains: application development, numerical computing, data analysis, machine learning, big data, cloud automation, web scraping, scripting etc. etc. We're using Python for this very analysis, so we're in good company!
# 
# 

# #### Q:  Which are the most loved languages i.e. a high percentage of people who have used the language want to continue learning & using it over the next year?
# 
# While this question may seem trick at first, it's really easy to solve using Pandas array operations. Here's what we can do:
# 
# - Create a new data frame `languages_loved_df` which contains a `True` value for a language only if the corresponding values in `languages_worked_df` and `languages_interested_df` are both `True`
# - Take the column wise sum of `languages_loved_df` and divide it by the column-wise sum of `languages_worked_df` to get the percentage of respondents who "love" the language
# - Sort the results in decreasing order and plot a horizontal bar graph

# In[150]:


languages_loved_df = languages_worked_df & languages_interested_df


# In[151]:


languages_loved_percentages = (languages_loved_df.sum() * 100/ languages_worked_df.sum()).sort_values(ascending=False)


# In[152]:


plt.figure(figsize=(12, 12))
sns.barplot(languages_loved_percentages, languages_loved_percentages.index)
plt.title("Most loved languages");
plt.xlabel('count');


# Rust has been StackOverflow's most-loved language for 4 years in a row.(https://stackoverflow.blog/2020/01/20/what-is-rust-and-why-is-it-so-popular/), followed by TypeScript which has gained a lot of popularity in the past few years as a good alternative to JavaScript for web development.
# 
# Python features at number 3, despite already being one of the most widely-used languages in world. This is testament to the fact the language has solid foundation, is really easy to learn & use, has a strong ecosystem of libraries for various and massive worldwide community of developers to enjoy using it.
# 

# #### Q: In which countries do developers work the highest number of hours per week? Consider countries with more than 250 responses only.
# 
# To answer this question, we'll need to use the `groupby` data frame method to aggregate the rows for each country. We'll also need to filter the results to only include the countries which have more than 250 respondents.

# In[153]:


countries_df = survey_df.groupby('Country')[['WorkWeekHrs']].mean().sort_values('WorkWeekHrs', ascending=False)


# In[154]:


high_response_countries_df = countries_df.loc[survey_df.Country.value_counts() > 250].head(15)


# In[155]:


high_response_countries_df


# The Asian countries like Iran, China & Israel have the highest working hours, followed by the United States. However, there isn't too much variation overall and the average working hours seem to be around 40 hours per week.
# 
# 

# #### Q: How important is it to start young to build a career in programming?
# 
# Let's create a scatter plot of `Age` vs. `YearsCodePro` (i.e. years of coding experience) to answer this question.

# In[156]:


schema.YearsCodePro


# In[157]:


sns.scatterplot('Age', 'YearsCodePro', hue='Hobbyist', data=survey_df)
plt.xlabel("Age")
plt.ylabel("Years of professional coding experience");


# You can see points all over the graph, which seems to indicate that you can **start programming professionally at any age**. Also, many people who have been coding for several decades professionally also seem to enjoy it has a hobby.
# 
# We can also view the distribution of `Age1stCode` column to see when the respondents tried programming for the first time.

# In[158]:


plt.title(schema.Age1stCode)
sns.distplot(survey_df.Age1stCode);


# As you might expect, most people seem to have had some exposure to programming before the age of 40, but there are people of all ages and walks of life who are learning to code.
# 
# 
# 

# In[159]:


import jovian


# In[160]:


jovian.commit()


# ## Inferences and Conclusions
# 
# Here's a summary of the interesting inferences drawn from the survey:
# 
# 
# 
# - The programming community is not as diverse as it can be, and although things are improving, one should take more efforts to support & encourage members of underrepresented communities - whether it is in terms of age, country, race, gender or otherwise.
# 
# - Most programmers hold a college degree, although a fairly large percentage did not have computer science as their major in college, so a computer science degree isn't compulsory for learning to code or building a career in programming.
# 
# - A significant percentage of programmers either work part time or as freelancers, and this can be a great way to break into the field, especially when you're just getting started.
# 
# - Javascript & HTML/CSS are the most used programming languages in 2020, closely followed by SQL & Python
# 
# - Python is the language most people are interested in learning - since it is an easy-to-learn general purpose programming language well suited for a variety of domains.
# 
# - Rust and TypeScript are the most "loved" languages in 2020, both of which have small but fast-growing communities. Python is a close third, despite already being a widely used language.
# 
# - Programmers around the world seems be working for around 40 hours a week on average, with slight variations by country.
# 
# - One can learn and start programming professionally at any age, and one is likely to have a long and fulfilling career if one also enjoy programming as a hobby.
# 

# ## References 
# 
# References:
# 
# - Stack Overflow Developer Survey: https://insights.stackoverflow.com/survey
# - Pandas user guide: https://pandas.pydata.org/docs/user_guide/index.html
# - Matplotlib user guide: https://matplotlib.org/3.3.1/users/index.html
# - Seaborn user guide & tutorial: https://seaborn.pydata.org/tutorial.html
# - `opendatasets` Python library: https://github.com/JovianML/opendatasets
# 
# 

# In[161]:


import jovian


# In[162]:


jovian.commit()


# In[ ]:




