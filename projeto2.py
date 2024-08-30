import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from scipy.stats import normaltest


df = pd.read_csv(r'C:\Users\leand\OneDrive\Documentos\FormacaoDSA\f_projeto2\Projeto2\dataset\aug_train.csv')

## Basic commands to view info about DataFrame
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.describe(include='object'))
# print(df.describe().drop(columns=['enrollee_id', 'target']))

## To view each columns from DataFrame in graphics
# columns_list = list(df.columns.values)[3:12]  #separating (separando) columns accordingly (de acordo) with type

## Display bar graphics from selected columns
# plt.figure(figsize=(15,10)) #size of graphic
# for i in columns_list:
#     x = sns.countplot(data=df.fillna('NaN'), x = i) #method to count values from column and replace empty values to new info 'NaN' in graphic
#     plt.xlabel(None) #exclude axis name (label) 'x'
#     plt.title(i, fontsize=20) #append title with column name and size of font
#     # Adicionar anotação da contagem de valor em cima de cada barra de cada grafico
#     for p in x.patches:  #for each bar (p) in count (x) on bars from graphic
#         x.annotate(f'\n{p.get_height()}', #defining what will be displayed (height of the bar/count)
#                       (p.get_x()+0.4, p.get_height()), ha = 'center', color = 'black', size= 12) #bar location and customization parameters. get_x() to take coordinate bar 'x'. get.height to take bar height (count) 
#     plt.show()


## Display histograms and boxplots from selected columns
# plt.figure(figsize=(13,9))
# plt.subplot(221)
# sns.color_palette('hls',8)
# sns.histplot(df['city_development_index'], kde= True, color='green')
# plt.title('CDI Histogram', fontsize=20)

# plt.subplot(222)
# sns.histplot(df['training_hours'], kde= True, color='red')
# plt.title('Training Hours Histogram', fontsize=20)

# plt.subplot(223)
# sns.boxplot(df['city_development_index'], color='green')

# plt.subplot(224)
# sns.boxplot(df['training_hours'], color='red')

# plt.show()
## Analisys result: both histograms dont has a normal distribution


## Check if histograms really dont has a normal distribution with statistics calculation
# statistics = ['training_hours', 'city_development_index']

# for i in statistics:

#     stats, pval = normaltest(df[i]) #to find value p (pval)

#     #check
#     if pval > 0.05:
#         print(i, ': normal distribution')
#     else:
#         print(i, ': not normal distribution')

## Realize number changes in columns values and your type (str to float)
dfcopy = df.copy()

# dfcopy['experience'] = np.where(dfcopy['experience']== "<1", 1, dfcopy['experience'])
# dfcopy['experience'] = np.where(dfcopy['experience']== ">20", 21, dfcopy['experience'])
# dfcopy['experience'] = dfcopy['experience'].astype(float)

# print(dfcopy['experience'].value_counts())

## or

dfcopy.loc[dfcopy['experience'] == "<1", 'experience'] = 1 
dfcopy.loc[dfcopy['experience']== ">20", 'experience'] = 21
dfcopy['experience'] = dfcopy['experience'].astype(float)

dfcopy.loc[dfcopy['last_new_job'] == "never", 'last_new_job'] = 0
dfcopy.loc[dfcopy['last_new_job']== ">4", 'last_new_job'] = 5
dfcopy['last_new_job'] = dfcopy['last_new_job'].astype(float)

# print(dfcopy['experience'].value_counts())
# print(dfcopy['last_new_job'].value_counts())

## Realize Spearman correlation analysis
# print(dfcopy.info())
# print(dfcopy.drop('enrollee_id', axis=1).corr('spearman', numeric_only=True))

## Create a correlation map to view better
# plt.figure(figsize=(7,7))
# sns.heatmap(dfcopy.drop('enrollee_id', axis=1).corr('spearman', numeric_only=True), cmap='YlGnBu', annot=True)
# plt.title('Correlation Map of Numerical Variables',fontsize=15)
# plt.show()


## Create a table and analysis object columns from method WOE and IV

## Information Value, Prevision Power
## < 0.02, not must be used for predict
## 0.02 - 0.1, weak predictor
## 0.1 - 0.3, mean predictor
## 0.3 - 0.5, strong predictor
## > 0.5, too good to be true

objects = dfcopy.select_dtypes(include=['object']).columns.drop(['city','company_size'])

iv = []

for i in objects:

    df_woe_iv = (pd.crosstab(df[i], df['target'], normalize='columns')
    .assign(woe =lambda dfx: np.log(dfx[1] / dfx[0]))
    .assign(iv = lambda dfx: np.sum(dfx['woe']* (dfx[1]-dfx[0]))))

    print(df_woe_iv, '\n----------------------------')

    iv.append(df_woe_iv['iv'][0]) 

df_iv = pd.DataFrame({'Features':objects,'iv':iv}).set_index('Features').sort_values(by= 'iv')

plt.figure(figsize=(9,9))
df_iv.plot(kind='barh', title='Value Information of Categoric Variables', colormap='Accent')
for index, value in enumerate(list(round(df_iv['iv'],3))):
    plt.text((value), index, str(value))
plt.legend(loc = 'lower right')
plt.show()
#obs: pesquisar mais sobre woe e iv!!