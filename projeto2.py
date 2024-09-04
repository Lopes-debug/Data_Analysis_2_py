import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msgn
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

# objects = dfcopy.select_dtypes(include=['object']).columns.drop(['city','company_size']) #select columns of object type

# iv = []  

# for i in objects:

#     df_woe_iv = (pd.crosstab(df[i], df['target'], normalize='columns')  #pd to create tabela cruzada com coluna df[i], coluna de referência df['target'], do tipo colunas (normalize='columns')
#     .assign(woe = lambda dfx: np.log(dfx[1] / dfx[0]))  #.assign cria nova coluna 'woe' que receberá o valor da função dfx. ao usar assign, automaticamente é fragmentado uma coluna do Df, em que será calculado a relação de proporção do valor da linha desta coluna com a classe categórica alvo 'target' (0 ou 1). E por final, np.log traz o resultado em logaritmo. isto feito para todas as linhas da coluna em questão.
#     .assign(iv = lambda dfx: np.sum(dfx['woe']* (dfx[1]-dfx[0])))) #mesma lógica anterior, alterando np.sum, multiplicação com o resultado de woe e subtrações

#     print(df_woe_iv, '\n----------------------------')

#     iv.append(df_woe_iv['iv'][0]) #adiciona à lista vazia o primeiro valor da coluna 'iv' do df_woe_iv

# df_iv = pd.DataFrame({'Features':objects,'iv':iv}).set_index('Features').sort_values(by= 'iv') #create a Df and set index as 'Features' and sorting by 'iv'

# plt.figure(figsize=(9,9))
# df_iv.plot(kind='barh', title='Value Information of Categoric Variables', colormap='Accent') #plot graphic horizontal barr, defining color and title
# for index, value in enumerate(list(round(df_iv['iv'],3))): #defining value as df_iv['iv'] but transforming in list and rounding three decimal places. and separating index with enumerate
#     plt.text((value), index, str(value)) #this line use the index, value to position text (iv value) in bar graphic
# plt.legend(loc = 'lower right') #positioned legend in lower right 
# plt.show()
#obs: pesquisar mais sobre woe e iv!!


## Data cleaning
# pd.set_option('display.max_columns', None) #code line to config print to show all columns 
dfcopy = dfcopy.drop(columns=['enrollee_id', 'city','gender', 'company_size', 'training_hours', 'last_new_job']) #remove some columns

null_values = dfcopy.isna().sum() #to check if missing values (nan) in df, sum all these values and reset index of the new columns of null values
# plt.figure(figsize=(13,9)) #size plot

# ax = sns.barplot(null_values, palette='husl') #barplot, acess first column of null values(there is only one), and color palette
# plt.xlabel('Atributes',fontsize=12) #name for x axis
# plt.ylabel('Counts of Missing Values', fontsize=15) #name for y axis
# plt.title('Plot of Missing Values', fontsize=15) #title of graphic 

# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, (p.get_height())), ha='center', color='black', size=11) #highlight value of bar in top of the bar. get_x to take position x of the bar, get.height to take value of the bar
# plt.show()


## To visualizate and check if have default in result - Map of Missing Values

null_values = pd.DataFrame(null_values)

# msgn.matrix(df[null_values[null_values[0]>0].index])  #analisando de dentro pra fora: df2[0]>0 refere à primeira coluna do df2 retorna uma série booleana True se valor > 0. df2[df2[0]>0] retorna um novo dataframe atualizado com a condição da série booleana estabelecida. e df1 será condicionado pelo novo df2 filtrado, onde permanecerá no df1 o mesmo que no df2. obs: preciso pausar e printar cada etapa para tirar dúvidas e consolidar conhecimento

# plt.show()    

## Result of columns to be cleaned:'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_type'

##Cleaning Started:
dfcopy = dfcopy.drop(['relevent_experience', 'city_development_index','target'], axis=1) #drop some columns

## major_discipline:
# sns.countplot(data=dfcopy.fillna('NaN'), x ='major_discipline', alpha=0.7, edgecolor='black') #plot the missing value counting of dfcopy['major_discipline'] 
# plt.xticks(rotation=45)  #rotate name of columns in x axle
# ax = plt.gca() #get current axes - return axes of graphs (labels(x,y), lines, title, subtitle(legenda))
# # bound = ax.get_xbound() #return the limits of x axle in tuple form (xmin, xmax) 
# for p in ax.patches:  #patches is a each bar in graph
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', color='black', size=10)
# plt.title('Missing Values of "major_discipline" Before Processing \n', fontsize = 15)
# plt.show()

## Proporção da coluna education com a coluna major (ambas tem interligação)
# print('Total de Valores Ausentes da Coluna "major_discipline:"', dfcopy['major_discipline'].isna().sum(),'\n')
# print('Relação dos Valores Ausentes e Variável "educantion_level":\n')
# print(dfcopy[dfcopy['major_discipline'].isna()]['education_level'].value_counts(dropna=False)) #filter the dfcopy['education_level'] to get only row where dfcopy['major_discipline'] is equal 'NaN'(not a number) and the end count the values, including missing values of 'education_level'

major_index = dfcopy[(dfcopy['major_discipline'].isna()) | (dfcopy['education_level']=='High School') | (dfcopy['education_level']=='Primary School') | (dfcopy['education_level'].isna())].index  #filter df with this conditions and save index of all this rows in variable to use after
# print(len(major_index))

dfcopy['major_discipline'][major_index] = 'Non Degree'  #change those rows values to 'Non Degree'
dfcopy[dfcopy['major_discipline'].isna()] = 'Other' #change the rest of missing values to 'Other' group

#To show the final graph 
# plt.figure(figsize=(10,10))
# ax = sns.countplot(data=dfcopy.fillna('NaN'), x = 'major_discipline', alpha=0.7, edgecolor='black')
# plt.xticks(rotation=45)
# for p in ax.patches:
#     plt.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', color='black', size=10)
# plt.show()


## enrolled_univesity:
# print(dfcopy.head())
# print(dfcopy['enrolled_university'].value_counts())

# sns.countplot(data = dfcopy.fillna('NaN'), x = 'enrolled_university', alpha=0.7, edgecolor='black')
# sns.despine()
# plt.xticks()
# ax = plt.gca()
# bound = ax.get_xbound()
# for p in ax.patches:
#     ax.annotate(f'\n {p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', color='black', size=10 )
# plt.title('Missing Values of "enrolled_university" Before Processing \n', fontsize=15)
# plt.show()

# print('Total Missing Values of "enrolled_university":', dfcopy['enrolled_university'].isna().sum())
# print('\n Relation of Missing Values and Column "education_level":')
# print(dfcopy[dfcopy['enrolled_university'].isna()]['education_level'].value_counts())

dfcopy['enrolled_university'].fillna('Other', inplace=True)
# print(dfcopy['enrolled_university'].value_counts())


## company_type:
# print(dfcopy.head()) #to check if is ok

# plt.figure(figsize=(50,50))
# ax = sns.countplot(data=dfcopy.fillna('NaN'), x='company_type', color='blue', edgecolor='black')
# plt.title('company_type', fontsize=15)
# plt.xticks(rotation=45)
# plt.legend('paitaon')
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}',(p.get_x()+0.4, p.get_height()), ha='center', color='black', size=10)
# plt.show()

# print(dfcopy['company_type'].value_counts(dropna=False))
# dfcopy.fillna({'company_type':'Other'},inplace=True)  #another nice ways to change nan values of a columns: df[col] = df[col].method(value) or `df.loc[row_indexer, "col"] = values` 
# print(dfcopy['company_type'].value_counts())


## education_level:
# print(dfcopy.head())

# plt.figure(figsize=(20,20))
# plt.title('education_level',fontsize=15)
# ax = sns.countplot(data=dfcopy.fillna('NaN'),x='education_level', edgecolor='blue')
# for p in ax.patches:
#     ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', fontsize=10, color='black')
# plt.show()

# print(dfcopy['education_level'].value_counts(dropna=False))
edu= dfcopy[dfcopy['education_level'].isna()].index
dfcopy['education_level'][edu]='Other'  #another nice way to change nan values of a columns: `df.loc[row_indexer, "col"] = values` 
# print(dfcopy['education_level'].value_counts())


## experience:
print(dfcopy.head())


