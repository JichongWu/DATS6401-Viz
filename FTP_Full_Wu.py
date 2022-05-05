import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
import dash as dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 100)
np.set_printoptions(linewidth=100)

#===============================
# 1. Load the datasets
#===============================
df06 = pd.read_csv('https://drive.google.com/uc?id=1ooE61IQzZC1wegCaVs9qzvieUvftxSMJ')
print(df06.shape)
df18 = pd.read_csv('https://drive.google.com/uc?id=1xNNAtLDDcYEsdvGxtfGdvWIZXK4hmJIF')
print(df18.shape)
df19 = pd.read_csv('https://drive.google.com/uc?id=12TRc9UoKHd6HwONSYwZIVQ5bTx_BU4QC')
print(df19.shape)
df20 = pd.read_csv('https://drive.google.com/uc?id=1qew_EJmSyebM1fvqWzeV7ltTRwHIIRjI')
print(df20.shape)
df21 = pd.read_csv('https://drive.google.com/uc?id=1U2NFtF7ulmlY8etLHK8rXSZW-tG8OJT3')
print(df21.shape)

print(f'Total observation: {len(df06)+len(df18)+len(df19)+len(df20)+len(df21):,}')

# combine datasets
df_combined = pd.concat([df06, df18, df19, df20, df21])
print(df_combined.head())
print(f'Shape of dataset is: {df_combined.shape}')

#===============================
# 2. Data Pre-Processing
#===============================
# change column names from uppercase to lowercase
col_names = df_combined.columns.str.lower()
print(f'new col names are: {col_names}')
df_combined.columns = col_names

# select relevent columns
df_select_col = df_combined[['event_id','begin_date_time','year','month_name','begin_day','begin_time','state',
                             'event_type','injuries_direct','injuries_indirect','deaths_direct',
                            'deaths_indirect','damage_property','damage_crops']]

# rename some col names
df_select_col.rename(columns={'begin_date_time':'date_time','month_name':'month','begin_day':'day','begin_time':'time'},
                    inplace=True)

print(f'Shape of revised dataset is: {df_select_col.shape}')
print(df_select_col.head())

# check missing values
print(df_select_col.isnull().sum())

# drop missing values
df_no_nan = df_select_col.dropna(how='any')
print(f'Shape of revised dataset is: {df_no_nan.shape}')
# double check if still any missing values
print(df_no_nan.isnull().sum())

# covert units - K:1000, M:1,000,000
df_unit = df_no_nan.copy()

### convert damage_property col
### two entries have "K" in damage_property col, need to be fixed
# identify index
df_unit['damage_property'].loc[df_unit['damage_property']=='K']
df_unit.loc[df_unit['event_id']==5505569]
df_unit.loc[df_unit['event_id']==5521704]
# replace 'K' with 0
df_unit.loc[df_unit['event_id']==5505569, 'damage_property'] = 0
df_unit.loc[df_unit['event_id']==5521704, 'damage_property'] = 0
# check results
print(df_unit.loc[df_unit['event_id']==5505569])
print(df_unit.loc[df_unit['event_id']==5521704])

# convert damage_property col
df_unit['damage_property'] = df_unit['damage_property'].replace({'K':'*1e3', 'M':'*1e6','B':'*1e9'},regex=True).map(pd.eval).astype(float)
# check result
print(df_unit.head(500))
print(df_unit.damage_property.unique())

### convert damage_crops col
### one entry have "M", need to be fixed

# identify index
df_unit['damage_crops'].loc[df_unit['damage_crops']=='M']
# locate event_id
df_unit.loc[df_unit.index.values == 23471]
# replace 'M' with 0
df_unit.loc[df_unit['event_id']==5518627, 'damage_crops'] = 0
# check result
print(df_unit.loc[df_unit.index.values == 23471])

# convert 'damage_crops' units to numbers
df_unit['damage_crops'] = df_unit['damage_crops'].replace({'K':'*1e3', 'M':'*1e6','B':'*1e9'},regex=True).map(pd.eval).astype(float)

# check final results
print(df_unit.head(500))
print(df_unit.damage_crops.unique())

# convert float to int
df_unit.damage_property = df_unit.damage_property.astype(np.int64)
df_unit.damage_crops = df_unit.damage_crops.astype(np.int64)
# check result
print(df_unit.head(200))

# get final df
df = df_unit.copy()
# check dimension
print(df.shape)

# save the final df to local disk
df.to_csv(r'C:\Users\jwu\Documents\GW\Data Visualization\Final Project\Final_df.csv')

# for convenience, now loading the pro-processed final df from local disk
df = pd.read_csv('https://drive.google.com/uc?id=1BsVXLePWnHZgq9ncQyMU1xgrZHzkZyqG')
print(df.head(100))
print(df.shape)

# reset index
df.set_index('Unnamed: 0', inplace=True)
df.index.name = None
# check result
print(df.head())
print(df.shape)
# convert df['date_time'] to time datatype
df['date_time'] = pd.to_datetime(df['date_time'])

#===============================
# 3. Exploreatory Data Analysis (EDA)
#===============================
#*******************************************
# 3.1 Outlier Detection & Removal
#*******************************************
# boxplot to check outliers
plt.figure(figsize=(12,10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot of NOAA Natural Disaster Data 2006, 2018-2021')
plt.xlabel('Features')
plt.ylabel('Values')
plt.tight_layout()
plt.show()

# calculate IQRs
print(df.describe())

q1_injuries_direct, q3_injuries_direct = np.percentile(df['injuries_direct'], [25,75])
q1_injuries_indirect, q3_injuries_indirect = np.percentile(df['injuries_indirect'], [25,75])
q1_deaths_direct, q3_deaths_direct = np.percentile(df['deaths_direct'], [25,75])
q1_deaths_indirect, q3_deaths_indirect = np.percentile(df['deaths_indirect'], [25,75])
q1_damage_property, q3_damage_property = np.percentile(df['damage_property'], [25,75])
q1_damage_crops, q3_damage_crops = np.percentile(df['damage_crops'], [25,75])

IQR_inj_direct = q3_injuries_direct - q1_injuries_direct
IQR_inj_indirect = q3_injuries_indirect - q1_injuries_indirect
IQR_deaths_dir = q3_deaths_direct - q1_deaths_direct
IQR_deaths_indir = q3_deaths_indirect - q1_deaths_indirect
IQR_damage_property = q3_damage_property - q1_damage_property
IQR_damage_crops = q3_damage_crops - q1_damage_crops

low_outlier_inj_direct = q1_injuries_direct - 1.5*IQR_inj_direct
high_outlier_inj_direct = q3_injuries_direct + 1.5*IQR_inj_direct

low_outlier_inj_indirect = q1_injuries_indirect - 1.5*IQR_inj_indirect
high_outlier_inj_indirect = q3_injuries_indirect + 1.5*IQR_inj_indirect

low_outlier_deaths_direct = q1_deaths_direct - 1.5*IQR_deaths_dir
high_outlier_deaths_direct = q3_deaths_direct + 1.5*IQR_deaths_dir

low_outlier_deaths_indirect = q1_deaths_indirect - 1.5*IQR_deaths_indir
high_outlier_deaths_indirect = q3_deaths_indirect + 1.5*IQR_deaths_indir

low_outlier_damage_property = q1_damage_property - 1.5*IQR_damage_property
high_outlier_damage_property = q3_damage_property + 1.5*IQR_damage_property

low_outlier_damage_crops = q1_damage_crops - 1.5*IQR_damage_crops
high_outlier_damage_crops = q3_damage_crops + 1.5*IQR_damage_crops

outlier_table = {'Outlier Boundary': ['low limit','high limit'],
                 'injuries_direct':[low_outlier_inj_direct, high_outlier_inj_direct],
                 'injuries_indirect':[low_outlier_inj_indirect, high_outlier_inj_indirect],
                 'deaths_direct': [low_outlier_deaths_direct, high_outlier_deaths_direct],
                 'deaths_indirect': [low_outlier_deaths_indirect, high_outlier_deaths_indirect],
                 'damage_property': [low_outlier_damage_property, high_outlier_damage_property],
                 'damage_crops': [low_outlier_damage_crops, high_outlier_damage_crops]}
df_outlier_table = pd.DataFrame(outlier_table)
print(df_outlier_table)

# remove outliers (2 values from damage_property >= 6 billion
df.loc[df['damage_property'] > 1e9].sort_values(by=['damage_property'], ascending=False)
df.shape

df = df[(df.damage_property != 1.7e10) & (df.damage_property != 6e9)]
# check result after removing outliers
df.shape

#*******************************************
# 3.2 PCA
#*******************************************
features = df.columns.to_list()[8:]
print(features)

### PCA anaylsis
X = df[features].values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

print('Original Dim', X.shape)
print('Transformed Dim', X_PCA.shape)
print(f'Explained variance ratio {pca.explained_variance_ratio_}')

### have all 6 features
X = df[features].values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=6, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

print('Original Dim', X.shape)
print('Transformed Dim', X_PCA.shape)

print(f'Explained variance ratio {pca.explained_variance_ratio_}')

# calculate % of explained data
x=0
for i in range(5):
    x=x+pca.explained_variance_ratio_[i]
print(x)
print(f'PCA analysis shows that if reduce 1 feature, 5 features can explain {x*100:.2f}% data')

# PCA plot
plt.figure()
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)
plt.xticks(x)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.title('PCA plot of 6 numerical features')
plt.xlabel('Features')
plt.ylabel('Explained Variance Ratio')
plt.show()

# SVD Analysis and condition number
X = df[features].values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print('*'*100)
print(f'Original Data: Singular Values {d}')
print(f'Original Data: condition number {LA.cond(X)}')
print('*'*100)

#============================
# SVD Analysis and condition number on the Transformed Data
#============================
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'Transformed Data: Singular Values {d_PCA}')
print(f'Transformed Data: condition number {LA.cond(X_PCA)}')
print('*'*100)
#*******************************************
# 3.3 Normality Test
#*******************************************
# 3.3.1 histogram plot

# histogram plot
df.reset_index(inplace=True)
features = df.columns.to_list()[9:]
print(features)

# get 6 dfs that exclude 0 values
injuries_direct = df.loc[df['injuries_direct'] !=0]
injuries_indirect = df.loc[df['injuries_indirect'] !=0]
deaths_direct = df.loc[df['deaths_direct'] !=0]
deaths_indirect = df.loc[df['deaths_indirect'] !=0]
damage_property = df.loc[df['damage_property'] !=0]
damage_crops = df.loc[df['damage_crops'] !=0]

# histogram plot
plt.figure(figsize=(12, 14))

for i in range(1, 7):
    plt.subplot(3, 2, i)
    df[features[i - 1]].hist(bins=30)
    plt.title(f'Histogram Plot of {features[i - 1]}')
    plt.xlabel(features[i - 1])
    plt.ylabel('Count')

plt.suptitle('Histogram Subplots of Numerical Features (Disaster Loss)\n', fontsize=20)
plt.tight_layout()
plt.show()

# histogram plot - exclude 0 values
plt.figure(figsize=(12, 14))

for i in range(1, 7):
    plt.subplot(3, 2, i)
    df.loc[df[features[i - 1]] != 0][features[i - 1]].hist(bins=30)
    plt.title(f'Histogram Plot of {features[i - 1]}')
    plt.xlabel(features[i - 1])
    plt.ylabel('Count')

plt.suptitle('Histogram Subplots of Numerical Features (Disaster Loss) Excluding Zero Values\n', fontsize=20)
plt.tight_layout()
plt.show()

# 3.3.2 QQ plot
## include 0 values
fig = plt.figure(figsize=(12, 14))

for i in range(1, 7):
    ax = fig.add_subplot(3, 2, i)
    sm.graphics.qqplot(df[features[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {features[i - 1]}')

plt.suptitle('Q-Q Subplots of Numerical Features (Disaster Loss)\n', fontsize=20)
plt.tight_layout()
plt.show()

## exclude 0 values
fig = plt.figure(figsize=(12, 14))

for i in range(1, 7):
    ax = fig.add_subplot(3, 2, i)
    sm.graphics.qqplot(df.loc[df[features[i - 1]] != 0][features[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {features[i - 1]}')

plt.suptitle('Q-Q Subplots of Numerical Features (Disaster Loss) Excluding Zero Values\n', fontsize=20)
plt.tight_layout()
plt.show()

# 3.3.3 K-S Test
for i in range(6):
    print(f"K-S test of {features[i]}: statistics = {stats.kstest(df[features[i]], 'norm')[0]},"
          f"p-value = {stats.kstest(df[features[i]], 'norm')[1]}")
# exclude 0 values
print(f"K-S test of injuries_direct excluding zero values: statistics = {stats.kstest(injuries_direct[features[0]], 'norm')[0]},"
          f"p-value = {stats.kstest(injuries_direct[features[0]], 'norm')[1]}")
print(f"K-S test of injuries_indirect excluding zero values: statistics = {stats.kstest(injuries_indirect[features[1]], 'norm')[0]},"
          f"p-value = {stats.kstest(injuries_indirect[features[1]], 'norm')[1]}")
print(f"K-S test of deaths_direct excluding zero values: statistics = {stats.kstest(deaths_direct[features[2]], 'norm')[0]},"
          f"p-value = {stats.kstest(deaths_direct[features[2]], 'norm')[1]}")
print(f"K-S test of deaths_indirect excluding zero values: statistics = {stats.kstest(deaths_indirect[features[3]], 'norm')[0]},"
          f"p-value = {stats.kstest(deaths_indirect[features[3]], 'norm')[1]}")
print(f"K-S test of damage_property excluding zero values: statistics = {stats.kstest(damage_property[features[4]], 'norm')[0]},"
          f"p-value = {stats.kstest(damage_property[features[4]], 'norm')[1]}")
print(f"K-S test of damage_crops excluding zero values: statistics = {stats.kstest(damage_crops[features[5]], 'norm')[0]},"
          f"p-value = {stats.kstest(damage_crops[features[5]], 'norm')[1]}")

# 3.3.4 Shapiro Test
for i in range(6):
    print(f"Shapiro test of {features[i]}: statistics = {shapiro(df[features[i]])[0]}， "
          f"p-value = {shapiro(df[features[i]])[1]}")

# exclude 0 values
print(f"Shapiro test of injuries_direct excluding zero values: statistics = {shapiro(injuries_direct[features[0]])[0]}, p-value = {shapiro(injuries_direct[features[0]])[1]}")
print(f"Shapiro test of injuries_indirect excluding zero values: statistics = {shapiro(injuries_indirect[features[1]])[0]}, p-value = {shapiro(injuries_indirect[features[1]])[1]}")
print(f"Shapiro test of deaths_direct excluding zero values: statistics = {shapiro(deaths_direct[features[2]])[0]}, p-value = {shapiro(deaths_direct[features[2]])[1]}")
print(f"Shapiro test of deaths_indirect excluding zero values: statistics = {shapiro(deaths_indirect[features[3]])[0]}, p-value = {shapiro(deaths_indirect[features[3]])[1]}")
print(f"Shapiro test of damage_property excluding zero values: statistics = {shapiro(damage_property[features[4]])[0]}, p-value = {shapiro(damage_property[features[4]])[1]}")
print(f"Shapiro test of damage_crops excluding zero values: statistics = {shapiro(damage_crops[features[5]])[0]}, p-value = {shapiro(damage_crops[features[5]])[1]}")

# 3.3.5 D’Agostino’s K2 Test
for i in range(6):
    print(f"D’Agostino’s K2 Test of {features[i]}: statistics = {normaltest(df[features[i]])[0]}， "
          f"p-value = {normaltest(df[features[i]])[1]}")

# exclude 0 values
print(f"D’Agostino’s K2 Test of injuries_direct excluding zero values: statistics = {normaltest(injuries_direct[features[0]])[0]}, p-value = {normaltest(injuries_direct[features[0]])[1]}")
print(f"D’Agostino’s K2 Test of injuries_indirect excluding zero values: statistics = {normaltest(injuries_indirect[features[1]])[0]}, p-value = {normaltest(injuries_indirect[features[1]])[1]}")
print(f"D’Agostino’s K2 Test of deaths_direct excluding zero values: statistics = {normaltest(deaths_direct[features[2]])[0]}, p-value = {normaltest(deaths_direct[features[2]])[1]}")
print(f"D’Agostino’s K2 Test of deaths_indirect excluding zero values: statistics = {normaltest(deaths_indirect[features[3]])[0]}, p-value = {normaltest(deaths_indirect[features[3]])[1]}")
print(f"D’Agostino’s K2 Test of damage_property excluding zero values: statistics = {normaltest(damage_property[features[4]])[0]}, p-value = {normaltest(damage_property[features[4]])[1]}")
print(f"D’Agostino’s K2 Test of damage_crops excluding zero values: statistics = {normaltest(damage_crops[features[5]])[0]}, p-value = {normaltest(damage_crops[features[5]])[1]}")

# 3.4 Heatmap & Pearson Correlation Coefficient Matrix
# Pearson r Matrix
print(df[features].corr())
# heatmap plot
plt.figure(figsize=(12,10))
sns.heatmap(df[features].corr(), annot=True, cmap='Blues')
plt.title('Heatmap of Pearson Correlation Coefficient Matrix \nof Numerical Features (Disaster Loss)\n', fontsize=20)
plt.tight_layout
plt.show()
# 3.5 Statistics Analysis
# 3.5.1 describe
print(df.describe())

# plt.figure(figsize=(15,12))
# sns.displot(data=injuries_direct, x='injuries_direct', hue='event_type',kind='kde',multiple='stack')
# plt.title('KDE Plot of injuries_direct by Event Type')
# plt.subplots_adjust(top=.9)
# plt.tight_layout
# plt.show()

# bivariate distribution plot
plt.figure(figsize=(9,7))
sns.kdeplot(data=deaths_direct,
           x='deaths_indirect',
           y='deaths_direct',
            fill=True
           )
plt.title('Bivariate Distribution Between deaths_direct and deaths_indirect',fontsize=18)
plt.tight_layout
plt.show()

#===============================
#===============================
# 4. Data Visualization
#===============================
#===============================

# get the event list
state = df['state'].unique().tolist()
state.sort()
print(state)

# get the state list
event = df['event_type'].unique().tolist()
event.sort()
print(event)


# get 6 dfs that exclude 0 values
injuries_direct = df.loc[df['injuries_direct'] !=0]
injuries_indirect = df.loc[df['injuries_indirect'] !=0]
deaths_direct = df.loc[df['deaths_direct'] !=0]
deaths_indirect = df.loc[df['deaths_indirect'] !=0]
damage_property = df.loc[df['damage_property'] !=0]
damage_crops = df.loc[df['damage_crops'] !=0]

# the sumup datasets
df_year = df.groupby('year').sum()[features]
df_month = df.groupby('month').sum()[features]
df_state = df.groupby('state').sum()[features]
df_event = df.groupby('event_type').sum()[features]

features = ['injuries_direct','injuries_indirect','deaths_direct','deaths_indirect','damage_property','damage_crops']

# Loss counts table summary and heatmaps
# lineplot
sns.lineplot(data=df_year[features[0:4]])
plt.title('Lineplot of Injuries and Deaths')
plt.show()

# lineplot
sns.lineplot(data=df_year[features[4:]])
plt.title('Lineplot of Property and Crop Damage')
plt.show()

# Stack bar plot
fig=px.bar(df, x=df.year, y=features, color='state', title='Barplot-Stack of Total Loss by year')
fig.show(renderer='browser')

# Group bar plot
plt.figure(figsize=(18,10))
sns.countplot(data=df, x='year', hue='event_type', palette='Spectral')
plt.title('Group Bar Plot of Disaster Events Loss by Year and By Event Type')
plt.legend(loc='upper right')
plt.xticks(rotation=90)
plt.show()

# Group bar plot
plt.figure(figsize=(18,10))
sns.countplot(data=df, x='year', hue='month', palette='Spectral')
plt.legend(loc='upper right')
plt.title('Group Bar Plot of Disaster Loss by Year and by Month')
plt.xticks(rotation=90)
plt.show()

# Group bar plot
plt.figure(figsize=(18,10))
sns.countplot(data=df, x='state', hue='year', palette='Spectral')
plt.legend(loc='upper right')
plt.title('Group Bar Plot of Disaster Loss by State and by Year')
plt.xticks(rotation=90)
plt.show()

# Count plot
plt.figure(figsize=(18,10))
sns.countplot(data=df.loc[df['deaths_direct']!=0], x='deaths_direct', palette='Spectral')
plt.legend(loc='upper right')
plt.title('Count Plot of Direct Deaths')
plt.xticks(rotation=90)
plt.show()

# Count plot
plt.figure(figsize=(18,10))
sns.countplot(data=df.loc[df['injuries_direct']!=0], x='injuries_direct', palette='Spectral')
#plt.legend(loc='upper right')
plt.title('Count Plot of Direct Injuries')
plt.xticks(rotation=90)
plt.show()

# Catplot
plt.figure(figsize=(18,10))
sns.catplot(data=df.loc[df['injuries_direct']!=0], x='year', y='injuries_direct', hue='month',palette='Spectral')
#plt.legend(loc='upper right')
plt.title('Cat Plot of Direct Injuries by Year and by Month')
plt.xticks(rotation=90)
plt.show()

# pie chart
fig, ax = plt.subplots(1,1)
explode = (.03, .03,.03,.03,.03)
colors = sns.color_palette('Spectral')
ax.pie(df_year.damage_property, labels=['2006','2018','2019','2020','2021'], colors = colors, autopct = '%1.3f%%')
plt.title('Pie Chart of Property Damage (USD$) by Year')
plt.show()

# pie chart
fig, ax = plt.subplots(1,1)
colors = sns.color_palette('Spectral')
ax.pie(df_month.injuries_direct, labels=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
       colors = colors, autopct = '%1.3f%%')
plt.title('Pie Chart of Direct Injuries by Month')
plt.show()

# pie chart
fig, ax = plt.subplots(1,1)
colors = sns.color_palette('Spectral')
ax.pie(df_event.deaths_direct, labels=event,
       colors = colors, autopct = '%1.3f%%')
plt.title('Pie Chart of Direct Death by Event Type')
plt.show()

# pie chart
fig, ax = plt.subplots(1,1)
colors = sns.color_palette('Spectral')
ax.pie(df_state.deaths_indirect, labels=state,
       colors = colors, autopct = '%1.3f%%')
plt.title('Pie Chart of Indirect Death by State')
plt.show()

# displot
sns.displot(data=df_state, x=features[0], kind='kde')
plt.title('Displot of Direct Injuries, Kind="KDE"')

# displot
sns.displot(data=df_state, x=features[0], kde=True)
plt.title('Displot of Direct Injuries, kde=True')

# displot
sns.displot(data=df_state, x=features[0], y=features[3])
plt.title('Bivariate Plot between Direct Injuries and Indirect Deaths')

# Pair Plot
sns.pairplot(injuries_direct)
plt.title('Pair Plot of Direct Injuries')
plt.tight_layout()
plt.show()

# Pair Plot
sns.pairplot(injuries_direct, hue='year')
plt.title('Pair Plot of Direct Injuries by Year')
plt.show()

# heatmap - sum of loss by year
plt.figure(figsize=(12,10))
sns.heatmap(df_year[features], annot=True, cmap='Blues')
plt.title('Heatmap of Total Loss by Year', fontsize=20)
plt.show()

# heatmap - sum of loss by month
plt.figure(figsize=(12,10))
sns.heatmap(df_month[features], annot=True, cmap='Blues')
plt.title('Heatmap of Total Loss by Month', fontsize=20)
plt.show()

# heatmap - sum of loss by state
plt.figure(figsize=(12,10))
sns.heatmap(df_state[features], annot=True, cmap='Blues')
plt.title('Heatmap of Total Loss by State', fontsize=20)
plt.show()

# heatmap - sum of loss by event
plt.figure(figsize=(12,10))
sns.heatmap(df_event[features], annot=True, cmap='Blues')
plt.title('Heatmap of Total Loss by Event Type', fontsize=20)
plt.show()

# Histplot
sns.histplot(data=df_state, x='injuries_direct')
plt.title('Hisogram Plot of Direct Injuries')
plt.show()

# Scatter plot and regression line
plt.figure(figsize=(12,10))
sns.regplot(data=df, x='injuries_indirect', y='deaths_indirect')
plt.title('Scatter Plot Between deaths_indirect and injuries_indirect')
plt.show()

plt.figure(figsize=(10,12))
sns.lineplot(data=df, x='date_time',y='injuries_direct', hue='state')
plt.show()

plt.figure(figsize=(10,12))
df.plot.line(x='date_time',y='injuries_direct')
plt.xticks(rotation=90)
plt.subplots_adjust(bottom=0.5)
plt.show()


# 3.5.2 Scatter plot
plt.figure(figsize=(12,10))
sns.regplot(data=df, x='injuries_direct', y='deaths_direct')
plt.title('Scatter Plot Between deaths_direct and injuries_direct',fontsize=20)
plt.show()

plt.figure(figsize=(12,10))
sns.regplot(data=df, x='injuries_indirect', y='deaths_indirect')
plt.title('Scatter Plot Between deaths_indirect and injuries_indirect',fontsize=20)
plt.show()

# KDE plot
plt.figure(figsize=(15,10))

plt.subplot(231)
sns.kdeplot(data=injuries_direct, x='injuries_direct')
plt.title('KDE plot of injuries_direct')

plt.subplot(232)
sns.kdeplot(data=injuries_indirect, x='injuries_indirect')
plt.title('KDE plot of injuries_indirect')

plt.subplot(233)
sns.kdeplot(data=deaths_direct, x='deaths_direct')
plt.title('KDE plot of deaths_direct')

plt.subplot(234)
sns.kdeplot(data=deaths_indirect, x='deaths_indirect')
plt.title('KDE plot of deaths_indirect')

plt.subplot(235)
sns.kdeplot(data=damage_property, x='damage_property')
plt.title('KDE plot of injuries_direct')

plt.subplot(236)
sns.kdeplot(data=damage_crops, x='damage_crops')
plt.title('KDE plot of injuries_crops')

plt.suptitle('KDE Subplots of Numerical Features (Disaster Loss)\n', fontsize=20)
plt.tight_layout()
plt.show()

# KDE plots with hue
plt.figure(figsize=(15,12))
sns.kdeplot(data=injuries_direct, x='injuries_direct', hue='year', fill=True, common_norm=False, palette='crest', alpha=0.3)
plt.title('KDE Plot of injuries_direct by Year', fontsize=20)
plt.show()

plt.figure(figsize=(15,12))
sns.kdeplot(data=injuries_direct, x='injuries_direct', hue='month')
plt.title('KDE Plot of injuries_direct by Month', fontsize=20)
plt.show()

plt.figure(figsize=(15,12))
sns.kdeplot(data=injuries_direct, x='injuries_direct', hue='event_type')
plt.title('KDE Plot of injuries_direct by Event Type', fontsize=20)
plt.show()

# Bivariate Distribution plot
plt.figure(figsize=(8,8))
sns.kdeplot(data=deaths_direct,
           x='deaths_indirect',
           y='deaths_direct',
            fill=True
           )
plt.title('Bivariate Distribution Between deaths_direct and deaths_indirect')
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(12,10))
sns.regplot(data=df, x='injuries_direct', y='deaths_direct')
plt.title('Scatter Plot Between deaths_direct and injuries_direct')
plt.show()

# scatter plot
plt.figure(figsize=(12,10))
sns.regplot(data=df, x='injuries_indirect', y='deaths_indirect')
plt.title('Scatter Plot Between deaths_indirect and injuries_indirect')
plt.show()

# boxplot
sns.boxplot(data=df_year[features[0:4]], palette='Spectral')
plt.legend(loc='upper right')
plt.title('Box Plot of Injuries and Deaths')
plt.ylabel('Count (People)')
plt.xticks(rotation=90)
plt.show()

# boxplot
sns.boxplot(data=df_year[features[4:]], palette='Spectral')
plt.legend(loc='upper right')
plt.title('Box Plot of Damages')
plt.ylabel('US($))')
plt.xticks(rotation=90)
plt.show()

# area plot
plt.figure(figsize=(18,10))
plt.stackplot(injuries_direct.state, injuries_direct[features[0]], injuries_direct[features[1]],
              injuries_direct[features[2]],injuries_direct[features[3]], labels=['injuries_direct','injuries_indirect',
                                                                                 'deaths_direct','deaths_indirect'])
plt.legend(loc='upper right')
plt.title('Area Plot of Injuries and Deaths by State')
plt.xticks(rotation=90)
plt.show()

# Violin plot
plt.figure(figsize=(18,10))
sns.catplot(data=df.loc[df['deaths_indirect']!=0], x='month', y='deaths_indirect', hue='year',palette='Spectral', kind='violin')
#plt.legend(loc='upper right')
plt.title('Violin Plot of Indirect Deaths by Month and by Year')
plt.xticks(rotation=90)
plt.show()

#################### df_no2006 = pd.date_range(start='1/1/2018', end='1/1/2022')

# bar plot of injuries_direct, year 2006, 2018-2021
fig = px.histogram(df, x='date_time',y='injuries_direct', color='event_type',
                   title='Bar Plot of injuries_direct by event_type, 2006,2018-2021')
fig.show(renderer = 'browser')

# bar plot of injuries_direct, year 2018-2021
fig = px.histogram(df.loc[df['year']!=2006], x='date_time',y='injuries_direct',
                   color='event_type',
                   title='Bar Plot of injuries_direct by event_type, 2018-2021')
fig.show(renderer = 'browser')

# bar plot of direct injuries and deaths, 2018-2021
fig = px.histogram(df.loc[df['year']!=2006], x='date_time',
                   y=['injuries_direct','deaths_direct'],
                   title='Bar Plot of Direct Injuries and Deaths, 2018-2021')
fig.show(renderer = 'browser')


fig = px.histogram(df, x='date_time', y=features,
                   title='Bar Plot of All Loss, 2006, 2018-2021')
fig.show(renderer = 'browser')









#*******************************************
#*******************************************
# 4.2 Dashboard
#*******************************************
#*******************************************
import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
import dash as dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import datetime
#*******************************************
# df prep
# for convenience, now loading the pro-processed final df from local disk
df = pd.read_csv('https://drive.google.com/uc?id=1BsVXLePWnHZgq9ncQyMU1xgrZHzkZyqG')
# reset index
df.set_index('Unnamed: 0', inplace=True)
df.index.name = None
# convert df['date_time'] to time datatype
df['date_time'] = pd.to_datetime(df['date_time'])
# remove 2 outliers
df_no_outlier = df[(df.damage_property != 1.7e10) & (df.damage_property != 6e9)]

df_no_outlier.reset_index(inplace=True)
features = ['injuries_direct','injuries_indirect','deaths_direct','deaths_indirect','damage_property','damage_crops']

# get 6 dfs that exclude 0 values
injuries_direct = df_no_outlier.loc[df_no_outlier['injuries_direct'] != 0]
injuries_indirect = df_no_outlier.loc[df_no_outlier['injuries_indirect'] != 0]
deaths_direct = df_no_outlier.loc[df_no_outlier['deaths_direct'] != 0]
deaths_indirect = df_no_outlier.loc[df_no_outlier['deaths_indirect'] != 0]
damage_property = df_no_outlier.loc[df_no_outlier['damage_property'] != 0]
damage_crops = df_no_outlier.loc[df_no_outlier['damage_crops'] != 0]

# get the state list
state_list = df_no_outlier['state'].unique().tolist()
state_list.sort()

# get the event list
event_list = df_no_outlier['event_type'].unique().tolist()
event_list.sort()

# QQ-plot subplots
plt.style.use('seaborn-deep')
fig_normal_qq = plt.figure()

for i in range(1, 7):
    ax = fig_normal_qq.add_subplot(3, 2, i)
    sm.graphics.qqplot(df.loc[df[features[i - 1]] != 0][features[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {features[i - 1]}')
    plt.tight_layout()

plotly_fig = mpl_to_plotly(fig_normal_qq)

# pearson heatmap
# fig_pearson = sns.heatmap(df_no_outlier[features].corr(), annot=True, cmap='Blues').set(title='Heatmap of Pearson Correlation Coefficient Matrix \nof Numerical Features (Disaster Loss)')
# plotly_fig_pearson = mpl_to_plotly(fig_pearson)

fig_pearson = px.imshow(df_no_outlier[features].corr(),
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title='Heatmap of Pearson Correlation Coefficient Matrix of Numerical Features (Disaster Loss)',
                        width=800,
                        height=800
                        )

# for the Plots Tab
df_year_full = df.groupby('year').sum()
df_month_full = df.groupby('month').sum()
df_state_full = df.groupby('state').sum()
df_event_full = df.groupby('event_type').sum()

df_year = df.groupby('year').sum()[features]
df_month = df.groupby('month').sum()[features]
df_state = df.groupby('state').sum()[features]
df_event = df.groupby('event_type').sum()[features]

# stack bar
# fig=px.bar(df, x=df.year, y=features, color='state', title='Barplot-Stack of Total Loss by year')
# fig.show(renderer='browser')

# group bar
# fig = px.bar(df, x=df.month, y=features[5], color='year', barmode='stack',title=f'Barplot-Group of Total Loss by year')
# fig.show(renderer='browser')

#*******************************************

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('my-app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    # header box
    html.Div([
    html.P(style={'fontSize':50, 'textAlign':'center'}),
    html.H1('Dashboard for DATS6401 Final Term Project'),
    html.H5('by Jichong Wu   |   May 1, 2022')],
        style = {'font-weight': 'bold','padding' : '50px', 'textAlign':'center','backgroundColor' : '#3aaab2','color':'white'}),

    html.Div([
    html.H3('Project Summary')], style = {'padding' : '40px','textAlign':'left'}),
    html.H6('Climate change has become one of the greatest challenges of the humanity in recent years. '
            'With increasing global surface temperatures, one of the impacts is the rising possibility of more severe weathers '
            'and natural disasters caused by the changing climate, such as droughts, heat, hurricanes, floods, wildfires, and '
            'increased intensity of storms. Studying the data of these disaster impacts including deaths, injuries and other loss '
            'measured by dollar costs would help us better understand its patterns and develop precautionary actions to prepare '
            'the disaster events and mitigate potential risks.\nThis dataset has 168,759 observations and 15 features after preprocessing and'
            'cleaning. It covers natural disaster events occurred in the US from year 2006, 2018, 2019, 2020, and 2021 by categorical data such as year, month, day, '
            'location, event type, and includes 6 important numerical data on event impacts - loss and damages for this study – direct injuries, '
            'indirect injuries, direct deaths, indirect deaths, property damage and crops damage. It covers 67 states and documented 54 different types of natural disasters.'),

    dcc.Tabs(id='main-tab',
             children=[
                 dcc.Tab(label='Outlier Detection', value='outlier', style={'font-size': '20px',
    'font-weight': 'bold','backgroundColor':'cyan'}),
                 dcc.Tab(label='PCA Analysis', value='pca', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'lightcoral'}),
                 dcc.Tab(label='Normality Test', value='normality', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'palegreen'}),
                 dcc.Tab(label='Corr Coef Matrix', value='pearson', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'yellow'}),
                 dcc.Tab(label='Loss Analysis ', value='loss', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'magenta'}),
                 dcc.Tab(label='All Other Plots', value='plots', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'lightpink'}),
                 dcc.Tab(label='Upload Picture', value='pic', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'turquoise'}),
                      ]),
             html.Div(id='main-layout'),
    html.Br(),
])

# *********************************************
# Tab1. outliers boxplot
fig_with_outliers = px.box(df, x=features, title='Box Plot of Numberical Features with Outliers')


# Tab 1 - outlier button layout
outlier_layout = html.Div([
        html.H2('Outlier Detection'),
        html.B('Before Removing Outliers', style={'fontSize': 20}),
        dcc.Graph(figure=fig_with_outliers),

        html.Br(),
        html.B('After Removing Outliers', style={'fontSize': 20}),
        dcc.Dropdown(id='tab1',
                     options=[
                         {'label': 'Direct Injuries', 'value': 'injuries_direct'},
                         {'label': 'Indirect Injuries', 'value': 'injuries_indirect'},
                         {'label': 'Direct Deaths', 'value': 'deaths_direct'},
                         {'label': 'Indirect Deaths', 'value': 'deaths_indirect'},
                         {'label': 'Property Damage ($)', 'value': 'damage_property'},
                         {'label': 'Crops Damage ($)', 'value': 'damage_crops'},
                     ], value='', clearable=False, multi=True),
        dcc.Graph(id='graph-tab1'),

        html.Br(),
        html.Div(id='out1'),
html.Br(),
html.Br(),
])

# Tab2 - pca button layout
pca_layout = html.Div([
                html.Br(),
                html.Br(),
                html.P('Principal Component Analysis', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
                html.Div([
                    dcc.Graph(id='graph-pca'),
                    html.P('Select the Number of Features', style={'fontSize': 20}),
                    dcc.Slider(
                        id='slider-pca', min=1, max=6, step=1, value=1,
                        marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}),
                    html.Div(id='slider-output-container', style={'fontSize': 20, 'font-weight': 'bold'})
                ], className='six columns'),

                html.Div([html.Div([
                html.P('SVD Analysis and Condition Number', style={'fontSize': 20, 'font-weight': 'bold'}),
                html.Br(),
                html.Button('Calculate PCA and SVD',
                            id='pca-button', n_clicks=0, style={'color':'white','background-color':'#3aaab2','fontSize': 20, 'font-weight': 'bold','height': '50px','width': '400px'}),
                html.Div(id='pca-container-button-out'),
                ],
                className='six columns')
    ]),
html.Br(),
html.Br(),
])

# Tab3 - normality button layout
normality_layout = html.Div([
                html.Br(),
                html.Br(),
                html.P('Normality Test - Plot', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
                html.Div([
                    dcc.RadioItems(options=['Histogram','QQ Plot'],
                                   value='',
                                   id='radio-normal-plot',
                                   style={'fontSize': 15, 'font-weight': 'bold'}),
                    html.Br(),
                    dcc.Graph(id='graph-normality'),
                    html.Br(),
                    html.P(id='out3')
                ],
                    className='six columns'),

                # html.Div([html.Div([
                #     html.P('Normality Test - Calculation', style={'fontSize': 20, 'font-weight': 'bold'}),
                #     html.Br(),
                #     html.Label('Select a Test Method'),
                #     dcc.RadioItems(options=['K-S Test','Shapiro Test',"D'Agostino's K2 Test"],value='',id='radio-normal-test',style={'fontSize': 15, 'font-weight': 'bold'}),
                #     html.Br(),
                #     html.Label('Select a Feature'),
                #     dcc.RadioItems(options=['injuries_direct',
                #                             'injuries_indirect',
                #                             'deaths_direct',
                #                             'deaths_indirect',
                #                             'damage_property',
                #                             'damage_crops'],
                #                    value='',
                #                    id='radio-normal-test-col',
                #                    style={'fontSize': 15, 'font-weight': 'bold'}),
                #     html.Br(),
                #     html.P('Normality Test Results', style={'fontSize': 20, 'font-weight': 'bold'}),
                #     html.P(id='out3-test'),
                #     ],
                #     className='six columns'),
                # html.Br(),
                # html.Br(),
                # ])
])

# Tab4 - pearson button layout
pearson_layout = html.Div([
                html.Br(),
                html.Br(),
                html.Div([
                    dcc.Graph(id='graph-pearson', figure=fig_pearson),
                    html.Br(),
                ]),
                html.Br(),
                html.Div([
                    html.Button("Download Image", id="btn_image",
                                style={'backgroundColor' : '#3aaab2','color':'white','font-size': '20px'}),
                    dcc.Download(id="download-image")
                        ]),
])

# Tab5 - loss button layout
loss_layout = html.Div([
                html.Br(),
                html.B('Select a legend (data will be grouped by)', style={'fontSize': 20}),
                dcc.RadioItems(id='tab5-radio',
                              options=['year','month','state','event_type'],inline=True),
                dcc.Graph(id='tab5-graph')], style={'textAlign':'center'}
)

# Tab6 - plots button layout
plots_layout = html.Div([
                html.Br(),
                html.B('Select a plot type'),
                dcc.Dropdown(id='tab6-drop',
                             options=['Lineplot','Barplot-stack', 'Barplot-group', 'Countplot','Catplot',
                                      'Piechart','Displot','Pairplot','KDE','Scatter plot','Boxplot',
                                      'Area plot','Violin plot'],
                             value=''),
                html.Br(),
                html.B('Select a legend (data will be grouped by)'),
                dcc.RadioItems(id='tab6-radio',
                               options=['year', 'month', 'state', 'event'], inline=True),
                html.Br(),
                html.B('Select a loss category'),
                html.Br(),
                dcc.Checklist(id='tab6-check',
                              options=['injuries_direct','injuries_indirect','deaths_direct','deaths_indirect',
                                       'damage_property','damage_crops'], inline=True),
                html.Br(),
                dcc.Graph(id='tab6-graph')
])

# Tab7 - pic button layout
pic_layout = html.Div([
    html.B('You can upload a small picture file below', style={'font-size': '15px', 'font-weight': 'bold', 'textAlign':'left'}),
    dcc.Upload(id='upload-image',
               children=html.Div([
                   'Drag and Drop or ',
                   html.A('Select Files')
               ]),
               style={
                   'width': '98%',
                   'height': '60px',
                   'lineHeight': '60px',
                   'borderWidth': '1px',
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   'margin': '10px'
               }, multiple=True
            ),
        html.Div([
        html.Div(id='output-image-upload')],style={'textAlign':'center'}),
    html.Br(),
    html.B('Please rate this APP, with a full score of 100: '),
    html.Br(),
    dcc.Input(id='tab7-input', value=''),
    html.Br(),
    html.P(id='tab7-output')
])

# =============== layout update ==================
@my_app.callback(
    Output(component_id='main-layout', component_property='children'),
    [Input(component_id='main-tab', component_property='value')]
)

def update_layout(tabs):
    if tabs == 'outlier':
        return outlier_layout
    elif tabs == 'pca':
        return pca_layout
    elif tabs == 'normality':
        return normality_layout
    elif tabs == 'pearson':
        return pearson_layout
    elif tabs == 'loss':
        return loss_layout
    elif tabs == 'plots':
        return plots_layout
    elif tabs == 'pic':
        return pic_layout


# =============== callback tab1 outlier ===================
@my_app.callback(
    Output(component_id='graph-tab1',component_property='figure'),
    [Input(component_id='tab1',component_property='value')]
)

def outlier_box(feature):
    fig = px.box(df_no_outlier, x=feature, title='Box Plot of Selected Features')
    return fig

# =============== callback tab2 pca ===================
@my_app.callback(
    Output(component_id='graph-pca', component_property='figure'),
    Output(component_id='slider-output-container', component_property='children'),
    [Input(component_id='slider-pca', component_property='value')]
)

def update_pca_graph(a):
    X = df_no_outlier[features].values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=int(a), svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)
    x0 = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)
    fig = px.line(x=x0, y=np.cumsum(pca.explained_variance_ratio_), title=f'PCA plot of {a} Numerical Features')

    num = 0
    for i in range(a):
        num = num + pca.explained_variance_ratio_[i]
    return fig, (html.Br(),
                 html.P(f'Explained variance ratio is: {pca.explained_variance_ratio_}', style={'color':'coral'}),
                 html.Br(),
                 html.P(f'Explained data % is {num*100:.2f}% data', style={'color':'coral'}),
                 html.Br(),
                 html.Br()
                 )

@my_app.callback(
    Output('pca-container-button-out', 'children'),
    [Input('pca-button', 'n_clicks')]
)

def update_svd(n_clicks):
    X = df_no_outlier[features].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)

    H = np.matmul(X.T, X)
    _, d, _ = np.linalg.svd(H)

    H_PCA = np.matmul(X_PCA.T, X_PCA)
    _, d_PCA, _ = np.linalg.svd(H_PCA)

    if n_clicks > 0:
        return (html.Br(),
                html.P(f'PCA suggested feature number after reduced dimension is: {X_PCA.shape[1]}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Original Data: Singular Values {d}',style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Original Data: condition number {LA.cond(X)}', style={'fontSize': 20, 'font-weight': 'bold', 'color':'coral'}),
                html.Br(),
                html.P(f'Transformed Data: Singular Values {d_PCA}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Transformed Data: condition number {LA.cond(X_PCA)}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.Br(),
                )

# =============== callback tab3 normality ===================
@my_app.callback(
    Output('graph-normality', 'figure'),
    Output('out3','children'),
    Input('radio-normal-plot', 'value'),
)

def update_normal_plot(value):

    if value == 'Histogram':
        fig_normal_hist = make_subplots(rows=2, cols=3, y_title='Count')
        fig_normal_hist.add_trace(go.Histogram(x=injuries_direct['injuries_direct'], name='injuries_direct'),row=1, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=injuries_indirect['injuries_indirect'], name='injuries_indirect'), row=1, col=2)
        fig_normal_hist.add_trace(go.Histogram(x=deaths_direct['deaths_direct'], name='deaths_direct'), row=1, col=3)
        fig_normal_hist.add_trace(go.Histogram(x=deaths_indirect['deaths_indirect'], name='deaths_indirect'), row=2, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=damage_property['damage_property'], name='damage_property'), row=2, col=2)
        fig_normal_hist.add_trace(go.Histogram(x=damage_crops['damage_crops'], name='damage_crops'), row=2, col=3)

        # Update xaxis properties
        fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2, row=1, col=1)
        fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2,  row=1, col=2)
        fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2,  row=1, col=3)
        fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2, row=2, col=1)
        fig_normal_hist.update_xaxes(title_text='USD($)',rangeselector_font_size=2, row=2, col=2)
        fig_normal_hist.update_xaxes(title_text='USD($)',rangeselector_font_size=2,  row=2, col=3)

        fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')

        return fig_normal_hist, 'You have chosen '+value

    if value == 'QQ Plot':
        return plotly_fig, 'You have chosen '+value

# @my_app.callback(
#     Output('out3-test','children'),
#     Input('radio-normal-test', 'value'),
#     Input('radio-normal-test-col','value')
# )
#
# def update_normal_test(radio1, radio2):
#     if radio1 == 'K-S Test':
#         return 'KSSS'
#         # if radio2 == 'injuries_direct':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(injuries_direct[features[0]], 'norm')[0]}, p-value = {stats.kstest(injuries_direct[features[0]], 'norm')[1]}"
#         # elif radio2 == 'injuries_indirect':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(injuries_indirect[features[0]], 'norm')[0]}, p-value = {stats.kstest(injuries_indirect[features[0]], 'norm')[1]}"
#         # elif radio2 == 'deaths_direct':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(deaths_direct[features[0]], 'norm')[0]}, p-value = {stats.kstest(deaths_direct[features[0]], 'norm')[1]}"
#         # elif radio2 == 'deaths_indirect':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(deaths_indirect[features[0]], 'norm')[0]}, p-value = {stats.kstest(deaths_indirect[features[0]], 'norm')[1]}"
#         # elif radio2 == 'damage_property':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(damage_property[features[0]], 'norm')[0]}, p-value = {stats.kstest(damage_property[features[0]], 'norm')[1]}"
#         # elif radio2 == 'damage_crops':
#         #     return f"K-S test of radio2: statistics = {stats.kstest(damage_crops[features[0]], 'norm')[0]}, p-value = {stats.kstest(damage_crops[features[0]], 'norm')[1]}"
#
#     if radio1 == 'Shapiro Test':
#         return 'shapiro'

# =============== callback tab4 corr, download ===================
@my_app.callback(
    Output("download-image", "data"),
    Input("btn_image", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(r'C:\Users\jwu\Documents\GW\Data Visualization\Final Project\dash_pic.png')

# =============== callback tab5 loss ===================
@my_app.callback(
    Output('tab5-graph', 'figure'),
    Input('tab5-radio', 'value'),
)
def update_tab5_graph(value):
    fig_tab6 = px.imshow(df_no_outlier.groupby(value).sum()[features],
                            text_auto=True,
                            color_continuous_scale='Blues',
                            title= f'Heatmap of Total Loss by {value}',
                            width=1500, height=1800
                            )
    return fig_tab6

# =============== callback tab6 plots ===================
@my_app.callback(
    Output('tab6-graph', 'figure'),
    Input('tab6-drop', 'value'),
    Input('tab6-radio', 'value'),
    Input('tab6-check', 'value'),
)

def update_tab6_graph(plot, legend, loss):
    if plot == 'Lineplot':
        if legend == 'year':
            fig1 = px.line(df_year, x=df_year.index, y=loss, title=f'Lineplot of Total Loss by {legend}', width=1500, height=800)
            return fig1
        if legend == 'month':
            fig2 = px.line(df_month, x=df_month.index, y=loss, title=f'Lineplot of Total Loss by {legend}', width=1500, height=800)
            return fig2
        if legend == 'state':
            fig3 = px.line(df_state, x=df_state.index, y=loss, title=f'Lineplot of Total Loss by {legend}', width=1500, height=800)
            return fig3
        if legend == 'event':
            fig4 = px.line(df_event, x=df_event.index, y=loss, title=f'Lineplot of Total Loss by {legend}', width=1500, height=800)
            return fig4

    if plot == 'Barplot-stack':
        if legend == 'year':
            fig5 = px.bar(df, x=df.year, y=loss, color='state', title=f'Barplot-Stack of Total Loss by {legend}')
            return fig5
        if legend == 'month':
            fig6 = px.bar(df, x=df.month, y=loss, color='year', title=f'Barplot-Stack of Total Loss by {legend}')
            return fig6
        if legend == 'state':
            fig7 = px.bar(df, x=df.state, y=loss, color='year', title=f'Barplot-Stack of Total Loss by {legend}')
            return fig7
        if legend == 'event':
            fig8 = px.bar(df, x=df.event, y=loss, color='year', title=f'Barplot-Stack of Total Loss by {legend}')
            return fig8

    if plot == 'Barplot-group':
        if legend == 'year':
            fig9 = px.bar(df, x=df.year, y=loss, color='state',barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig9
        if legend == 'month':
            fig10 = px.bar(df, x=df.month, y=loss, color='year', barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig10
        if legend == 'state':
            fig11 = px.bar(df, x=df.state, y=loss, color='year', barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig11
        if legend == 'event':
            fig12 = px.bar(df, x=df.event, y=loss, color='year',barmode="group", title=f'Barplot-Group of Total Loss by {legend}')
            return fig12

    if plot == 'Countplot':
        if legend == 'year':
            fig9 = px.bar(df, x=df.year, y=loss, color='state',barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig9
        if legend == 'month':
            fig10 = px.bar(df, x=df.month, y=loss, color='year', barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig10
        if legend == 'state':
            fig11 = px.bar(df, x=df.state, y=loss, color='year', barmode="group",title=f'Barplot-Group of Total Loss by {legend}')
            return fig11
        if legend == 'event':
            fig12 = px.bar(df, x=df.event, y=loss, color='year',barmode="group", title=f'Barplot-Group of Total Loss by {legend}')
            return fig12


    else:
        return dash.no_update



        # 'Lineplot', 'Barplot - stack', 'barplot - group', 'Countplot', 'Catplot',
        # 'Piechart', 'Displot', 'Pairplot', 'KDE', 'Scatter plot', 'Boxplot',
        # 'Area plot', 'Violin plot'],

# =============== callback tab7 image upload ===================
def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@my_app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))

def update_img(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@my_app.callback(
    Output('tab7-output', 'children'),
    Input('tab7-input', 'value'),
)

def rating(input):
    return (html.Br(),
            html.B(f'Your rating of this APP is: {input}', style={'color':'coral', 'font-size': '20px'})
            )

my_app.run_server(port=8889,
          host='0.0.0.0')