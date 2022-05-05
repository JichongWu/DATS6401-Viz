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






