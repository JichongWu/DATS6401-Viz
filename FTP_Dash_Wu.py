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
    html.H3('       Project Summary')], style = {'textAlign':'left'}),
    html.H6('       Climate change has become one of the greatest challenges of the humanity in recent years. '
            'With increasing global surface temperatures, one of the impacts is the rising possibility of more severe weathers '
            'and natural disasters caused by the changing climate, such as droughts, heat, hurricanes, floods, wildfires, and '
            'increased intensity of storms. Studying the data of these disaster impacts including deaths, injuries and other loss '
            'measured by dollar costs would help us better understand its patterns and develop precautionary actions to prepare '
            'the disaster events and mitigate potential risks.', style = {'textAlign':'left'}),
    html.H6('       This dataset has 168,759 observations and 15 features after preprocessing and'
            'cleaning. It covers natural disaster events occurred in the US from year 2006, 2018, 2019, 2020, and 2021 by categorical data such as year, month, day, '
            'location, event type, and includes 6 important numerical data on event impacts - loss and damages for this study – direct injuries, '
            'indirect injuries, direct deaths, indirect deaths, property damage and crops damage. '
            'It covers 67 states and documented 54 different types of natural disasters.', style = {'textAlign':'left'}),
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

my_app.run_server(port=8888,
          host='0.0.0.0')