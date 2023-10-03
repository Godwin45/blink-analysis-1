from flask import Flask, render_template
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import plotly.subplots as sp
import plotly.figure_factory as ff

app = Flask(__name__)


# Load the data
df = pd.read_csv('Sample Data.csv')
df_reset = pd.read_csv('Sample Data.csv')

def separate_timestamp_and_weight(df):
    df[['Timestamp', 'WEIGHT']] = df['Timestamp;WEIGHT'].str.split(';', expand=True)
    df.drop(columns=['Timestamp;WEIGHT'], inplace=True)
    return df

df = separate_timestamp_and_weight(df)

# Convert the 'Timestamp' column to datetime objects
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['WEIGHT'] = pd.to_numeric(df['WEIGHT'], errors='coerce')


def separate_timestamp_and_weight(df_reset):
    df_reset[['Timestamp', 'WEIGHT']] = df_reset['Timestamp;WEIGHT'].str.split(';', expand=True)
    df_reset.drop(columns=['Timestamp;WEIGHT'], inplace=True)
    return df_reset

df_reset = separate_timestamp_and_weight(df_reset)
df_reset['Timestamp'] = pd.to_datetime(df_reset['Timestamp'])

df_reset['WEIGHT'] = pd.to_numeric(df_reset['WEIGHT'], errors='coerce')
df_reset['Timestamp'] = df_reset['Timestamp'].astype('datetime64[ns]').view('int64')

df['DayOfWeek'] = df['Timestamp'].dt.day_name()
df['HourOfDay'] = df['Timestamp'].dt.hour
# Create Graph 1
# Create a pivot table for the heatmap
heatmap_data = df.pivot_table(index='DayOfWeek', columns='HourOfDay', values='WEIGHT', aggfunc=np.mean)

# Create the heatmap using Plotly
fig1 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='YlGnBu',
    zmin=heatmap_data.values.min(),
    zmax=heatmap_data.values.max(),
    colorbar=dict(title='Weight', ticks='outside', len=0.5, yanchor='top', y=1.2),
    hoverongaps=False,
))

# Customize the layout
fig1.update_layout(
    title='Weight Heatmap (Day of Week vs. Hour of Day)',
    xaxis_title='Hour of Day',
    yaxis_title='Day of Week',
    xaxis=dict(tickmode='array', tickvals=list(heatmap_data.columns)),
    yaxis=dict(tickmode='array', tickvals=list(heatmap_data.index)),
    showlegend=True,
    autosize=True,
    width=800,
    height=600,
    template="plotly_dark"
)

graph1 = go.Figure(fig1)

# Create Graph 2
fig2 = px.violin(df, x=df['Timestamp'].dt.day_name(), y='WEIGHT', box=True, title='Weight Distribution by Day of the Week')
fig2.update_xaxes(title='Day of the Week')
fig2.update_yaxes(title='Weight')

# Customize the layout
fig2.update_layout(template="plotly_dark")

graph2 = go.Figure(fig2)

# Create 3 Graph
days = df['Timestamp'].dt.day_name().unique()

# Calculate the number of rows and columns for subplots
num_rows = len(days) // 3 + (len(days) % 3 > 0)
num_cols = min(len(days), 3)

# Create subplots for each day of the week using Plotly
fig3 = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=days, shared_yaxes=True, vertical_spacing=0.1)

fig3.update_layout(title_text='Weight Distribution by Day of the Week', title_x=0.5, title_font_size=16)

for i, day in enumerate(days):
    row = i // num_cols + 1
    col = i % num_cols + 1

    day_data = df[df['Timestamp'].dt.day_name() == day]['WEIGHT']

    trace = go.Histogram(x=day_data, nbinsx=30, marker_color='skyblue', opacity=0.7, showlegend=False)
    fig3.add_trace(trace, row=row, col=col)

    fig3.update_xaxes(title_text='Weight', row=row, col=col)
    fig3.update_yaxes(title_text='Frequency', row=row, col=col)

# Adjust layout
fig3.update_layout(height=500 * num_rows, width=1000, template="plotly_dark")
graph3 = go.Figure(fig3)
#graph4 

# Get unique days in the dataset
days = df['Timestamp'].dt.day_name().unique()

# Calculate the number of rows and columns for subplots
num_rows = len(days) // 3 + (len(days) % 3 > 0)
num_cols = min(len(days), 3)

# Create subplots for each day of the week using Plotly
fig4 = sp.make_subplots(rows=num_rows, cols=num_cols, subplot_titles=days, shared_yaxes=True, vertical_spacing=0.1)

fig4.update_layout(title_text='Weight Distribution by Day of the Week', title_x=0.5, title_font_size=16)

for i, day in enumerate(days):
    row = i // num_cols + 1
    col = i % num_cols + 1

    day_data = df[df['Timestamp'].dt.day_name() == day]['WEIGHT']

    # Create a KDE plot
    trace = ff.create_distplot([day_data], [day], colors=['skyblue'], show_hist=False, show_rug=False)
    kde_trace = trace['data'][0]
    fig4.add_trace(kde_trace, row=row, col=col)

    fig4.update_xaxes(title_text='Weight', row=row, col=col)
    fig4.update_yaxes(title_text='Density', row=row, col=col)

# Adjust layout
fig4.update_layout(height=500 * num_rows, width=1000, template="plotly_dark")
graph4 = go.Figure(fig4)

#graph 5
average_weight = df.groupby(df['Timestamp'].dt.day_name())['WEIGHT'].mean().reset_index()
std_deviation = df.groupby(df['Timestamp'].dt.day_name())['WEIGHT'].std().reset_index()

# Create a bar plot with error bars using Plotly Express
fig5 = px.bar(
    average_weight,
    x='Timestamp',
    y='WEIGHT',
    error_y=std_deviation['WEIGHT'],
    labels={'WEIGHT': 'Average Weight'},
    title='Average Weight by Day of the Week',
    template="plotly_dark",  # You can change the template as needed
)

fig5.update_xaxes(title='Day of the Week')
fig5.update_yaxes(title='Average Weight')

graph5 = go.Figure(fig5)

#FIG6
fig6 = px.box(
    df,
    x=df['Timestamp'].dt.day_name(),
    y='WEIGHT',
    points='all',  # Show all data points
    title='Weight Distribution by Day of the Week',
    template="plotly_dark",  # You can change the template as needed
)

fig6.update_xaxes(title='Day of the Week')
fig6.update_yaxes(title='Weight')

graph6 = go.Figure(fig6)

#FIG7

max_weights_by_day = df.groupby(df['Timestamp'].dt.day_name())['WEIGHT'].max()

# Create gauge-like visualizations for each day with a dark theme
figs7 = []
for day, max_weight in max_weights_by_day.items():
    fig7 = go.Figure()  # Create a new figure object for each day
    fig7.add_trace(go.Indicator(
        mode="gauge+number",
        value=max_weight,
        title=f"Max Weight on {day}",
        gauge={
            'axis': {'range': [None, max_weights_by_day.max()], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_weight
            },
            'steps': [
                {'range': [0, max_weights_by_day.max() / 3], 'color': "rgba(255,0,0,0.7)"},
                {'range': [max_weights_by_day.max() / 3, max_weights_by_day.max() * 2 / 3], 'color': "rgba(255,165,0,0.7)"},
                {'range': [max_weights_by_day.max() * 2 / 3, max_weights_by_day.max()], 'color': "rgba(0,128,0,0.7)"}
            ],
        },
    ))
    fig7.update_layout(
        paper_bgcolor='black',  # Background color
        font=dict(color='white')  # Font color
    )
    figs7.append(go.Figure(fig7))

# Display gauge-like visualizations
for fig7 in figs7:
    graph7 = go.Figure(fig7)
    

    #FIG8

min_weights_by_day = df.groupby(df['Timestamp'].dt.day_name())['WEIGHT'].min().reset_index()

# Create a list of gauge-like visualizations for minimum weights
figs = []

for index, row in min_weights_by_day.iterrows():
    day = row['Timestamp']
    min_weight = row['WEIGHT']

    fig8 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min_weight,
        title=f"Min Weight on {day}",
        gauge={
            'axis': {'range': [None, min_weights_by_day['WEIGHT'].max()], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': min_weight,
            },
            'steps': [
                {'range': [0, min_weights_by_day['WEIGHT'].max() / 3], 'color': "rgba(255,0,0,0.7)"},
                {'range': [min_weights_by_day['WEIGHT'].max() / 3, min_weights_by_day['WEIGHT'].max() * 2 / 3], 'color': "rgba(255,165,0,0.7)"},
                {'range': [min_weights_by_day['WEIGHT'].max() * 2 / 3, min_weights_by_day['WEIGHT'].max()], 'color': "rgba(0,128,0,0.7)"}
            ],
        },
    ))
    
    fig8.update_layout(
        paper_bgcolor='black',  # Background color
        font=dict(color='white')  # Font color
    )
    
    figs.append(fig8)

# Show the interactive gauge-like visualizations for minimum weights
for fig8 in figs:
    graph8 = go.Figure(fig8)

#FIG 9

# Calculate the mode (most common value) of weights for each day of the week
mode_weights_by_day = df.groupby(df['Timestamp'].dt.day_name())['WEIGHT'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()

# Create a list of gauge-like visualizations for mode weights
figs = []

for index, row in mode_weights_by_day.iterrows():
    day = row['Timestamp']
    mode_weight = row['WEIGHT']

    fig9 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mode_weight,
        title=f"Mode Weight on {day}",
        gauge={
            'axis': {'range': [None, df['WEIGHT'].max()], 'tickwidth': 1},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': mode_weight,
            },
            'steps': [
                {'range': [0, df['WEIGHT'].max() / 3], 'color': "rgba(255,0,0,0.7)"},
                {'range': [df['WEIGHT'].max() / 3, df['WEIGHT'].max() * 2 / 3], 'color': "rgba(255,165,0,0.7)"},
                {'range': [df['WEIGHT'].max() * 2 / 3, df['WEIGHT'].max()], 'color': "rgba(0,128,0,0.7)"}
            ],
        },
    ))
    
    fig9.update_layout(
        paper_bgcolor='black',  # Background color
        font=dict(color='white')  # Font color
    )
    
    graph9 = figs.append(fig9)

# Show the interactive gauge-like visualizations for mode weights
for fig9 in figs:
    graph9 = go.Figure(fig9)



@app.route('/')
def index():
    return render_template('index.html', graph1=graph1, graph2=graph2, graph3=graph3, graph4=graph4, graph5=graph5, graph6=graph6, graph7=graph7, graph8=graph8, graph9=graph9)

if __name__ == '__main__':
    app.run(debug=True)
