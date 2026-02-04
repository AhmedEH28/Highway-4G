import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# --- LOAD AND PROCESS DATA ---

# Load the clean dataset
df = pd.read_csv('Crawdad.csv')

# Convert Time to datetime
df['Time'] = pd.to_datetime(df['Time'])

# Calculate Derived Metrics
df['Noise'] = df['RSRP'] - df['SINR']

# Speed calculation
df['prev_lat'] = df['Latitude'].shift(1)
df['prev_lon'] = df['Longitude'].shift(1)
df['prev_lat'] = df['prev_lat'].fillna(df['Latitude'])
df['prev_lon'] = df['prev_lon'].fillna(df['Longitude'])

df['dist_diff'] = np.sqrt((df['Latitude'] - df['prev_lat'])**2 + (df['Longitude'] - df['prev_lon'])**2) * 111000
df['time_diff'] = df['Time'].diff().dt.total_seconds().fillna(1)
df['Speed_kmh'] = (df['dist_diff'] / df['time_diff']) * 3.6

# Handover detection (using Cell_Id = actual tower identifier)
df['Cell_Id_Change'] = df['Cell_Id'].diff().fillna(0)
handover_events = df[df['Cell_Id_Change'] != 0]

# Connection Classification
conditions = [
    (df['RSRP'] >= -85) & (df['SINR'] >= 20),
    (df['RSRP'] >= -95) & (df['RSRP'] < -85),
    (df['RSRP'] < -95)
]
choices = ['Excellent', 'Moderate', 'Poor']
df['Connection_Class'] = np.select(conditions, choices, default='Poor')

# --- DASHBOARD SETUP ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# --- CREATE VISUALIZATIONS ---

# 1. Geo Map - Signal Strength
fig_geo = px.scatter_geo(df, 
                         lat='Latitude', 
                         lon='Longitude', 
                         color='RSRP', 
                         color_continuous_scale='RdYlGn',
                         scope='europe',
                         title='Drive Test Route (Colored by Signal Strength - RSRP)',
                         hover_data=['Time', 'PCI', 'RSRP', 'SINR'],
                         height=500)
fig_geo.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)

# 2. Geo Map - Connection Class
fig_class = px.scatter_geo(df, 
                           lat='Latitude', 
                           lon='Longitude', 
                           color='Connection_Class',
                           scope='europe',
                           title='Drive Test Route by Connection Quality Class',
                           color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                           height=500)
fig_class.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)

# 3. Signal Quality Distributions
fig_dist = make_subplots(
    rows=2, cols=2,
    subplot_titles=('RSRP Distribution', 'RSRQ Distribution', 
                   'SINR Distribution', 'Connection Class Distribution'),
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "bar"}]]
)

# RSRP
fig_dist.add_trace(
    go.Histogram(x=df['RSRP'], name='RSRP', marker_color='blue', nbinsx=30),
    row=1, col=1
)
fig_dist.add_vline(x=-80, line_dash="dash", line_color="green", row=1, col=1, 
                  annotation_text="Good", annotation_position="top")
fig_dist.add_vline(x=-90, line_dash="dash", line_color="red", row=1, col=1,
                  annotation_text="Poor", annotation_position="bottom")

# RSRQ
fig_dist.add_trace(
    go.Histogram(x=df['RSRQ'], name='RSRQ', marker_color='orange', nbinsx=30),
    row=1, col=2
)

# SINR
fig_dist.add_trace(
    go.Histogram(x=df['SINR'], name='SINR', marker_color='green', nbinsx=30),
    row=2, col=1
)

# Connection Class
class_counts = df['Connection_Class'].value_counts()
fig_dist.add_trace(
    go.Bar(x=class_counts.index, y=class_counts.values, 
           marker_color=['green', 'yellow', 'red'], name='Count'),
    row=2, col=2
)

fig_dist.update_xaxes(title_text="RSRP (dBm)", row=1, col=1)
fig_dist.update_xaxes(title_text="RSRQ (dB)", row=1, col=2)
fig_dist.update_xaxes(title_text="SINR (dB)", row=2, col=1)
fig_dist.update_xaxes(title_text="Class", row=2, col=2)
fig_dist.update_yaxes(title_text="Count", row=1, col=1)
fig_dist.update_yaxes(title_text="Count", row=1, col=2)
fig_dist.update_yaxes(title_text="Count", row=2, col=1)
fig_dist.update_yaxes(title_text="Count", row=2, col=2)

fig_dist.update_layout(height=600, showlegend=False, title_text="Signal Quality Metrics Distribution")

# 4. PCI Timeline
fig_pci = go.Figure()
fig_pci.add_trace(go.Scatter(
    x=df['Time'], y=df['RSRP'],
    mode='lines', name='RSRP', line=dict(color='gray', width=1),
    yaxis='y', opacity=0.5
))
fig_pci.add_trace(go.Scatter(
    x=df['Time'], y=df['Cell_Id'],
    mode='lines+markers', name='Cell_Id (Tower ID)',
    line=dict(shape='hv'),
    yaxis='y2'
))
fig_pci.update_layout(
    title='Signal Strength (RSRP) vs Tower Switching (Cell_Id) Over Time',
    xaxis=dict(title='Time'),
    yaxis=dict(title='RSRP (dBm)', side='left'),
    yaxis2=dict(title='Cell_Id (Tower ID)', side='right', overlaying='y', showgrid=False),
    legend=dict(x=0.01, y=0.99),
    height=500
)

# 5. Speed Impact
fig_speed = px.scatter(df, x='Speed_kmh', y='SINR', 
                       color='RSRP',
                       trendline="ols",
                       title='Impact of Speed on Signal Quality (SINR)',
                       labels={'Speed_kmh': 'Vehicle Speed (km/h)', 'SINR': 'Signal Quality (dB)'},
                       height=450)

# 6. Elevation Impact
fig_elev = px.scatter(df, x='Elevation', y='RSRP', 
                      color='Cell_Id',
                      title='Impact of Elevation on Signal Strength (Colored by Tower)',
                      labels={'Elevation': 'Elevation (m)', 'RSRP': 'Signal Power (dBm)'},
                      height=450)

# 7. Correlation Heatmap
corr_df = df[['RSRP', 'RSRQ', 'SINR', 'Elevation', 'Speed_kmh']].corr()
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_df.values,
    x=corr_df.columns,
    y=corr_df.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_df.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 12}
))
fig_corr.update_layout(title='Correlation Matrix', height=450)

# 8. Time Series - All Metrics
fig_timeseries = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=('RSRP Over Time', 'SINR Over Time', 'Speed Over Time'),
    vertical_spacing=0.08
)

fig_timeseries.add_trace(
    go.Scatter(x=df['Time'], y=df['RSRP'], name='RSRP', line=dict(color='blue')),
    row=1, col=1
)
fig_timeseries.add_trace(
    go.Scatter(x=df['Time'], y=df['SINR'], name='SINR', line=dict(color='green')),
    row=2, col=1
)
fig_timeseries.add_trace(
    go.Scatter(x=df['Time'], y=df['Speed_kmh'], name='Speed', line=dict(color='red')),
    row=3, col=1
)

fig_timeseries.update_xaxes(title_text="Time", row=3, col=1)
fig_timeseries.update_yaxes(title_text="RSRP (dBm)", row=1, col=1)
fig_timeseries.update_yaxes(title_text="SINR (dB)", row=2, col=1)
fig_timeseries.update_yaxes(title_text="Speed (km/h)", row=3, col=1)
fig_timeseries.update_layout(height=700, showlegend=True, title_text="Signal Metrics Time Series")

# 9. Cell Tower Performance Comparison (using Cell_Id)
tower_stats = df.groupby('Cell_Id').agg({
    'RSRP': 'mean',
    'SINR': 'mean',
    'RSRQ': 'mean',
    'Cell_Id': 'count'
}).rename(columns={'Cell_Id': 'Sample_Count'}).reset_index()
tower_stats = tower_stats.sort_values('Sample_Count', ascending=False).head(10)

fig_towers = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 10 Towers by Usage', 'Average Signal Quality by Tower'),
    specs=[[{"type": "bar"}, {"type": "scatter"}]]
)

fig_towers.add_trace(
    go.Bar(x=tower_stats['Cell_Id'].astype(str), y=tower_stats['Sample_Count'], 
           name='Samples', marker_color='steelblue'),
    row=1, col=1
)

fig_towers.add_trace(
    go.Scatter(x=tower_stats['Cell_Id'].astype(str), y=tower_stats['RSRP'],
               mode='markers+lines', name='Avg RSRP', 
               marker=dict(size=10, color='blue')),
    row=1, col=2
)
fig_towers.add_trace(
    go.Scatter(x=tower_stats['Cell_Id'].astype(str), y=tower_stats['SINR'],
               mode='markers+lines', name='Avg SINR',
               marker=dict(size=10, color='green'), yaxis='y2'),
    row=1, col=2
)

fig_towers.update_xaxes(title_text="Tower (Cell_Id)", row=1, col=1)
fig_towers.update_xaxes(title_text="Tower (Cell_Id)", row=1, col=2)
fig_towers.update_yaxes(title_text="Sample Count", row=1, col=1)
fig_towers.update_yaxes(title_text="RSRP (dBm)", row=1, col=2)
fig_towers.update_layout(height=450, showlegend=True, 
                        title_text="Cell Tower Performance Analysis (by Cell_Id)",
                        yaxis2=dict(title='SINR (dB)', overlaying='y', side='right'))

# 10. Signal Quality Heatmap (RSRP vs SINR)
fig_heatmap = go.Figure(go.Histogram2d(
    x=df['RSRP'],
    y=df['SINR'],
    colorscale='Viridis',
    nbinsx=50,
    nbinsy=50
))
fig_heatmap.update_layout(
    title='Signal Quality Density Map (RSRP vs SINR)',
    xaxis_title='RSRP (dBm)',
    yaxis_title='SINR (dB)',
    height=450
)

# 11. Connection Quality Over Time
df_hour = df.copy()
df_hour['Hour'] = df_hour['Time'].dt.hour
quality_by_hour = df_hour.groupby(['Hour', 'Connection_Class']).size().reset_index(name='Count')
quality_pivot = quality_by_hour.pivot(index='Hour', columns='Connection_Class', values='Count').fillna(0)

fig_quality_time = go.Figure()
for col in ['Excellent', 'Moderate', 'Poor']:
    if col in quality_pivot.columns:
        color = {'Excellent': 'green', 'Moderate': 'yellow', 'Poor': 'red'}[col]
        fig_quality_time.add_trace(go.Bar(
            x=quality_pivot.index,
            y=quality_pivot[col],
            name=col,
            marker_color=color
        ))

fig_quality_time.update_layout(
    barmode='stack',
    title='Connection Quality Distribution by Hour of Day',
    xaxis_title='Hour of Day',
    yaxis_title='Sample Count',
    height=450
)

# 12. RSRP vs RSRQ Quality Zones
fig_quality_zones = px.scatter(df.sample(min(5000, len(df))), 
                               x='RSRP', y='RSRQ',
                               color='Connection_Class',
                               color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                               title='Signal Quality Zones (RSRP vs RSRQ)',
                               labels={'RSRP': 'RSRP (dBm)', 'RSRQ': 'RSRQ (dB)'},
                               opacity=0.6,
                               height=450)
fig_quality_zones.add_hline(y=-10, line_dash="dash", line_color="gray", 
                            annotation_text="Good RSRQ", annotation_position="right")
fig_quality_zones.add_vline(x=-85, line_dash="dash", line_color="gray",
                            annotation_text="Good RSRP", annotation_position="top")

# --- STATISTICS CARDS ---

def create_stat_card(title, value, color):
    return dbc.Card([
        dbc.CardBody([
            html.H4(title, className="card-title", style={'fontSize': '14px'}),
            html.H2(value, style={'color': color, 'fontWeight': 'bold'}),
        ])
    ], style={'textAlign': 'center', 'margin': '10px'})

# Calculate statistics
total_samples = len(df)
total_handovers = len(handover_events)
avg_rsrp = df['RSRP'].mean()
avg_sinr = df['SINR'].mean()
avg_speed = df['Speed_kmh'].mean()
excellent_pct = (df['Connection_Class'] == 'Excellent').sum() / len(df) * 100
moderate_pct = (df['Connection_Class'] == 'Moderate').sum() / len(df) * 100
poor_pct = (df['Connection_Class'] == 'Poor').sum() / len(df) * 100

# --- DASHBOARD LAYOUT ---

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📡 Drive Test Analysis Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '20px', 'marginBottom': '10px'}),
            html.P("Interactive Analysis of LTE Network Performance", 
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    ]),
    
    html.Hr(),
    
    # Summary Statistics Cards
    dbc.Row([
        dbc.Col([
            html.H4("📊 Key Metrics", style={'color': '#34495e', 'marginBottom': '15px'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col(create_stat_card("Total Samples", f"{total_samples:,}", "#3498db"), width=2),
        dbc.Col(create_stat_card("Handovers", f"{total_handovers}", "#e74c3c"), width=2),
        dbc.Col(create_stat_card("Avg RSRP", f"{avg_rsrp:.1f} dBm", "#9b59b6"), width=2),
        dbc.Col(create_stat_card("Avg SINR", f"{avg_sinr:.1f} dB", "#1abc9c"), width=2),
        dbc.Col(create_stat_card("Avg Speed", f"{avg_speed:.1f} km/h", "#f39c12"), width=2),
        dbc.Col(create_stat_card("Excellent", f"{excellent_pct:.1f}%", "#27ae60"), width=2),
    ]),
    
    html.Hr(),
    
    # Interactive Filters
    dbc.Row([
        dbc.Col([
            html.H4("🎛️ Interactive Filters", style={'color': '#34495e', 'marginTop': '10px'}),
            html.P("Filter the data to explore specific conditions", style={'color': '#7f8c8d', 'fontSize': '14px'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("RSRP Range (dBm):", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='rsrp-slider',
                min=df['RSRP'].min(),
                max=df['RSRP'].max(),
                value=[df['RSRP'].min(), df['RSRP'].max()],
                marks={int(df['RSRP'].min()): f"{int(df['RSRP'].min())}",
                       -100: '-100', -90: '-90', -80: '-80',
                       int(df['RSRP'].max()): f"{int(df['RSRP'].max())}"},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=4),
        dbc.Col([
            html.Label("SINR Range (dB):", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='sinr-slider',
                min=df['SINR'].min(),
                max=df['SINR'].max(),
                value=[df['SINR'].min(), df['SINR'].max()],
                marks={int(df['SINR'].min()): f"{int(df['SINR'].min())}",
                       0: '0', 10: '10', 20: '20',
                       int(df['SINR'].max()): f"{int(df['SINR'].max())}"},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=4),
        dbc.Col([
            html.Label("Connection Quality:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='quality-dropdown',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Excellent', 'value': 'Excellent'},
                    {'label': 'Moderate', 'value': 'Moderate'},
                    {'label': 'Poor', 'value': 'Poor'}
                ],
                value='All',
                clearable=False
            )
        ], width=4),
    ], style={'marginBottom': '20px'}),
    
    # Filtered Data Info
    dbc.Row([
        dbc.Col([
            html.Div(id='filter-info', style={'textAlign': 'center', 'padding': '10px', 
                                             'backgroundColor': '#ecf0f1', 'borderRadius': '5px',
                                             'marginBottom': '15px'})
        ])
    ]),
    
    html.Hr(),
    
    # Tabbed Interface
    dbc.Tabs([
        # Tab 1: Geographic Analysis
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("🗺️ Geographic Analysis", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='geo-signal-map')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='geo-quality-map')
                ], width=6),
            ]),
        ], label="📍 Geographic Maps"),
        
        # Tab 2: Signal Quality
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Signal Quality Distributions", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='signal-mapbox')
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='signal-distributions')
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='correlation-heatmap')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='quality-zones-scatter')
                ], width=6),
            ]),
        ], label="📈 Signal Quality"),
        
        # Tab 3: Time Series
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("⏱️ Time Series Analysis", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Metric:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='timeseries-metric',
                        options=[
                            {'label': 'All Metrics', 'value': 'all'},
                            {'label': 'RSRP', 'value': 'RSRP'},
                            {'label': 'SINR', 'value': 'SINR'},
                            {'label': 'RSRQ', 'value': 'RSRQ'},
                            {'label': 'Speed', 'value': 'Speed_kmh'}
                        ],
                        value='all',
                        clearable=False
                    )
                ], width=3),
            ], style={'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='timeseries-plot')
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='pci-timeline')
                ], width=12),
            ]),
        ], label="⏰ Time Series"),
        
        # Tab 4: Handover & Towers
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("🗼 Cell Tower & Handover Analysis", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='tower-performance')
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='quality-by-hour')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='signal-density-heatmap')
                ], width=6),
            ]),
        ], label="🗼 Towers & Handovers"),
        
        # Tab 5: Impact Analysis
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("🔍 Environmental Impact Analysis", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='speed-impact')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='elevation-impact')
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Select X-Axis:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='custom-x-axis',
                        options=[
                            {'label': 'Speed (km/h)', 'value': 'Speed_kmh'},
                            {'label': 'Elevation (m)', 'value': 'Elevation'},
                            {'label': 'Time of Day (Hour)', 'value': 'Hour'},
                        ],
                        value='Speed_kmh'
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Select Y-Axis:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='custom-y-axis',
                        options=[
                            {'label': 'RSRP (dBm)', 'value': 'RSRP'},
                            {'label': 'SINR (dB)', 'value': 'SINR'},
                            {'label': 'RSRQ (dB)', 'value': 'RSRQ'},
                        ],
                        value='SINR'
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Color By:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='custom-color',
                        options=[
                            {'label': 'Connection Class', 'value': 'Connection_Class'},
                            {'label': 'Cell_Id (Tower)', 'value': 'Cell_Id'},
                            {'label': 'PCI (Radio Channel)', 'value': 'PCI'},
                            {'label': 'RSRP', 'value': 'RSRP'},
                        ],
                        value='Connection_Class'
                    )
                ], width=3),
            ], style={'marginBottom': '15px'}),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='custom-scatter')
                ], width=12),
            ]),
        ], label="🔍 Impact Analysis"),
        
    ], style={'marginTop': '20px'}),
    
    html.Hr(),
    
    # Footer
    html.Footer([
        html.P("📡 Drive Test Analysis Dashboard - AIoT Week 4 | Powered by Plotly Dash", 
              style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '30px', 'marginBottom': '20px'})
    ])
    
], fluid=True)

# --- CALLBACKS FOR INTERACTIVITY ---

@callback(
    [Output('filter-info', 'children'),
     Output('geo-signal-map', 'figure'),
     Output('geo-quality-map', 'figure'),
     Output('signal-mapbox', 'figure'),
     Output('signal-distributions', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('quality-zones-scatter', 'figure'),
     Output('timeseries-plot', 'figure'),
     Output('pci-timeline', 'figure'),
     Output('tower-performance', 'figure'),
     Output('quality-by-hour', 'figure'),
     Output('signal-density-heatmap', 'figure'),
     Output('speed-impact', 'figure'),
     Output('elevation-impact', 'figure'),
     Output('custom-scatter', 'figure')],
    [Input('rsrp-slider', 'value'),
     Input('sinr-slider', 'value'),
     Input('quality-dropdown', 'value'),
     Input('timeseries-metric', 'value'),
     Input('custom-x-axis', 'value'),
     Input('custom-y-axis', 'value'),
     Input('custom-color', 'value')]
)
def update_dashboard(rsrp_range, sinr_range, quality_filter, ts_metric, custom_x, custom_y, custom_color):
    # Filter data
    df_filtered = df[
        (df['RSRP'] >= rsrp_range[0]) & (df['RSRP'] <= rsrp_range[1]) &
        (df['SINR'] >= sinr_range[0]) & (df['SINR'] <= sinr_range[1])
    ]
    
    if quality_filter != 'All':
        df_filtered = df_filtered[df_filtered['Connection_Class'] == quality_filter]
    
    # If no data after filtering, use full dataset
    if len(df_filtered) == 0:
        df_filtered = df
    
    # Filter info
    filter_text = html.Div([
        html.Span(f"📊 Showing {len(df_filtered):,} of {len(df):,} samples ", style={'fontWeight': 'bold'}),
        html.Span(f"| RSRP: [{rsrp_range[0]:.1f}, {rsrp_range[1]:.1f}] dBm "),
        html.Span(f"| SINR: [{sinr_range[0]:.1f}, {sinr_range[1]:.1f}] dB "),
        html.Span(f"| Quality: {quality_filter}")
    ])
    
    # 1. Geo Signal Map
    fig1 = px.scatter_geo(df_filtered, lat='Latitude', lon='Longitude', color='RSRP', 
                         color_continuous_scale='RdYlGn', scope='europe',
                         title='Signal Strength Map (RSRP)',
                         hover_data=['Time', 'Cell_Id', 'PCI', 'RSRP', 'SINR'], height=500)
    fig1.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
    
    # 2. Geo Quality Map
    fig2 = px.scatter_geo(df_filtered, lat='Latitude', lon='Longitude', color='Connection_Class',
                         scope='europe', title='Connection Quality Map',
                         color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                         height=500)
    fig2.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
    
    # 3. Signal Strength Mapbox (Interactive Street Map)
    fig3_mapbox = px.scatter_mapbox(df_filtered, 
                             lat="Latitude", 
                             lon="Longitude", 
                             color="RSRP", 
                             size_max=15, 
                             zoom=12,
                             color_continuous_scale=px.colors.diverging.RdYlGn, 
                             title="Interactive Map of Signal Strength (RSRP)",
                             hover_data=['PCI', 'Cell_Id', 'SINR', 'RSRQ'],
                             height=600)
    fig3_mapbox.update_layout(mapbox_style="open-street-map")
    
    # 4. Signal Distributions
    fig4 = make_subplots(rows=2, cols=2,
                        subplot_titles=('RSRP Distribution', 'RSRQ Distribution', 
                                      'SINR Distribution', 'Connection Class'),
                        specs=[[{"type": "histogram"}, {"type": "histogram"}],
                              [{"type": "histogram"}, {"type": "bar"}]])
    
    fig4.add_trace(go.Histogram(x=df_filtered['RSRP'], name='RSRP', marker_color='blue', nbinsx=30), row=1, col=1)
    fig4.add_vline(x=-80, line_dash="dash", line_color="green", row=1, col=1)
    fig4.add_vline(x=-90, line_dash="dash", line_color="red", row=1, col=1)
    
    fig4.add_trace(go.Histogram(x=df_filtered['RSRQ'], name='RSRQ', marker_color='orange', nbinsx=30), row=1, col=2)
    fig4.add_trace(go.Histogram(x=df_filtered['SINR'], name='SINR', marker_color='green', nbinsx=30), row=2, col=1)
    
    class_counts = df_filtered['Connection_Class'].value_counts()
    fig4.add_trace(go.Bar(x=class_counts.index, y=class_counts.values, 
                         marker_color=['green', 'yellow', 'red']), row=2, col=2)
    
    fig4.update_layout(height=600, showlegend=False, title_text="Signal Quality Distributions")
    
    # 5. Correlation Heatmap
    corr_df = df_filtered[['RSRP', 'RSRQ', 'SINR', 'Elevation', 'Speed_kmh']].corr()
    fig5 = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
                                     colorscale='RdBu', zmid=0, text=corr_df.values.round(2),
                                     texttemplate='%{text}', textfont={"size": 12}))
    fig5.update_layout(title='Correlation Matrix', height=450)
    
    # 6. Quality Zones Scatter
    sample_data = df_filtered.sample(min(5000, len(df_filtered)))
    fig6 = px.scatter(sample_data, x='RSRP', y='RSRQ', color='Connection_Class',
                     color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                     title='Signal Quality Zones (RSRP vs RSRQ)', opacity=0.6, height=450)
    fig6.add_hline(y=-10, line_dash="dash", line_color="gray")
    fig6.add_vline(x=-85, line_dash="dash", line_color="gray")
    
    # 7. Time Series
    if ts_metric == 'all':
        fig7 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=('RSRP', 'SINR', 'Speed'),
                            vertical_spacing=0.08)
        fig7.add_trace(go.Scatter(x=df_filtered['Time'], y=df_filtered['RSRP'], 
                                 name='RSRP', line=dict(color='blue')), row=1, col=1)
        fig7.add_trace(go.Scatter(x=df_filtered['Time'], y=df_filtered['SINR'], 
                                 name='SINR', line=dict(color='green')), row=2, col=1)
        fig7.add_trace(go.Scatter(x=df_filtered['Time'], y=df_filtered['Speed_kmh'], 
                                 name='Speed', line=dict(color='red')), row=3, col=1)
        fig7.update_layout(height=700, title_text="Metrics Over Time")
    else:
        fig7 = px.line(df_filtered, x='Time', y=ts_metric, title=f'{ts_metric} Over Time')
        fig7.update_layout(height=500)
    
    # 8. Tower Timeline (PCI over time)
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=df_filtered['Time'], y=df_filtered['RSRP'],
                             mode='lines', name='RSRP', line=dict(color='gray', width=1),
                             yaxis='y', opacity=0.5))
    fig8.add_trace(go.Scatter(x=df_filtered['Time'], y=df_filtered['Cell_Id'],
                             mode='lines', name='Cell_Id (Tower)', line=dict(shape='hv'), yaxis='y2'))
    fig8.update_layout(title='RSRP vs Tower Switching (Cell_Id)',
                      yaxis=dict(title='RSRP (dBm)'), 
                      yaxis2=dict(title='Cell_Id (Tower ID)', overlaying='y', side='right'),
                      height=500)
    
    # 9. Tower Performance (using Cell_Id)
    tower_stats = df_filtered.groupby('Cell_Id').agg({
        'RSRP': 'mean', 'SINR': 'mean', 'Cell_Id': 'count'
    }).rename(columns={'Cell_Id': 'Count'}).reset_index().sort_values('Count', ascending=False).head(10)
    
    fig9 = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Top Towers by Usage', 'Avg Signal Quality'),
                        specs=[[{"type": "bar"}, {"type": "scatter"}]])
    
    fig9.add_trace(go.Bar(x=tower_stats['Cell_Id'].astype(str), y=tower_stats['Count'],
                         marker_color='steelblue'), row=1, col=1)
    fig9.add_trace(go.Scatter(x=tower_stats['Cell_Id'].astype(str), y=tower_stats['RSRP'],
                             mode='markers+lines', name='RSRP', marker=dict(size=10)), row=1, col=2)
    fig9.update_layout(height=450, title_text="Cell Tower Performance (by Cell_Id)")
    
    # 10. Quality by Hour
    df_hour = df_filtered.copy()
    df_hour['Hour'] = df_hour['Time'].dt.hour
    quality_by_hour = df_hour.groupby(['Hour', 'Connection_Class']).size().reset_index(name='Count')
    quality_pivot = quality_by_hour.pivot(index='Hour', columns='Connection_Class', values='Count').fillna(0)
    
    fig10 = go.Figure()
    for col in ['Excellent', 'Moderate', 'Poor']:
        if col in quality_pivot.columns:
            color = {'Excellent': 'green', 'Moderate': 'yellow', 'Poor': 'red'}[col]
            fig10.add_trace(go.Bar(x=quality_pivot.index, y=quality_pivot[col], 
                                 name=col, marker_color=color))
    fig10.update_layout(barmode='stack', title='Connection Quality by Hour', 
                      xaxis_title='Hour', yaxis_title='Count', height=450)
    
    # 11. Signal Density Heatmap
    fig11 = go.Figure(go.Histogram2d(x=df_filtered['RSRP'], y=df_filtered['SINR'],
                                     colorscale='Viridis', nbinsx=50, nbinsy=50))
    fig11.update_layout(title='Signal Quality Density (RSRP vs SINR)',
                       xaxis_title='RSRP (dBm)', yaxis_title='SINR (dB)', height=450)
    
    # 12. Speed Impact
    fig12 = px.scatter(df_filtered, x='Speed_kmh', y='SINR', color='RSRP',
                      trendline="ols", title='Speed Impact on Signal Quality',
                      labels={'Speed_kmh': 'Speed (km/h)', 'SINR': 'SINR (dB)'}, height=450)
    
    # 13. Elevation Impact
    fig13 = px.scatter(df_filtered, x='Elevation', y='RSRP', color='Cell_Id',
                      title='Elevation Impact on Signal Strength (Colored by Tower)',
                      labels={'Elevation': 'Elevation (m)', 'RSRP': 'RSRP (dBm)'}, height=450)
    
    # 14. Custom Scatter
    sample_custom = df_filtered.sample(min(5000, len(df_filtered)))
    color_map = {'Connection_Class': {'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'}}
    
    fig14 = px.scatter(sample_custom, x=custom_x, y=custom_y, color=custom_color,
                      color_discrete_map=color_map.get(custom_color, None),
                      title=f'{custom_y} vs {custom_x} (Colored by {custom_color})',
                      height=500, opacity=0.7)
    
    return (filter_text, fig1, fig2, fig3_mapbox, fig4, fig5, fig6, fig7, 
            fig8, fig9, fig10, fig11, fig12, fig13, fig14)

# --- RUN SERVER ---

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Drive Test Analysis Dashboard...")
    print("="*60)
    print(f"📊 Loaded {total_samples:,} data points")
    print(f"🔄 Detected {total_handovers} handover events")
    print(f"✅ Excellent connections: {excellent_pct:.1f}%")
    print(f"⚠️  Moderate connections: {moderate_pct:.1f}%")
    print(f"❌ Poor connections: {poor_pct:.1f}%")
    print("="*60)
    print("🌐 Dashboard running at: http://127.0.0.1:8050/")
    print("   Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)
