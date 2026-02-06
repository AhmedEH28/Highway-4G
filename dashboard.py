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

# RSRP Categorization for better color visualization (4 categories)
def categorize_rsrp(rsrp):
    if rsrp >= -80: return '1. Excellent (≥ -80)'
    elif rsrp >= -90: return '2. Good (-80 to -90)'
    elif rsrp >= -100: return '3. Fair to Poor (-90 to -100)'
    else: return '4. Poor (< -100)'

df['RSRP_Status'] = df['RSRP'].apply(categorize_rsrp)

# RSRQ Categorization for better color visualization (4 categories)
def categorize_rsrq(rsrq):
    if rsrq >= -10: return '1. Excellent (≥ -10 dB)'
    elif rsrq > -15: return '2. Good (-10 to -15 dB)'
    elif rsrq > -20: return '3. Fair to Poor (-15 to -20 dB)'
    else: return '4. Poor (≤ -20 dB)'

df['RSRQ_Status'] = df['RSRQ'].apply(categorize_rsrq)

# Color mapping for RSRP categories
rsrp_color_map = {
    '1. Excellent (≥ -80)': '#008000',        # Green
    '2. Good (-80 to -90)': '#FFFF00',        # Yellow
    '3. Fair to Poor (-90 to -100)': '#FFA500',  # Orange
    '4. Poor (< -100)': '#FF0000'             # Red
}

# Color mapping for RSRQ categories
rsrq_color_map = {
    '1. Excellent (≥ -10 dB)': '#008000',        # Green
    '2. Good (-10 to -15 dB)': '#FFFF00',        # Yellow
    '3. Fair to Poor (-15 to -20 dB)': '#FFA500',  # Orange
    '4. Poor (≤ -20 dB)': '#FF0000'              # Red
}

# --- PROBLEM AREA DETECTION ---

# Detect problem areas
poor_signal_areas = df[df['RSRP'] < -100]
num_poor_areas = len(poor_signal_areas)

# Find worst tower
tower_performance = df.groupby('Cell_Id').agg({
    'Connection_Class': lambda x: (x == 'Poor').sum() / len(x) * 100,
    'Cell_Id': 'count'
}).rename(columns={'Connection_Class': 'Poor_Pct', 'Cell_Id': 'Count'}).reset_index()
worst_tower = tower_performance.loc[tower_performance['Poor_Pct'].idxmax()]

# Calculate network health score (0-100)
health_score = (
    (df['Connection_Class'] == 'Excellent').sum() * 3 + 
    (df['Connection_Class'] == 'Moderate').sum() * 1.5
) / len(df) * 100 / 3

# --- DASHBOARD SETUP ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            html.H1("📡 Drive Signal Test Analysis Dashboard", 
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
        # Tab 0: KPI Dashboard (NEW!)
        dbc.Tab([
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Executive Summary", style={'color': '#34495e', 'marginTop': '20px'})
                ])
            ]),
            
            # Network Health Score - Big Gauge
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=health_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Network Health Score", 'font': {'size': 24}},
                            delta={'reference': 70, 'increasing': {'color': "green"}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 40], 'color': '#ffcccc'},
                                    {'range': [40, 70], 'color': '#ffffcc'},
                                    {'range': [70, 100], 'color': '#ccffcc'}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90}}
                        )).update_layout(height=400)
                    )
                ], width=6),
                
                # Big Number Cards
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("Coverage Quality", style={'color': '#7f8c8d'}),
                                html.H2(f"{excellent_pct:.1f}%", style={'color': '#27ae60', 'fontWeight': 'bold'}),
                                html.P("Excellent", style={'color': '#95a5a6'})
                            ])
                        ], style={'textAlign': 'center', 'backgroundColor': '#ecf0f1'}), width=6),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("Total Handovers", style={'color': '#7f8c8d'}),
                                html.H2(f"{total_handovers:,}", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
                                html.P("Tower Switches", style={'color': '#95a5a6'})
                            ])
                        ], style={'textAlign': 'center', 'backgroundColor': '#ecf0f1'}), width=6),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg Signal Power", style={'color': '#7f8c8d'}),
                                html.H2(f"{avg_rsrp:.1f}", style={'color': '#9b59b6', 'fontWeight': 'bold'}),
                                html.P("dBm (RSRP)", style={'color': '#95a5a6'})
                            ])
                        ], style={'textAlign': 'center', 'backgroundColor': '#ecf0f1'}), width=6),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg Signal Quality", style={'color': '#7f8c8d'}),
                                html.H2(f"{avg_sinr:.1f}", style={'color': '#1abc9c', 'fontWeight': 'bold'}),
                                html.P("dB (SINR)", style={'color': '#95a5a6'})
                            ])
                        ], style={'textAlign': 'center', 'backgroundColor': '#ecf0f1'}), width=6),
                    ]),
                ], width=6),
            ]),
            
            html.Hr(),
            
            # Problem Area Detector
            dbc.Row([
                dbc.Col([
                    html.H4("⚠️ Problem Area Detector", style={'color': '#e74c3c', 'marginTop': '20px'})
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5(["🔴 ", f"{num_poor_areas:,} measurements with signal < -100 dBm"], className="alert-heading"),
                        html.P(f"That's {num_poor_areas/len(df)*100:.1f}% of total measurements - users likely experiencing connection drops")
                    ], color="danger"),
                ], width=12),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5(["🗼 ", f"Tower {int(worst_tower['Cell_Id'])} needs attention"], className="alert-heading"),
                        html.P(f"{worst_tower['Poor_Pct']:.1f}% of connections are poor quality ({int(worst_tower['Count'])} total measurements)")
                    ], color="warning"),
                ], width=12),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5(["📊 ", f"Overall Network Health: {'Good' if health_score > 70 else 'Needs Improvement'}"], className="alert-heading"),
                        html.P(f"Health score: {health_score:.1f}/100 - {'Above' if health_score > 70 else 'Below'} target threshold")
                    ], color="success" if health_score > 70 else "info"),
                ], width=12),
            ]),
            
        ], label="🎯 KPI Dashboard"),
        
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
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='rsrq-mapbox')
                ], width=6),
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
                            {'label': 'RSRP Status', 'value': 'RSRP_Status'},
                            {'label': 'RSRQ Status', 'value': 'RSRQ_Status'},
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
     Output('rsrq-mapbox', 'figure'),
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
    # Handle None values on initial page load
    if rsrp_range is None:
        rsrp_range = [df['RSRP'].min(), df['RSRP'].max()]
    if sinr_range is None:
        sinr_range = [df['SINR'].min(), df['SINR'].max()]
    if quality_filter is None:
        quality_filter = 'All'
    if ts_metric is None:
        ts_metric = 'all'
    if custom_x is None:
        custom_x = 'Speed_kmh'
    if custom_y is None:
        custom_y = 'SINR'
    if custom_color is None:
        custom_color = 'Connection_Class'
    
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
    
    # 1. Geo Signal Map (Categorized)
    fig1 = px.scatter_geo(df_filtered, lat='Latitude', lon='Longitude', color='RSRP_Status', 
                         color_discrete_map=rsrp_color_map,
                         category_orders={"RSRP_Status": list(rsrp_color_map.keys())},
                         scope='europe',
                         title='Signal Strength Map (Categorized RSRP)',
                         hover_data=['Time', 'Cell_Id', 'PCI', 'RSRP', 'SINR'], height=500)
    fig1.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
    
    # 2. Geo Quality Map
    fig2 = px.scatter_geo(df_filtered, lat='Latitude', lon='Longitude', color='Connection_Class',
                         scope='europe', title='Connection Quality Map',
                         color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                         height=500)
    fig2.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
    
    # 3. Signal Strength Mapbox (Interactive Street Map - Enhanced Details)
    fig3_mapbox = px.scatter_mapbox(df_filtered, 
                             lat="Latitude", 
                             lon="Longitude", 
                             color="RSRP_Status", 
                             color_discrete_map=rsrp_color_map,
                             category_orders={"RSRP_Status": list(rsrp_color_map.keys())},
                             size_max=15, 
                             zoom=12,
                             title="Interactive Map of Signal Strength (Categorized RSRP)",
                             custom_data=['Time', 'Latitude', 'Longitude', 'Elevation', 'RSRP', 'RSRQ', 'SINR', 'PCI', 'Cell_Id', 'RSRP_Status'],
                             height=600)
    fig3_mapbox.update_layout(mapbox_style="open-street-map")
    
    # Custom hover template
    fig3_mapbox.update_traces(
        hovertemplate="<b>Timestamp: %{customdata[0]}</b><br>" +
                     "<br>" +
                     "<b>GPS Coordinates:</b><br>" +
                     "  Lat: %{customdata[1]:.6f}°<br>" +
                     "  Lon: %{customdata[2]:.6f}°<br>" +
                     "  Elevation: %{customdata[3]:.1f}m<br>" +
                     "<br>" +
                     "<b>Signal Quality:</b><br>" +
                     "  RSRP: %{customdata[4]:.0f} dBm<br>" +
                     "  RSRQ: %{customdata[5]:.0f} dB<br>" +
                     "  SINR: %{customdata[7]:.0f} dB<br>" +
                     "<br>" +
                     "<b>Cell Info:</b><br>" +
                     "  PCI: %{customdata[8]}<br>" +
                     "  Cell_ID: %{customdata[9]}<br>" +
                     "<br>" +
                     "<b>Anomaly: NO</b><extra></extra>"
    )
    
    # 3b. Signal Quality Mapbox (Interactive Street Map - Enhanced Details)
    fig3b_mapbox = px.scatter_mapbox(df_filtered, 
                             lat="Latitude", 
                             lon="Longitude", 
                             color="RSRQ_Status", 
                             color_discrete_map=rsrq_color_map,
                             category_orders={"RSRQ_Status": list(rsrq_color_map.keys())},
                             size_max=15, 
                             zoom=12,
                             title="Interactive Map of Signal Quality (Categorized RSRQ)",
                             custom_data=['Time', 'Latitude', 'Longitude', 'Elevation', 'RSRP', 'RSRQ', 'SINR', 'PCI', 'Cell_Id', 'RSRQ_Status'],
                             height=600)
    fig3b_mapbox.update_layout(mapbox_style="open-street-map")
    
    # Custom hover template
    fig3b_mapbox.update_traces(
        hovertemplate="<b>Timestamp: %{customdata[0]}</b><br>" +
                     "<br>" +
                     "<b>GPS Coordinates:</b><br>" +
                     "  Lat: %{customdata[1]:.6f}°<br>" +
                     "  Lon: %{customdata[2]:.6f}°<br>" +
                     "  Elevation: %{customdata[3]:.1f}m<br>" +
                     "<br>" +
                     "<b>Signal Quality:</b><br>" +
                     "  RSRP: %{customdata[4]:.0f} dBm<br>" +
                     "  RSRQ: %{customdata[5]:.0f} dB<br>" +
                     "  SINR: %{customdata[7]:.0f} dB<br>" +
                     "<br>" +
                     "<b>Cell Info:</b><br>" +
                     "  PCI: %{customdata[8]}<br>" +
                     "  Cell_ID: %{customdata[9]}<br>" +
                     "<br>" +
                     "<b>Anomaly: NO</b><extra></extra>"
    )
    
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
    color_map = {
        'Connection_Class': {'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
        'RSRP_Status': rsrp_color_map,
        'RSRQ_Status': rsrq_color_map
    }
    
    fig14 = px.scatter(sample_custom, x=custom_x, y=custom_y, color=custom_color,
                      color_discrete_map=color_map.get(custom_color, None),
                      title=f'{custom_y} vs {custom_x} (Colored by {custom_color})',
                      height=500, opacity=0.7)
    
    return (filter_text, fig1, fig2, fig3_mapbox, fig3b_mapbox, fig4, fig5, fig6, fig7, 
            fig8, fig9, fig10, fig11, fig12, fig13, fig14)

# --- RUN SERVER ---

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Drive SIgnal Test Analysis Dashboard...")
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
