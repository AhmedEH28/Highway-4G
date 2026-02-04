import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Create output folder for analysis results
output_folder = 'analysis_output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

# --- 1. DATA LOADING & CLEANING ---

# Load the clean dataset
df = pd.read_csv('Crawdad.csv')

# Convert Time to datetime objects for proper time-series plotting
df['Time'] = pd.to_datetime(df['Time'])

# Calculate Derived Metrics (Missing Data Engineering)
# RSSI is roughly RSRP + 10*log10(N_subframes) or often approximated as RSRP + RSRQ + (Constant)
# A common engineering approximation for LTE RSSI: RSSI ≈ RSRP - RSRQ - 10*log(12*Nrb) (complex)
# Simplified approximation used for visualization when RSSI is missing: 
# We will derive an 'Interference' metric: Noise = RSRP - SINR (since SINR = Signal / Noise)
df['Noise'] = df['RSRP'] - df['SINR']

# Create a 'Speed' column (Distance between points / Time difference)
# Haversine formula is better, but we'll use simple Euclidean for rough approximation in meters
# Assuming coordinates are close enough
df['prev_lat'] = df['Latitude'].shift(1)
df['prev_lon'] = df['Longitude'].shift(1)
# Fill NaT for first row
df['prev_lat'] = df['prev_lat'].fillna(df['Latitude'])
df['prev_lon'] = df['prev_lon'].fillna(df['Longitude'])

# Approximate distance calculation (very rough, assumes flat earth for small distances)
df['dist_diff'] = np.sqrt((df['Latitude'] - df['prev_lat'])**2 + (df['Longitude'] - df['prev_lon'])**2) * 111000 # Convert degrees to meters
# Calculate time difference in seconds
df['time_diff'] = df['Time'].diff().dt.total_seconds().fillna(1)
# Speed in km/h
df['Speed_kmh'] = (df['dist_diff'] / df['time_diff']) * 3.6

print("Data Loaded and Processed. Shape:", df.shape)
print(df.head())

# Save processed data
df.to_csv(os.path.join(output_folder, 'processed_data.csv'), index=False)
print(f"Saved processed data to {output_folder}/processed_data.csv")

# --- 2. GEO-SPATIAL VISUALIZATION (The Drive Route) ---

# Create a map of the drive test colored by Signal Strength (RSRP)
fig = px.scatter_geo(df, 
                     lat='Latitude', 
                     lon='Longitude', 
                     color='RSRP', 
                     color_continuous_scale='RdYlGn', # Red (bad) to Green (good)
                     scope='europe',
                     title='Drive Test Route in Salzburg (Colored by Signal Strength)',
                     hover_data=['Time', 'PCI', 'RSRP', 'SINR'],
                     height=600)

fig.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
fig.write_html(os.path.join(output_folder, 'geo_signal_strength_map.html'))
print(f"Saved geo map to {output_folder}/geo_signal_strength_map.html")
fig.show()

# --- 3. SIGNAL QUALITY DISTRIBUTION ANALYSIS ---

plt.figure(figsize=(15, 10))

# Plot 1: RSRP Distribution (Signal Power)
plt.subplot(2, 2, 1)
sns.histplot(df['RSRP'], bins=30, kde=True, color='blue')
plt.title('Distribution of RSRP (Reference Signal Received Power)')
plt.xlabel('RSRP (dBm)')
plt.ylabel('Frequency')
# Add lines for "Good" (> -80), "Fair" (-90 to -80), "Poor" (< -90)
plt.axvline(-80, color='green', linestyle='--', label='Good Threshold')
plt.axvline(-90, color='red', linestyle='--', label='Poor Threshold')
plt.legend()

# Plot 2: RSRQ Distribution (Signal Quality)
plt.subplot(2, 2, 2)
sns.histplot(df['RSRQ'], bins=30, kde=True, color='orange')
plt.title('Distribution of RSRQ (Reference Signal Received Quality)')
plt.xlabel('RSRQ (dB)')
plt.ylabel('Frequency')

# Plot 3: SINR Distribution (Signal-to-Interference Ratio)
plt.subplot(2, 2, 3)
sns.histplot(df['SINR'], bins=30, kde=True, color='green')
plt.title('Distribution of SINR (Signal to Interference & Noise Ratio)')
plt.xlabel('SINR (dB)')
plt.ylabel('Frequency')

# Plot 4: Correlation Heatmap
plt.subplot(2, 2, 4)
# Select only numeric columns for correlation
corr_df = df[['RSRP', 'RSRQ', 'SINR', 'Elevation', 'Speed_kmh']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'signal_quality_distributions.png'), dpi=300, bbox_inches='tight')
print(f"Saved signal quality plots to {output_folder}/signal_quality_distributions.png")
plt.show()

# --- 4. HANDOVER & CELL TOWER ANALYSIS ---

# PCI (Physical Cell ID) indicates which tower sector the phone is connected to.
# Changes in PCI indicate a "Handover" event.

# Identify Handovers
df['PCI_Change'] = df['PCI'].diff().fillna(0)
handover_events = df[df['PCI_Change'] != 0]

print(f"\nTotal Handovers detected: {len(handover_events)}")

# Save handover events
handover_events.to_csv(os.path.join(output_folder, 'handover_events.csv'), index=False)
print(f"Saved handover events to {output_folder}/handover_events.csv")

# Plot PCI over Time
fig_pci = go.Figure()

# Add RSRP as a background trace
fig_pci.add_trace(go.Scatter(
    x=df['Time'], y=df['RSRP'],
    mode='lines', name='RSRP', line=dict(color='gray', width=1),
    yaxis='y', opacity=0.5
))

# Add PCI as a step plot
fig_pci.add_trace(go.Scatter(
    x=df['Time'], y=df['PCI'],
    mode='lines+markers', name='PCI (Tower ID)',
    line=dict(shape='hv'), # Step plot
    yaxis='y2'
))

# Create layout with dual y-axes
fig_pci.update_layout(
    title='Signal Strength (RSRP) vs Tower Switching (PCI) over Time',
    xaxis=dict(title='Time'),
    yaxis=dict(title='RSRP (dBm)', side='left'),
    yaxis2=dict(title='PCI (Tower ID)', side='right', overlaying='y', showgrid=False),
    legend=dict(x=0.01, y=0.99),
    height=500
)

fig_pci.write_html(os.path.join(output_folder, 'pci_timeline.html'))
print(f"Saved PCI timeline to {output_folder}/pci_timeline.html")
fig_pci.show()

# --- 5. SPEED & ELEVATION IMPACT ---

# Does speed affect signal quality?
fig_speed = px.scatter(df, x='Speed_kmh', y='SINR', 
                       color='RSRP', 
                       trendline="ols", # Ordinary Least Squares regression
                       title='Impact of Speed on Signal Quality (SINR)',
                       labels={'Speed_kmh': 'Vehicle Speed (km/h)', 'SINR': 'Signal Quality (dB)'})
fig_speed.write_html(os.path.join(output_folder, 'speed_impact_on_sinr.html'))
print(f"Saved speed impact plot to {output_folder}/speed_impact_on_sinr.html")
fig_speed.show()

# Does elevation affect signal power?
fig_elev = px.scatter(df, x='Elevation', y='RSRP', 
                      color='PCI',
                      title='Impact of Elevation on Signal Strength',
                      labels={'Elevation': 'Meters', 'RSRP': 'Signal Power (dBm)'})
fig_elev.write_html(os.path.join(output_folder, 'elevation_impact_on_rsrp.html'))
print(f"Saved elevation impact plot to {output_folder}/elevation_impact_on_rsrp.html")
fig_elev.show()

# --- 6. CLASSIFICATION (The "Machine Learning" prep) ---

# Let's create a target label for classification: "Good" vs "Bad" connection.
# Based on 3GPP standards:
# Good RSRP > -85 dBm AND SINR > 20 dB
conditions = [
    (df['RSRP'] >= -85) & (df['SINR'] >= 20),
    (df['RSRP'] >= -95) & (df['RSRP'] < -85),
    (df['RSRP'] < -95)
]
choices = ['Excellent', 'Moderate', 'Poor']
df['Connection_Class'] = np.select(conditions, choices, default='Poor')

# Visualize the classes on the map
fig_class = px.scatter_geo(df, lat='Latitude', lon='Longitude', color='Connection_Class',
                           scope='europe',
                           title='Drive Test Route by Connection Class',
                           color_discrete_map={'Excellent':'green', 'Moderate':'yellow', 'Poor':'red'},
                           height=500)
fig_class.update_geos(center=dict(lat=47.85, lon=13.15), projection_scale=10)
fig_class.write_html(os.path.join(output_folder, 'connection_class_map.html'))
print(f"Saved connection class map to {output_folder}/connection_class_map.html")
fig_class.show()

# Save connection class statistics
class_stats = df['Connection_Class'].value_counts().to_frame()
class_stats.to_csv(os.path.join(output_folder, 'connection_class_statistics.csv'))
print(f"Saved connection class statistics to {output_folder}/connection_class_statistics.csv")

# Save the final processed dataframe for future ML use
df.to_csv(os.path.join(output_folder, 'final_processed_data_with_classes.csv'), index=False)
print(f"Saved final processed data to {output_folder}/final_processed_data_with_classes.csv")

print(f"\n=== All analysis outputs saved to '{output_folder}' folder ===")