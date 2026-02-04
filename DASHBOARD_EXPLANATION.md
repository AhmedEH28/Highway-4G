# 📡 Drive Test Dashboard - Metrics Explanation

## Overview
This dashboard analyzes LTE (4G) network performance data collected during a drive test in Salzburg, Austria. The data captures how well a mobile device connects to cell towers while moving.

---

## 📊 Key Statistics Explained

### 1. **Total Samples: 263,827 data points**
**What it means:** 
- The dataset contains 263,827 individual measurements
- Each measurement represents one moment in time (typically 1 second apart)

**How obtained:**
- From the `Crawdad.csv` file
- Counted using `len(df)` in Python
- Each row = one GPS location + signal measurements

---

### 2. **Handovers: 9,151 events**
**What it means:**
- The phone switched between cell towers 9,151 times during the drive
- A "handover" happens when your phone disconnects from one tower and connects to another
- This is normal as you move around and towers have limited range

**How obtained:**
```python
# Detected by tracking changes in PCI (Physical Cell ID)
df['PCI_Change'] = df['PCI'].diff().fillna(0)
handover_events = df[df['PCI_Change'] != 0]
total_handovers = len(handover_events)
```

**Why it matters:**
- Too many handovers = unstable connection, dropped calls
- Smooth handovers = good network planning

---

### 3. **Connection Quality Classifications**

#### ✅ **Excellent: 20.9%**
**What it means:**
- 20.9% of the time, the connection was excellent quality
- Strong signal, low interference
- Perfect for video calls, streaming, fast downloads

**Technical criteria:**
```python
RSRP >= -85 dBm  AND  SINR >= 20 dB
```
- **RSRP ≥ -85 dBm:** Very strong signal from the tower
- **SINR ≥ 20 dB:** Very low interference/noise

---

#### ⚠️ **Moderate: 16.6%**
**What it means:**
- 16.6% of the time, the connection was decent but not great
- Acceptable for browsing, messaging, standard calls
- May struggle with HD video or large downloads

**Technical criteria:**
```python
RSRP between -95 and -85 dBm
```
- Signal is present but weaker
- Might be far from tower or obstacles in the way

---

#### ❌ **Poor: 62.5%**
**What it means:**
- 62.5% of the time, the connection was poor quality
- Weak signal, high interference
- Slow speeds, possible dropped connections

**Technical criteria:**
```python
RSRP < -95 dBm
```
- Very weak signal from tower
- Could be: far from tower, inside buildings, in valleys, or network congestion

**⚠️ Important:** 62.5% poor connections suggests this area has network coverage issues!

---

## 📶 LTE Signal Metrics Explained

### **RSRP (Reference Signal Received Power)**
**What it is:**
- Measures the **strength** of the signal from the cell tower
- Like measuring how "loud" the tower's signal is at your location

**Units:** dBm (decibel-milliwatts) - always negative numbers

**Interpretation:**
| RSRP Value | Quality | What it means |
|------------|---------|---------------|
| > -80 dBm | Excellent | Right next to tower, perfect signal |
| -80 to -90 dBm | Good | Normal coverage, good performance |
| -90 to -100 dBm | Fair | Weak signal, slower speeds |
| < -100 dBm | Poor | Very weak, connection issues likely |

**How obtained:** 
- Directly from the dataset (`RSRP` column)
- Measured by the phone's modem during the drive test

---

### **SINR (Signal-to-Interference-plus-Noise Ratio)**
**What it is:**
- Measures **signal quality** - how clear the signal is
- Compares useful signal vs. interference/noise
- Like measuring how clearly you can hear someone in a noisy room

**Units:** dB (decibels)

**Interpretation:**
| SINR Value | Quality | What it means |
|------------|---------|---------------|
| > 20 dB | Excellent | Crystal clear signal |
| 13 to 20 dB | Good | Good quality, reliable connection |
| 0 to 13 dB | Fair | Some interference, reduced performance |
| < 0 dB | Poor | Heavy interference, very slow |

**How obtained:**
- Directly from the dataset (`SINR` column)
- Measured by analyzing signal vs. interference levels

---

### **RSRQ (Reference Signal Received Quality)**
**What it is:**
- Another **quality** metric
- Indicates how "clean" the signal is
- Considers interference from other cells

**Units:** dB (decibels) - always negative

**Interpretation:**
| RSRQ Value | Quality |
|------------|---------|
| > -10 dB | Excellent |
| -10 to -15 dB | Good |
| -15 to -20 dB | Fair |
| < -20 dB | Poor |

---

## 🗼 Cell Tower Metrics

### **PCI (Physical Cell ID)**
**What it is:**
- Unique identifier for each cell tower sector
- Each tower typically has 3 sectors (covering different directions)
- Range: 0 to 503

**How used:**
- Track which tower you're connected to
- Detect handovers (when PCI changes)
- Analyze per-tower performance

---

## 📍 Location & Movement Metrics

### **GPS Coordinates**
- **Latitude & Longitude:** Your exact position during measurement
- Used to create maps showing signal strength by location

### **Speed (km/h)**
**How calculated:**
```python
# 1. Calculate distance between consecutive GPS points
distance = sqrt((lat2-lat1)² + (lon2-lon1)²) × 111,000 meters

# 2. Calculate time difference
time_diff = seconds between measurements

# 3. Calculate speed
Speed = (distance / time_diff) × 3.6  # Convert to km/h
```

**Why it matters:**
- High speeds can affect signal quality (Doppler effect)
- Network needs to handle handovers faster when moving quickly

### **Elevation (meters)**
**Why it matters:**
- Higher elevations may have better line-of-sight to towers
- Valleys and low areas may have blocked signals

---

## 🎯 Dashboard Features Explained

### **Interactive Filters**
You can adjust:
1. **RSRP Range:** Filter to show only strong or weak signal areas
2. **SINR Range:** Filter by signal quality levels
3. **Connection Quality:** Focus on Excellent/Moderate/Poor zones only

### **Geographic Maps**
- **Signal Strength Map:** Shows where signal was strong (green) or weak (red)
- **Quality Map:** Shows where connection was Excellent/Moderate/Poor

### **Time Series**
- See how signal changed over the drive duration
- Spot problem areas or times

### **Tower Analysis**
- Which towers were used most
- Which towers performed best/worst
- Handover patterns

---

## 🔍 Real-World Implications

### **For Users:**
- **20.9% Excellent:** Only 1 in 5 locations has great service
- **62.5% Poor:** More than half the area has coverage problems
- **9,151 Handovers:** Phone constantly switching towers (can drain battery)

### **For Network Engineers:**
- Need more towers or better positioning in this area
- High poor connection % indicates coverage gaps
- Many handovers suggest overlapping coverage issues

---

## 📈 How the Data Was Collected

1. **Drive Test Equipment:**
   - Smartphone or specialized device with GPS
   - Software logging signal metrics every second
   - Driving routes throughout Salzburg area

2. **Measurements Recorded:**
   - Every 1 second: GPS location, RSRP, RSRQ, SINR, PCI, Elevation
   - Total duration: ~73 hours of driving (263,827 seconds)

3. **Data Processing:**
   - Loaded from CSV file
   - Calculated derived metrics (Speed, Noise, Connection Class)
   - Cleaned and prepared for visualization

---

## 🛠️ Technical Implementation

### Connection Classification Logic
```python
if (RSRP >= -85) and (SINR >= 20):
    Connection_Class = 'Excellent'
elif (RSRP >= -95) and (RSRP < -85):
    Connection_Class = 'Moderate'
else:  # RSRP < -95
    Connection_Class = 'Poor'
```

### Statistics Calculation
```python
# Percentages
excellent_count = (df['Connection_Class'] == 'Excellent').sum()
excellent_pct = (excellent_count / total_samples) * 100

# Average values
avg_rsrp = df['RSRP'].mean()
avg_sinr = df['SINR'].mean()
avg_speed = df['Speed_kmh'].mean()
```

---

## 📝 Summary

This dashboard helps you understand:
- ✅ **Where** signal is good or bad (maps)
- ✅ **When** signal drops (time series)
- ✅ **Why** signal varies (speed, elevation, tower performance)
- ✅ **How much** of the area has good coverage (statistics)

The data reveals that Salzburg has significant coverage challenges, with only 20.9% excellent connections and 62.5% poor connections, suggesting the need for network improvements.
