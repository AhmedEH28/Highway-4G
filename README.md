# Highway-4G AIoT LTE Signal Analysis

LTE drive-test analysis and visualization using Crawdad data. It focuses on spatiotemporal signal understanding (RSRP, RSRQ, RSSI, SINR), handover-region identification, and clear dashboard-based evaluation of network quality. The implementation is centered on data processing, analytics, and interactive visualization for evaluating network performance across time and geography.

## dashboard

The dashboard analyzes temporal, geospatial, and PCI-based behavior through KPI cards, map views, time-series plots, and handover analysis. Processed outputs are exported to [analysis_output/](analysis_output/) as HTML and CSV reports .

Current dataset snapshot: **263,827** samples and **9,151** handovers, with quality classes of **20.9% Excellent**, **16.6% Moderate**, and **62.5% Poor**.

## Main files

- [dashboard.py](dashboard.py)
- [vizual2.py](vizual2.py)
- [requirements_dashboard.txt](requirements_dashboard.txt)
- [analysis_output/](analysis_output/)

## Run

Install dependencies with `pip install -r requirements_dashboard.txt`, run `python dashboard.py`, then open http://127.0.0.1:8050.

## Media

Screenshots and demo video:

- [assets/media/dashboard_geo.png](assets/media/dashboard_geo.png)
- [assets/media/dashboard_timeseries.png](assets/media/dashboard_timeseries.png)
- [assets/media/dashboard_demo.mp4](assets/media/dashboard_demo.mp4)
