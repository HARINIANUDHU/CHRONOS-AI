
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="STORM AI | Weather Anomaly Intelligence",
    page_icon="🌩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #e8f1f8;
}
section[data-testid="stSidebar"] {
    background: #0a1728;
    border-right: 1px solid #1c3854;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    letter-spacing: 0.5px;
}
.metric-card {
    background: linear-gradient(135deg, #0c2035, #102a44);
    border: 1px solid #244663;
    border-radius: 14px;
    padding: 16px;
    min-height: 115px;
}
.metric-title {
    color: #8ca9bf;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    color: #f2f8fc;
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 8px;
}
.metric-sub {
    color: #66d9c5;
    font-size: 0.78rem;
    margin-top: 5px;
}
.panel {
    background: #0b1b2d;
    border: 1px solid #1c3854;
    border-radius: 14px;
    padding: 14px 16px;
}
.alert {
    background: #35181b;
    border-left: 4px solid #ff5c67;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 9px;
}
.info {
    background: #102c3a;
    border-left: 4px solid #36c7d4;
    padding: 12px;
    border-radius: 8px;
}
.small {
    color: #8ca9bf;
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# REAL WEATHER API DATA
# -----------------------------

import requests

API_URL = "https://api.open-meteo.com/v1/forecast"

LOCATIONS = [
    ("Chennai", 13.0827, 80.2707),
    ("Puducherry", 11.9416, 79.8083),
    ("Cuddalore", 11.7480, 79.7714),
    ("Nagapattinam", 10.7667, 79.8420),
    ("Madurai", 9.9252, 78.1198),
    ("Coimbatore", 11.0168, 76.9558),
    ("Bengaluru", 12.9716, 77.5946),
    ("Visakhapatnam", 17.6868, 83.2185),
    ("Kolkata", 22.5726, 88.3639),
    ("Mumbai", 19.0760, 72.8777),
    ("Hyderabad", 17.3850, 78.4867),
    ("Delhi", 28.6139, 77.2090),
]


@st.cache_data(ttl=1800)
def fetch_city_forecast(city, lat, lon):

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "precipitation_probability",
            "wind_speed_10m",
            "surface_pressure",
        ],
        "forecast_hours": 120,
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    hourly = response.json()["hourly"]

    return pd.DataFrame({
        "Time": pd.to_datetime(hourly["time"]),
        "Temperature": hourly["temperature_2m"],
        "Humidity": hourly["relative_humidity_2m"],
        "Rainfall": hourly["precipitation"],
        "RainProbability": hourly[
            "precipitation_probability"
        ],
        "Wind": hourly["wind_speed_10m"],
        "Pressure": hourly["surface_pressure"],
    })


@st.cache_data(ttl=1800)
def make_data():

    rows = []

    for city, lat, lon in LOCATIONS:

        weather = fetch_city_forecast(
            city, lat, lon
        )

        for lead in [24, 48, 72, 96, 120]:

            period = weather.iloc[:lead]

            if period.empty:
                continue

            rows.append({
                "City": city,
                "Latitude": lat,
                "Longitude": lon,
                "LeadHour": lead,

                "Rainfall": period[
                    "Rainfall"
                ].sum(),

                "Temperature": period[
                    "Temperature"
                ].mean(),

                "Wind": period["Wind"].mean(),

                "Pressure": period[
                    "Pressure"
                ].mean(),

                "Humidity": period[
                    "Humidity"
                ].mean(),

                "RainProbability": period[
                    "RainProbability"
                ].max(),
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No weather forecast data received"
        )

    # Relative anomaly calculation

    df["RainAnomaly"] = 0.0
    df["TempAnomaly"] = 0.0
    df["WindAnomaly"] = 0.0

    for lead in df["LeadHour"].unique():

        mask = df["LeadHour"] == lead
        subset = df.loc[mask]

        for column, anomaly_column in [
            ("Rainfall", "RainAnomaly"),
            ("Temperature", "TempAnomaly"),
            ("Wind", "WindAnomaly"),
        ]:

            reference = subset[column].median()

            if column == "Temperature":

                df.loc[mask, anomaly_column] = (
                    subset[column] - reference
                )

            elif abs(reference) > 1e-6:

                df.loc[mask, anomaly_column] = (
                    (subset[column] - reference)
                    / abs(reference)
                ) * 100

    # Isolation Forest model

    features = df[
        [
            "Rainfall",
            "Temperature",
            "Wind",
            "Pressure"
        ]
    ].fillna(0)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.15,
        random_state=42
    )

    raw = model.fit_predict(features)

    score = -model.score_samples(features)

    score_min = score.min()
    score_max = score.max()

    score = 100 * (
        (score - score_min)
        / (score_max - score_min + 1e-9)
    )

    df["AI_Anomaly_Score"] = score

    df["AI_Flag"] = np.where(
        (raw == -1)
        | (df["RainAnomaly"] > 120)
        | (df["WindAnomaly"] > 60),
        "EXTREME",
        "NORMAL"
    )

    df["Severity"] = np.select(
        [
            df["AI_Anomaly_Score"] >= 75,
            df["AI_Anomaly_Score"] >= 50,
            df["AI_Anomaly_Score"] >= 30,
        ],
        [
            "Extreme",
            "High",
            "Moderate"
        ],
        default="Low"
    )

    return df


# Load real forecast data

try:
    df = make_data()

except Exception as e:
    st.error(f"Weather API error: {e}")
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 🌩️ STORM AI")
st.sidebar.caption("Spatio-Temporal Weather Anomaly Intelligence")

st.sidebar.markdown("### Forecast Controls")
lead = st.sidebar.select_slider(
    "Forecast lead time",
    options=[24, 48, 72, 96, 120],
    value=48,
    format_func=lambda x: f"T+{x} hours"
)

variable = st.sidebar.selectbox(
    "Primary anomaly layer",
    ["Rainfall", "Temperature", "Wind"]
)

region = st.sidebar.selectbox(
    "Focus region",
    ["All India", "Tamil Nadu & Coast", "South India"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.success("Isolation Forest baseline: ACTIVE")
st.sidebar.caption("Demo model. Replace with trained spatio-temporal model for production.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Data status")
st.sidebar.write("🟢 Forecast feed: DEMO DATA")
st.sidebar.write("🟢 AI engine: RUNNING")
st.sidebar.write("🟡 Live radar: Prototype")

# -----------------------------
# Header
# -----------------------------
st.markdown("# 🌩️ STORM AI")
st.markdown(
    "**AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts**"
)
st.markdown(
    '<div class="info">AI analyzes forecast variables against reference conditions, '
    'flags unusual weather patterns, and visualizes where and when anomalies may evolve.</div>',
    unsafe_allow_html=True
)

view = df[df["LeadHour"] == lead].copy()

# -----------------------------
# KPI cards
# -----------------------------
extreme_count = int((view["Severity"] == "Extreme").sum())
high_count = int((view["Severity"] == "High").sum())
max_score = view["AI_Anomaly_Score"].max()
max_rain = view["Rainfall"].max()

c1, c2, c3, c4 = st.columns(4)
cards = [
    ("AI ANOMALIES", str(extreme_count + high_count), "High + Extreme zones"),
    ("EXTREME ZONES", str(extreme_count), "AI flagged"),
    ("MAX RAINFALL", f"{max_rain:.0f} mm", f"T+{lead}h forecast"),
    ("AI CONFIDENCE", f"{max_score:.0f}/100", "Anomaly intensity score"),
]
for col, (title, value, sub) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

st.write("")

# -----------------------------
# Main map
# -----------------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown("### 🗺️ Spatio-Temporal Anomaly Map")
    st.caption(f"Forecast view: T+{lead} hours • Layer: {variable}")

    color_values = {
        "Rainfall": view["RainAnomaly"],
        "Temperature": view["TempAnomaly"],
        "Wind": view["WindAnomaly"],
    }[variable]

    fig = go.Figure()

    # Base locations
    fig.add_trace(go.Scattergeo(
        lat=view["Latitude"],
        lon=view["Longitude"],
        text=view["City"],
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=np.clip(view["AI_Anomaly_Score"] / 2.5 + 8, 9, 38),
            color=color_values,
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Anomaly"),
            line=dict(width=1, color="#dceaf4"),
            opacity=0.9,
        ),
        customdata=np.stack([
            view["City"],
            view["Rainfall"].round(1),
            view["Temperature"].round(1),
            view["Wind"].round(1),
            view["AI_Anomaly_Score"].round(1),
            view["Severity"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Rainfall: %{customdata[1]} mm<br>"
            "Temperature: %{customdata[2]} °C<br>"
            "Wind: %{customdata[3]} km/h<br>"
            "AI score: %{customdata[4]}<br>"
            "Severity: %{customdata[5]}<extra></extra>"
        ),
        name="Forecast anomalies"
    ))

    fig.update_geos(
        scope="asia",
        center=dict(lat=17, lon=80),
        projection_scale=4.5,
        showland=True,
        landcolor="#0e2236",
        showocean=True,
        oceancolor="#06101c",
        showcountries=True,
        countrycolor="#35546e",
        coastlinecolor="#4c708d",
        showlakes=True,
        lakecolor="#081827",
        bgcolor="#07111f",
    )
    
    fig.update_layout(
        geo=dict(
        scope="asia",
        projection_type="mercator",
        center=dict(lat=22, lon=79),
        lataxis_range=[5, 38],
        lonaxis_range=[65, 100],
        showland=True,
        landcolor="#172554",
        showocean=True,
        oceancolor="#07111f",
        showcountries=True,
        countrycolor="#64748b",
        showcoastlines=True,
        coastlinecolor="#94a3b8",
        bgcolor="#07111f",
            ),
            paper_bgcolor="#07111f",
            plot_bgcolor="#07111f",
            font=dict(color="#e8f1f8"),
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
        )
st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 🚨 AI Alerts")
    alerts = view.sort_values("AI_Anomaly_Score", ascending=False).head(5)

    for _, r in alerts.iterrows():
        cls = "alert" if r["Severity"] in ["Extreme", "High"] else "info"
        st.markdown(
            f'<div class="{cls}"><b>{r["Severity"].upper()} • {r["City"]}</b><br>'
            f'{variable} anomaly • AI score {r["AI_Anomaly_Score"]:.0f}/100<br>'
            f'<span class="small">Forecast lead: T+{lead}h</span></div>',
            unsafe_allow_html=True
        )

# -----------------------------
# Forecast evolution
# -----------------------------
st.markdown("### 📈 Temporal Evolution")

selected_city = st.selectbox(
    "Track anomaly for a location",
    sorted(df["City"].unique()),
    index=sorted(df["City"].unique()).index("Chennai")
)

city_df = df[df["City"] == selected_city].sort_values("LeadHour")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=city_df["LeadHour"],
    y=city_df["Rainfall"],
    mode="lines+markers",
    name="Forecast rainfall",
    line=dict(width=3),
))
fig2.add_trace(go.Scatter(
    x=city_df["LeadHour"],
    y=city_df["AI_Anomaly_Score"],
    mode="lines+markers",
    name="AI anomaly score",
    yaxis="y2",
    line=dict(width=3, dash="dot"),
))
fig2.update_layout(
    height=350,
    margin=dict(l=10, r=10, t=20, b=10),
    paper_bgcolor="#0b1b2d",
    plot_bgcolor="#0b1b2d",
    font=dict(color="#d9e8f2"),
    xaxis=dict(title="Forecast lead time (hours)", gridcolor="#1d354b"),
    yaxis=dict(title="Rainfall (mm)", gridcolor="#1d354b"),
    yaxis2=dict(title="AI anomaly score", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Detail table
# -----------------------------
st.markdown("### 📋 Forecast Anomaly Details")
display_cols = [
    "City", "LeadHour", "Rainfall", "Temperature", "Wind",
    "Pressure", "RainAnomaly", "AI_Anomaly_Score", "Severity"
]
detail = view[display_cols].copy()
detail.columns = [
    "Location", "Lead (h)", "Rain (mm)", "Temp (°C)", "Wind (km/h)",
    "Pressure (hPa)", "Rain anomaly (%)", "AI score", "Severity"
]
st.dataframe(
    detail.sort_values("AI score", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.caption(
    "⚠️ SIH prototype: values shown here are synthetic demonstration data. "
    "Do not use this dashboard for real-world weather warnings. "
    "Production deployment requires validated forecast datasets, model evaluation, "
    "uncertainty estimates, and operational meteorological sources."
)
