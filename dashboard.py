"""
ULD Forecasting Dashboard

Interactive dashboard for Delta Airlines ULD operations.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
from datetime import datetime, timedelta
import asyncio

# Page config
st.set_page_config(
    page_title="ULD Forecasting System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1E3A5F;
    }
</style>
""", unsafe_allow_html=True)


def fetch_data(endpoint: str) -> dict | list | None:
    """Fetch data from API."""
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{API_URL}{endpoint}")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


def post_data(endpoint: str, data: dict = None) -> dict | None:
    """Post data to API."""
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{API_URL}{endpoint}", json=data or {})
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Delta_Air_Lines_logo.svg/1200px-Delta_Air_Lines_logo.svg.png", width=150)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "📍 Network Map", "📈 Forecasting", "🔄 Recommendations", "⚙️ Optimization"],
        index=0
    )

    st.markdown("---")

    # Health check
    health = fetch_data("/health")
    if health and health.get("status") == "healthy":
        st.success("✅ System Online")
    else:
        st.error("❌ System Offline")


# Main content
if page == "📊 Dashboard":
    st.markdown('<p class="main-header">✈️ ULD Forecasting System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Delta Air Lines - Unit Load Device Operations</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    stations = fetch_data("/api/v1/stations/")
    network_summary = fetch_data("/api/v1/tracking/network")
    recommendations = fetch_data("/api/v1/recommendations/repositioning")

    with col1:
        total_stations = stations.get("total", 0) if stations else 0
        st.metric("Stations", total_stations, "Delta Network")

    with col2:
        hubs = fetch_data("/api/v1/stations/hubs")
        hub_count = hubs.get("total", 0) if hubs else 0
        st.metric("Hub Stations", hub_count, "Primary Hubs")

    with col3:
        if network_summary:
            total_ulds = sum(s.get("total", 0) for s in network_summary.values())
            st.metric("Total ULDs", f"{total_ulds:,}", "Network-wide")
        else:
            st.metric("Total ULDs", "—", "Loading...")

    with col4:
        rec_count = len(recommendations) if recommendations else 0
        st.metric("Active Recommendations", rec_count, "Pending Actions")

    st.markdown("---")

    # Two columns for charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Station Inventory Overview")
        if network_summary:
            df = pd.DataFrame([
                {"Station": k, "Total": v.get("total", 0), "Available": v.get("available", 0)}
                for k, v in network_summary.items()
            ])
            df = df.sort_values("Total", ascending=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df["Station"],
                x=df["Total"],
                name="Total",
                orientation="h",
                marker_color="#1E3A5F"
            ))
            fig.add_trace(go.Bar(
                y=df["Station"],
                x=df["Available"],
                name="Available",
                orientation="h",
                marker_color="#28a745"
            ))
            fig.update_layout(
                barmode="overlay",
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading inventory data...")

    with col2:
        st.subheader("🎯 Availability Ratio by Station")
        if network_summary:
            df = pd.DataFrame([
                {"Station": k, "Availability": v.get("availability_ratio", 0) * 100}
                for k, v in network_summary.items()
            ])
            df = df.sort_values("Availability", ascending=False)

            fig = px.bar(
                df,
                x="Station",
                y="Availability",
                color="Availability",
                color_continuous_scale=["#dc3545", "#ffc107", "#28a745"],
                range_color=[0, 100]
            )
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )
            fig.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Target: 70%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Loading availability data...")


elif page == "📍 Network Map":
    st.markdown("## 📍 Delta Network Map")
    st.markdown("---")

    stations = fetch_data("/api/v1/stations/")

    if stations:
        # Station coordinates (approximate)
        coords = {
            "ATL": (33.6407, -84.4277),
            "DTW": (42.2124, -83.3534),
            "MSP": (44.8820, -93.2218),
            "SLC": (40.7884, -111.9778),
            "JFK": (40.6413, -73.7781),
            "LAX": (33.9416, -118.4085),
            "SEA": (47.4502, -122.3088),
            "BOS": (42.3656, -71.0096),
        }

        network_summary = fetch_data("/api/v1/tracking/network")

        df = pd.DataFrame([
            {
                "Station": s["code"],
                "Name": s["name"],
                "Tier": s["tier"],
                "lat": coords.get(s["code"], (0, 0))[0],
                "lon": coords.get(s["code"], (0, 0))[1],
                "ULDs": network_summary.get(s["code"], {}).get("total", 50) if network_summary else 50,
                "Availability": network_summary.get(s["code"], {}).get("availability_ratio", 0.7) * 100 if network_summary else 70
            }
            for s in stations.get("stations", [])
            if s["code"] in coords
        ])

        fig = px.scatter_geo(
            df,
            lat="lat",
            lon="lon",
            size="ULDs",
            color="Tier",
            hover_name="Station",
            hover_data=["Name", "ULDs", "Availability"],
            color_discrete_map={"hub": "#1E3A5F", "focus_city": "#667eea"},
            scope="usa",
            size_max=40
        )
        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                showland=True,
                landcolor="rgb(243, 243, 243)",
                countrycolor="rgb(204, 204, 204)",
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Station list
        st.subheader("Station Details")
        st.dataframe(
            df[["Station", "Name", "Tier", "ULDs", "Availability"]].sort_values("ULDs", ascending=False),
            use_container_width=True,
            hide_index=True
        )


elif page == "📈 Forecasting":
    st.markdown("## 📈 Demand Forecasting")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        stations = fetch_data("/api/v1/stations/")
        station_codes = [s["code"] for s in stations.get("stations", [])] if stations else ["ATL"]
        selected_station = st.selectbox("Select Station", station_codes, index=0)

    with col2:
        hours_ahead = st.slider("Forecast Horizon (hours)", 6, 48, 24)

    with col3:
        granularity = st.selectbox("Granularity", ["hourly", "daily"], index=0)

    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Demand forecast
            demand = fetch_data(f"/api/v1/forecasting/demand/{selected_station}?hours_ahead={hours_ahead}&granularity={granularity}")
            supply = fetch_data(f"/api/v1/forecasting/supply/{selected_station}?hours_ahead={hours_ahead}&granularity={granularity}")
            imbalance = fetch_data(f"/api/v1/forecasting/imbalance/{selected_station}?hours_ahead={hours_ahead}&granularity={granularity}")

            if demand:
                st.subheader(f"📊 {selected_station} Demand Forecast")

                df = pd.DataFrame(demand)
                df["forecast_time"] = pd.to_datetime(df["forecast_time"])

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                   subplot_titles=("Demand Forecast with Confidence Intervals", "Supply vs Demand"))

                # Demand with confidence interval
                fig.add_trace(go.Scatter(
                    x=df["forecast_time"],
                    y=df["q95"],
                    fill=None,
                    mode="lines",
                    line_color="rgba(30, 58, 95, 0.1)",
                    showlegend=False
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df["forecast_time"],
                    y=df["q05"],
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(30, 58, 95, 0.1)",
                    fillcolor="rgba(30, 58, 95, 0.2)",
                    name="90% CI"
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df["forecast_time"],
                    y=df["q50"],
                    mode="lines+markers",
                    line_color="#1E3A5F",
                    name="Demand (median)"
                ), row=1, col=1)

                # Supply vs Demand
                if supply:
                    supply_df = pd.DataFrame(supply)
                    supply_df["forecast_time"] = pd.to_datetime(supply_df["forecast_time"])

                    fig.add_trace(go.Scatter(
                        x=df["forecast_time"],
                        y=df["q50"],
                        mode="lines",
                        name="Demand",
                        line_color="#dc3545"
                    ), row=2, col=1)

                    fig.add_trace(go.Scatter(
                        x=supply_df["forecast_time"],
                        y=supply_df["q50"],
                        mode="lines",
                        name="Supply",
                        line_color="#28a745"
                    ), row=2, col=1)

                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

                # Imbalance alerts
                if imbalance:
                    st.subheader("⚠️ Imbalance Alerts")
                    imb_df = pd.DataFrame(imbalance)
                    critical = imb_df[imb_df["q50"] < -10]
                    if len(critical) > 0:
                        st.error(f"🚨 {len(critical)} periods with potential shortage (demand > supply)")
                    else:
                        st.success("✅ No critical imbalances forecast")


elif page == "🔄 Recommendations":
    st.markdown("## 🔄 Repositioning Recommendations")
    st.markdown("---")

    recommendations = fetch_data("/api/v1/recommendations/repositioning")

    if recommendations:
        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            critical = len([r for r in recommendations if r["priority"] == "critical"])
            st.metric("Critical", critical, "Immediate action")

        with col2:
            high = len([r for r in recommendations if r["priority"] == "high"])
            st.metric("High Priority", high, "Within 24h")

        with col3:
            total_ulds = sum(r["quantity"] for r in recommendations)
            st.metric("Total ULDs to Move", total_ulds, "Across network")

        st.markdown("---")

        # Recommendations table
        st.subheader("📋 Active Recommendations")

        df = pd.DataFrame(recommendations)
        if not df.empty:
            df["required_by"] = pd.to_datetime(df["required_by"]).dt.strftime("%Y-%m-%d %H:%M")
            df["shortage_probability"] = (df["shortage_probability"] * 100).round(1).astype(str) + "%"

            # Color code by priority
            def highlight_priority(row):
                if row["priority"] == "critical":
                    return ["background-color: #ffcccc"] * len(row)
                elif row["priority"] == "high":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            display_cols = ["priority", "origin", "destination", "uld_type", "quantity", "required_by", "shortage_probability", "reason"]
            st.dataframe(
                df[display_cols].style.apply(highlight_priority, axis=1),
                use_container_width=True,
                hide_index=True
            )

            # Flow visualization
            st.subheader("🔀 Movement Flow")

            # Create Sankey diagram
            origins = df["origin"].tolist()
            destinations = df["destination"].tolist()
            quantities = df["quantity"].tolist()

            all_nodes = list(set(origins + destinations))
            node_indices = {node: i for i, node in enumerate(all_nodes)}

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="#1E3A5F"
                ),
                link=dict(
                    source=[node_indices[o] for o in origins],
                    target=[node_indices[d] for d in destinations],
                    value=quantities,
                    color="rgba(30, 58, 95, 0.4)"
                )
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No active recommendations at this time.")


elif page == "⚙️ Optimization":
    st.markdown("## ⚙️ Network Optimization")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        horizon = st.slider("Optimization Horizon (hours)", 12, 72, 24)
        max_moves = st.slider("Max Moves per Station", 1, 10, 5)

    with col2:
        st.subheader("Constraints")
        st.checkbox("Respect hub capacity limits", value=True)
        st.checkbox("Minimize deadheading cost", value=True)
        st.checkbox("Prioritize shortage prevention", value=True)

    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Running network optimization..."):
            result = post_data("/api/v1/recommendations/optimize", {
                "horizon_hours": horizon,
                "max_moves_per_station": max_moves
            })

            if result:
                st.success("✅ Optimization Complete!")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Moves", result.get("total_moves", 0))

                with col2:
                    st.metric("ULDs Moved", result.get("total_ulds_moved", 0))

                with col3:
                    st.metric("Total Cost", f"${result.get('total_cost', 0):,.0f}")

                with col4:
                    st.metric("Solve Time", f"{result.get('solve_time_seconds', 0):.2f}s")

                # Show moves
                moves = result.get("moves", [])
                if moves:
                    st.subheader("📦 Recommended Moves")
                    moves_df = pd.DataFrame(moves)
                    st.dataframe(moves_df, use_container_width=True, hide_index=True)
            else:
                st.error("Optimization failed. Check API logs.")


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        ULD Forecasting System v1.0 | Delta Air Lines Operations |
        <a href="http://localhost:8000/docs" target="_blank">API Docs</a>
    </div>
    """,
    unsafe_allow_html=True
)
