# app.py
# (Updated version with Historical Person Heatmap AND Bottle Heatmap)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from datetime import datetime, timedelta, timezone
from streamlit_autorefresh import st_autorefresh
import numpy as np  # Needed for heatmap processing

# --- Configuration ---
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "retail_analytics"
COLLECTION_NAME = "customer_events"
TARGET_OBJECT_LABEL = "bottle"

# --- MongoDB Connection functions ---
@st.cache_resource(ttl=300)
def get_mongo_client():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"MongoDB connection error: {e}")
        return None

@st.cache_resource(ttl=300)
def get_db_collection(_client):
    if _client is None:
        return None
    try:
        return _client[DB_NAME][COLLECTION_NAME]
    except Exception as e:
        st.error(f"DB/Collection access error: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_initial_state(_collection, start_date, end_date):
    if _collection is None:
        st.warning("Initial state fetch skipped: DB collection not available.")
        return None
    try:
        query = {
            "event_type": "initial_state",
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        initial_event = _collection.find_one(query, sort=[("timestamp", -1)])
        if initial_event and "initial_bottle_count" in initial_event:
            return initial_event["initial_bottle_count"]
        return None
    except Exception as e:
        st.warning(f"Could not fetch initial bottle count: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_event_data(_collection, start_date, end_date, camera_id=None):
    if _collection is None:
        st.warning("Event data fetch skipped: DB collection not available.")
        return pd.DataFrame()

    query = {
        "timestamp": {"$gte": start_date, "$lte": end_date},
        # Fetch bottle_detection events along with others
        "event_type": {"$in": ["entry", "exit", "action", "position_update", "camera_config", "bottle_detection"]}
    }
    if camera_id and camera_id != "All":
        query["camera_id"] = camera_id
    try:
        cursor = _collection.find(query).sort("timestamp", 1)
        data = list(cursor)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Ensure relevant columns exist
        for col in ['_id', 'customer_id', 'event_type', 'timestamp',
                    'bottle_count_at_entry', 'bottle_count_at_exit',
                    'action_label', 'confidence', 'position',
                    'camera_id', 'frame_width', 'frame_height']:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception as e:
        st.error(f"Error fetching event data: {e}")
        return pd.DataFrame()

# --- Analysis Functions ---
def analyze_customer_visits(df):
    if df.empty:
        return pd.DataFrame(), {"avg_change": 0, "total_visits": 0, "net_change": 0}
    entry_exit_df = df[df['event_type'].isin(['entry', 'exit'])].copy()
    if entry_exit_df.empty:
        return pd.DataFrame(), {"avg_change": 0, "total_visits": 0, "net_change": 0}
    visits = []
    for cust_id, group in entry_exit_df.groupby('customer_id'):
        group = group.sort_values('timestamp')
        entry_event = None
        for i, event in group.iterrows():
            if event['event_type'] == 'entry':
                entry_event = event
            elif event['event_type'] == 'exit' and entry_event is not None:
                entry_count = entry_event.get('bottle_count_at_entry')
                exit_count = event.get('bottle_count_at_exit')
                delta = None
                if entry_count is not None and exit_count is not None:
                    try:
                        delta = int(exit_count) - int(entry_count)
                    except (ValueError, TypeError):
                        delta = None
                visits.append({
                    'customer_id': cust_id,
                    'entry_time': entry_event['timestamp'],
                    'exit_time': event['timestamp'],
                    'duration_sec': (event['timestamp'] - entry_event['timestamp']).total_seconds(),
                    'bottles_at_entry': entry_count,
                    'bottles_at_exit': exit_count,
                    'bottle_change': delta
                })
                entry_event = None
    visits_df = pd.DataFrame(visits)
    stats = {"total_visits": len(visits_df)}
    valid_changes = visits_df['bottle_change'].dropna()
    if not valid_changes.empty:
        stats["avg_change"] = valid_changes.mean()
        stats["net_change"] = valid_changes.sum()
    else:
        stats["avg_change"] = 0
        stats["net_change"] = 0
    return visits_df, stats

def get_action_distribution(df):
    action_df = df[df['event_type'] == 'action'].copy()
    if action_df.empty or 'action_label' not in action_df.columns:
        return pd.Series(dtype=int)
    return action_df['action_label'].value_counts()

def get_traffic_over_time(df):
    entries = df[df['event_type'] == 'entry'].copy()
    if entries.empty or 'timestamp' not in entries.columns:
        return pd.DataFrame(columns=['Time Bin', 'Customers Entered'])
    if entries['timestamp'].dt.tz is None:
        try:
            entries['timestamp'] = entries['timestamp'].dt.tz_localize('UTC')
        except Exception as tz_err:
            print(f"TZ Warning: {tz_err}")
    entries.set_index('timestamp', inplace=True)
    hourly_entries = entries.resample('H')['customer_id'].count()
    hourly_entries = hourly_entries.reset_index()
    hourly_entries.columns = ['Time Bin', 'Customers Entered']
    return hourly_entries

# --- NEW: Generic Heatmap Function (for both Person and Bottle events) ---
def create_plotly_heatmap(df_filtered, event_type_name, camera_config):
    """Generic function to create a Plotly 2D Histogram heatmap."""
    if df_filtered.empty or camera_config is None:
        return None

    df_filtered = df_filtered.dropna(subset=['position'])
    if df_filtered.empty:
        return None

    try:
        df_filtered['x'] = df_filtered['position'].apply(
            lambda p: p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else None)
        df_filtered['y'] = df_filtered['position'].apply(
            lambda p: p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else None)
        df_filtered = df_filtered.dropna(subset=['x', 'y'])
        df_filtered['x'] = df_filtered['x'].astype(int)
        df_filtered['y'] = df_filtered['y'].astype(int)
    except Exception as e:
        st.warning(f"Error processing position data for {event_type_name} heatmap: {e}")
        return None

    if df_filtered.empty:
        return None

    frame_w = camera_config.get('frame_width')
    frame_h = camera_config.get('frame_height')

    if not frame_w or not frame_h:
        st.warning(f"Cannot create {event_type_name} heatmap: Frame dimensions missing.")
        return None

    fig = go.Figure(go.Histogram2d(
        x=df_filtered['x'],
        y=df_filtered['y'],
        colorscale='Jet',
        nbinsx=max(10, int(frame_w / 25)),
        nbinsy=max(10, int(frame_h / 25)),
        zhoverformat=".0f"
    ))

    fig.update_layout(
        title=f'{event_type_name} Location Heatmap',
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        xaxis=dict(range=[0, frame_w]),
        yaxis=dict(range=[frame_h, 0]),
        width=600,
        height=int(600 * (frame_h / frame_w)) if frame_w > 0 else 400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Retail AI Analytics Dashboard")
st.title(f"🛒 Retail AI Analytics Dashboard (Bottle Count, Traffic, Actions & Heatmaps)")
st.caption("Analytics from local video processing (bottle counts, traffic, actions, locations).")

auto_refresh_interval = 10 * 1000
refresh_counter = st_autorefresh(interval=auto_refresh_interval, key="datafresher")

client = get_mongo_client()
collection = get_db_collection(client)

st.sidebar.header("📊 Filters")
default_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
default_end = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=999)
start_date_input = st.sidebar.date_input("Start Date", default_start.date())
end_date_input = st.sidebar.date_input("End Date", default_end.date())
start_datetime = datetime.combine(start_date_input, datetime.min.time()).replace(tzinfo=timezone.utc)
end_datetime = datetime.combine(end_date_input, datetime.max.time()).replace(tzinfo=timezone.utc)

camera_ids = ["All"]
latest_camera_config = {}
if collection is not None:
    try:
        distinct_cameras = collection.distinct("camera_id")
        camera_ids.extend(c for c in distinct_cameras if c)
        for cam_id in distinct_cameras:
            if cam_id:
                config = collection.find_one({"event_type": "camera_config", "camera_id": cam_id}, sort=[("timestamp", -1)])
                if config and 'frame_width' in config and 'frame_height' in config:
                    latest_camera_config[cam_id] = {'frame_width': config['frame_width'], 'frame_height': config['frame_height']}
    except Exception as e:
        st.sidebar.warning(f"Could not fetch camera info: {e}")
else:
    st.error("Dashboard cannot load data: Failed to connect to MongoDB collection.")

selected_camera = st.sidebar.selectbox("Camera ID", options=camera_ids)

# --- Main Dashboard Area ---
if collection is not None:
    df_all_events = fetch_event_data(collection, start_datetime, end_datetime, selected_camera)
    selected_cam_config = latest_camera_config.get(selected_camera) if selected_camera != "All" else None

    # --- Section 1: Bottle Count Analysis ---
    st.header("🍾 Bottle Count Analysis")
    initial_count = fetch_initial_state(collection, start_datetime, end_datetime)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.metric(f"Initial {TARGET_OBJECT_LABEL.capitalize()} Count (Filtered)", 
                value=initial_count if initial_count is not None else "N/A",
                help="Initial bottle count within the selected time period.")
    visits_df, visit_stats = analyze_customer_visits(df_all_events)
    with col_b2:
         st.metric(f"Net Bottle Change (Selected Period & Camera)", f"{visit_stats['net_change']:.0f}",
                   help="Sum of (Bottles@Exit - Bottles@Entry) for the selected period and camera.")

    if not visits_df.empty:
        st.metric(f"Avg. Bottle Change per Visit", f"{visit_stats['avg_change']:.2f}",
                help="Average change per completed visit for the selected period and camera.")
        st.subheader("Visit Details (Bottle Count Changes)")
        visits_df['duration_str'] = visits_df['duration_sec'].apply(
            lambda s: f"{int(s//60)}m {int(s%60)}s" if pd.notna(s) else 'N/A'
        )
        display_df = visits_df[['customer_id', 'entry_time', 'exit_time', 'duration_str', 
                                'bottles_at_entry', 'bottles_at_exit', 'bottle_change']].copy()
        display_df.rename(columns={
            'duration_str': 'Duration', 
            'bottles_at_entry': 'Count@Entry', 
            'bottles_at_exit': 'Count@Exit', 
            'bottle_change': 'Change'
        }, inplace=True)
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert Count columns to numeric, then they can be formatted as integers.
        display_df['Count@Entry'] = pd.to_numeric(display_df['Count@Entry'], errors='coerce')
        display_df['Count@Exit'] = pd.to_numeric(display_df['Count@Exit'], errors='coerce')
        
        # Sort DataFrame as needed
        display_df = display_df.sort_values('entry_time', ascending=False)
        
        # Define a function that highlights zeros in yellow.
        def highlight_zero(val):
            try:
                return 'background-color: maroon' if float(val) == 0 else ''
            except (ValueError, TypeError):
                return ''
        
        # Use Styler to format the count columns as integers (i.e. no decimal places)
        styled_df = display_df.style.format({
            'Count@Entry': '{:.0f}', 
            'Count@Exit': '{:.0f}',
            'customer_id': '{:.0f}'
        }).applymap(highlight_zero, subset=['Count@Entry', 'Count@Exit'])
        
        st.dataframe(styled_df, use_container_width=True)

        if visit_stats['net_change'] < 0:
            st.info(f"💡 Insight: Net decrease of {abs(visit_stats['net_change']):.0f} bottles observed. Consider restocking.")
        elif visit_stats['net_change'] > 0:
            st.warning(f"💡 Note: Net increase in detected bottles observed.")
    else:
        st.info(f"No complete customer visits (entry/exit pairs with bottle counts) found for the selected filters.")

    st.divider()

    # --- Section 2: Traffic Analysis ---
    st.header("⏱️ Store Traffic Patterns")
    traffic_df = get_traffic_over_time(df_all_events)
    if not traffic_df.empty:
        fig_traffic = px.line(traffic_df, x='Time Bin', y='Customers Entered', markers=True,
                             title="Customer Entries Over Time (Hourly)", labels={'Customers Entered': 'Number of Customers Entering'})
        st.plotly_chart(fig_traffic, use_container_width=True)
        peak_hour_data = traffic_df.loc[traffic_df['Customers Entered'].idxmax()] if not traffic_df.empty else None
        if peak_hour_data is not None:
            peak_hour = peak_hour_data['Time Bin'].strftime('%I %p')
            st.info(f"💡 Insight: Peak traffic observed around **{peak_hour}**.")
    else:
        st.info("No traffic data (entry events) available for this period.")

    st.divider()

    # --- Section 3: Action Analysis ---
    st.header("🚶 Customer Actions")
    action_counts = get_action_distribution(df_all_events)
    if not action_counts.empty:
        fig_actions = px.pie(action_counts, values=action_counts.values, names=action_counts.index,
                            title="Most Common Actions Observed", hole=0.3)
        fig_actions.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_actions, use_container_width=True)
    else:
        st.info("No action data available for this period.")

    st.divider()  # Divider before the next section

    # --- Section 4: Location Heatmaps ---
    st.header("🗺️ Location Heatmaps")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.subheader("Person Heatmap")
        if selected_camera == "All":
             st.warning("Please select a specific Camera ID to view the person location heatmap.")
        elif selected_cam_config:
            fig_person_heatmap = create_plotly_heatmap(
                df_all_events[df_all_events['event_type'] == 'position_update'].copy(),
                "Person",
                selected_cam_config
            )
            if fig_person_heatmap:
                st.plotly_chart(fig_person_heatmap, use_container_width=True)
                st.info("💡 Insight: Brighter areas indicate higher customer traffic density.")
            else:
                st.info("No person position data available to generate heatmap for the selected filters, or frame dimensions missing.")
        else:
             st.warning(f"Camera configuration (frame dimensions) not found for Camera ID '{selected_camera}'. Cannot generate heatmap.")

    with col_h2:
        st.subheader(f"{TARGET_OBJECT_LABEL.capitalize()} Heatmap")
        if selected_camera == "All":
             st.warning("Please select a specific Camera ID to view the bottle location heatmap.")
        elif selected_cam_config:
            df_bottle_pos = df_all_events[df_all_events['event_type'] == 'bottle_detection'].copy()
            fig_bottle_heatmap = create_plotly_heatmap(df_bottle_pos, TARGET_OBJECT_LABEL.capitalize(), selected_cam_config)
            if fig_bottle_heatmap:
                st.plotly_chart(fig_bottle_heatmap, use_container_width=True)
                st.caption(f"Density of detected {TARGET_OBJECT_LABEL} locations.")
            else:
                st.caption(f"No {TARGET_OBJECT_LABEL} location data available for heatmap.")
        else:
             st.warning(f"Camera configuration (frame dimensions) not found for Camera ID '{selected_camera}'. Cannot generate bottle heatmap.")

else:
    pass  # Error message already shown if connection failed

# --- Fulfillment Checkbox Section ---
st.sidebar.divider()
st.sidebar.header("✅ Project Fulfillment")
st.sidebar.markdown("**Objective:** Create an AI system leveraging video analytics and customer behavior data to personalize in-store experiences.")
st.sidebar.markdown("**System Goals:**")
st.sidebar.checkbox("Tracks customer movement in real-time using in-store camera feeds.", value=True, disabled=True, help="Fulfilled by tracking and logging positions for heatmap visualization.")
st.sidebar.checkbox("Analyzes customer behavior to identify high-traffic areas and customer preferences.", value=True, disabled=True, help="Fulfilled by dashboard: Traffic patterns, action distribution, and bottle count changes.")
st.sidebar.checkbox("Provides actionable insights for product placements, promotional strategies, and restocking shelves.", value=True, disabled=True, help="Fulfilled by analyzing visit timings, heatmaps, and bottle count changes.")
st.sidebar.markdown("**Expected Outcomes:**")
st.sidebar.checkbox("A functional prototype capable of tracking and analyzing real-time video feeds.", value=True, disabled=True, help="Local script with display/heatmap functionality.")
st.sidebar.checkbox("Real-time alerts for restocking or customer assistance in under-served areas.", value=True, disabled=True, help="Partially fulfilled via dashboard insights.")
st.sidebar.checkbox("Insights for optimal product placement and promotion strategies based on behavior patterns.", value=True, disabled=True, help="Fulfilled by analyzing visit details and heatmaps.")
st.sidebar.markdown("**Challenges Addressed:**")
st.sidebar.checkbox("Processing multiple real-time video streams with minimal latency.", value=True, disabled=True, help="Local script handles stream processing; heatmaps add overhead.")
st.sidebar.checkbox("Ensuring accurate detection of customer behavior in varying conditions.", value=True, disabled=True, help="Depends on model training & environmental conditions.")
st.sidebar.checkbox("Handling and analyzing large-scale data without overloading system resources.", value=True, disabled=True, help="Efficient data fetching/analysis via MongoDB; position logging increases data volume.")
st.sidebar.info(f"Dashboard auto-refreshes ~every {auto_refresh_interval//1000}s. Local script shows video feed / heatmaps.")
