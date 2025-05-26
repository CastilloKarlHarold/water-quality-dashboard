import numpy as np
import streamlit as st
import base64
import os
import plotly.express as px
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from PIL import Image as PilImage, ImageDraw

# Set up page
st.set_page_config(page_title="Taalvue", layout="wide")

# --- Handle tab switching with query_params ---
params = st.query_params
active_tab = params.get("tab", "home")
st.session_state.active_tab = active_tab

# --- Custom CSS for Navbar + Layout ---
st.markdown("""
    <style>
    /* Remove default top and side paddings */
    .main .block-container {
        padding: 0;
        margin: 0;
    }

    header {visibility: hidden;}  /* Hide default Streamlit header */

    /* Sticky Top Navbar */
    .custom-navbar {
        background: linear-gradient(to right, #b2f0b2, #2e8b57);
        height: 112px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        font-family: 'Arial Black', sans-serif;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        padding: 0 3rem;
        width: 100vw;
        box-sizing: border-box;
    }

    .custom-navbar-title {
        font-size: 34px;
        font-weight: bold;
        text-shadow: 1px 1px 2px black;
    }

    .nav-buttons {
        display: flex;
        gap: 40px;
        align-items: center;
    }

    .nav-link {
        font-size: 18px;
        font-weight: 600;
        color: white;
        text-decoration: none;
        cursor: pointer;
        border: none;
        background: none;
        padding: 0;
    }

    .nav-link:hover, .active {
        text-decoration: underline;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
    }

    /* Video container */
    .video-container {
        position: fixed;
        top: 2rem; /* Allow for navbar */
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        border-radius: 0;
        box-shadow: none;
        margin-top: -1rem;
        padding: 0;
    }

    .video-container video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    </style>
""", unsafe_allow_html=True)

# --- NAVIGATION BAR ---
st.markdown(f"""
<div class="custom-navbar">
    <div class="custom-navbar-title">Taalvue</div>
    <div class="nav-buttons">
        <form action="/" method="get"><button class="nav-link {'active' if st.session_state.active_tab == 'home' else ''}" name="tab" value="home">Home</button></form>
        <form action="/" method="get"><button class="nav-link {'active' if st.session_state.active_tab == 'prediction' else ''}" name="tab" value="prediction">Dashboard</button></form>
        <form action="/" method="get"><button class="nav-link {'active' if st.session_state.active_tab == 'about' else ''}" name="tab" value="about">About</button></form>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HOME PAGE ---
class CNN_LSTM_model:
    pass


if active_tab == "home":
    def get_video_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    with st.container():
        video_file_path = "background.mp4"
        if os.path.exists(video_file_path):
            video_base64 = get_video_base64(video_file_path)

            st.markdown(f"""
                <div class="video-container">
                    <video autoplay muted loop>
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    </video>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("📹 Video not found. Please check the file path.")

# --- PREDICTION PAGE ---
elif active_tab == "prediction":
    def get_video_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    video_file_path = "about-background.mp4"
    if os.path.exists(video_file_path):
        video_base64 = get_video_base64(video_file_path)
        st.markdown(f"""
            <div class="video-container">
                <video autoplay muted loop>
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(255, 255, 255, 0.6);
                    z-index: 1;">
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <style>
                .section-box {
                    background-color: #064635;
                    color: white;
                    padding: 0.75rem 1.25rem;
                    border-radius: 12px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin: 2rem 0 1rem 0;
                }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-box" style="font-size: 1.6rem; text-align: center;">
            💧 Exploratory Data Analysis
        </div>
    """, unsafe_allow_html=True)

    # Load CSV files from local directory
    data_path = "final_cleaned_dataset.csv"
    wqi_path = "wqi_prediction.csv"

    try:
        data_df = pd.read_csv(data_path)
        data_df['Date'] = pd.to_datetime(
            data_df['Year'].astype(str) + '-' + data_df['Month'].astype(str).str.zfill(2) + '-01'
        )
        wqi_df = pd.read_csv(wqi_path)
        if 'Date' in wqi_df.columns:
            wqi_df['Date'] = pd.to_datetime(wqi_df['Date'], errors='coerce')
        else:
            st.error("🚨 'Date' column not found in WQI CSV.")
            st.stop()
    except FileNotFoundError:
        st.error("🚨 One or both CSV files not found. Please ensure 'final_cleaned_dataset.csv' and 'wqi_prediction.csv' exist in the app directory.")
        st.stop()

    # Classify WQI
    def classify_wqi(wqi):
        if wqi >= 90:
            return "Excellent"
        elif wqi >= 70:
            return "Good"
        elif wqi >= 50:
            return "Medium"
        elif wqi >= 25:
            return "Poor"
        else:
            return "Very Poor"

    wqi_df['WQI_Class'] = wqi_df['WQI'].apply(classify_wqi)

    # Sidebar: site selection
    st.sidebar.header("🔎 Filter Data")
    all_sites = data_df["Site"].unique()
    selected_sites = st.sidebar.multiselect("Select Site(s)", all_sites, default=all_sites)

    filtered_df = data_df[data_df["Site"].isin(selected_sites)]
    filtered_wqi = wqi_df[wqi_df["Site"].isin(selected_sites)]

    # --- CORRELATION HEATMAP ---
    st.markdown('<div class="section-box">🔗 Parameter Correlation Heatmap</div>', unsafe_allow_html=True)

    # Select only numeric columns for correlation (excluding 'Date' and 'Site')
    numeric_columns = filtered_df.select_dtypes(include='number').columns
    correlation_matrix = filtered_df[numeric_columns].corr()

    heatmap_fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='Viridis',
        title="Parameter Correlation Matrix",
        labels=dict(color="Correlation"),
        aspect="auto"
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # --- PARAMETER VISUALS ---
    st.markdown('<div class="section-box">📊 Parameter Distribution</div>', unsafe_allow_html=True)
    parameter = st.selectbox("Select Parameter", data_df.columns.difference(["Date", "Site"]))
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(filtered_df, x="Site", y=parameter, title=f"{parameter} Bar Plot by Site"), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(filtered_df, x="Site", y=parameter, title=f"{parameter} Box Plot by Site"), use_container_width=True)

    # --- WQI TREND ---
    st.markdown('<div class="section-box">📈 WQI Over Time</div>', unsafe_allow_html=True)
    time_granularity = st.selectbox("View By", ["Week", "Month", "Year", "All"])
    if time_granularity != "All":
        freq_map = {"Week": "W", "Month": "M", "Year": "Y"}
        wqi_resampled = filtered_wqi.set_index("Date").groupby("Site").resample(freq_map[time_granularity])["WQI"].mean().reset_index()
    else:
        wqi_resampled = filtered_wqi.copy()

    st.plotly_chart(
        px.line(wqi_resampled, x="Date", y="WQI", color="Site", markers=True, title=f"WQI Over Time ({time_granularity})"),
        use_container_width=True
    )

    # --- WQI CLASSIFICATION ---
    st.markdown('<div class="section-box">🧪 WQI Classification Summary</div>', unsafe_allow_html=True)

    # Get the latest WQI data per site
    latest = filtered_wqi.sort_values("Date").groupby("Site").tail(1)

    if not latest.empty and 'WQI_Class' in latest.columns:
        fig_wqi = px.bar(
            latest.sort_values("WQI", ascending=False),
            x="Site", y="WQI", color="WQI_Class",
            color_discrete_map={
                "Excellent": "#28a745",
                "Good": "#17a2b8",
                "Poor": "#fd7e14",
                "Very Poor": "#dc3545"
            },
            title="🔍 Latest WQI per Site",
            labels={"WQI": "Water Quality Index"},
            height=500
        )
        fig_wqi.update_layout(
            xaxis_title="Site",
            yaxis_title="WQI",
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_wqi, use_container_width=True)
    else:
        st.warning("⚠️ No WQI classification data available.")

    if not filtered_wqi.empty:
        st.plotly_chart(
            px.bar(filtered_wqi, x="Date", y="WQI", color="WQI_Class", title="WQI Classification Over Time",
                   color_discrete_map={
                       "Excellent": "green",
                       "Good": "lightgreen",
                       "Medium": "orange",
                       "Poor": "red",
                       "Very Poor": "darkred"
                   }),
            use_container_width=True
        )
    else:
        st.warning("No data available to plot WQI Classification Over Time.")

    # --- SCATTER PLOT ---
    st.markdown('<div class="section-box">📌 Parameter Relationship</div>', unsafe_allow_html=True)
    x_param = st.selectbox("X-axis", data_df.columns.difference(["Date", "Site"]), key="x_param")
    y_param = st.selectbox("Y-axis", data_df.columns.difference(["Date", "Site"]), key="y_param")

    if x_param == y_param:
        st.warning("⚠️ Please select two different parameters for the X and Y axes.")
    else:
        scatter_fig = px.scatter(
            filtered_df,
            x=x_param,
            y=y_param,
            color="Site",
            title=f"{x_param} vs {y_param}",
            trendline="ols"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # --- TIME SERIES LINE CHART ---
    st.markdown('<div class="section-box">⏳ Time Series by Parameter</div>', unsafe_allow_html=True)
    ts_param = st.selectbox("Choose Parameter", data_df.columns.difference(["Date", "Site"]), key="ts_param")
    ts_granularity = st.selectbox("Time Granularity", ["Week", "Month", "Year", "All"], key="ts_granularity")
    if ts_granularity != "All":
        ts_freq_map = {"Week": "W", "Month": "M", "Year": "Y"}
        ts_df = filtered_df.set_index("Date").groupby("Site").resample(ts_freq_map[ts_granularity])[ts_param].mean().reset_index()
    else:
        ts_df = filtered_df.copy()

    st.plotly_chart(px.line(ts_df, x="Date", y=ts_param, color="Site", title=f"{ts_param} Over Time"), use_container_width=True)

    st.markdown("""
            <div class="section-box" style="font-size: 1.6rem; text-align: center;">
                💧 Water Quality Prediction
            </div>
        """, unsafe_allow_html=True)

    prediction_mode = st.radio(
        "Select Prediction Mode",
        ["Water Quality Prediction & Model Comparison", "Time-based Prediction & WQI Calculation"],
        key="prediction_mode_radio"
    )

    if prediction_mode == "Water Quality Prediction & Model Comparison":
        st.subheader("🔧 Configure Prediction Options")

        col1, col2 = st.columns(2)

        with col1:
            selected_site = st.selectbox("Select Site", ["All Sites"] + sorted(data_df["Site"].unique()))
            feature_set = st.selectbox("Select Feature Set", ["Water Parameters Only", "Water + External Factors"])
            water_param = st.selectbox("Select Water Parameter to Predict",
                                       data_df.columns.difference(["Date", "Site", "Year", "Month"]))
            date_range = st.date_input("Select Date Range",
                                       [datetime(2013, 1, 1), datetime(2023, 12, 31)],
                                       format="YYYY-MM-DD")

        with col2:
            with st.container():
                st.markdown(f"""
                    <div style="background-color: #d4edda; padding: 15px; border-radius: 10px;">
                        <h4 style="color: #155724;">📋 Selected Options Summary</h4>
                        <ul style="color: #155724;">
                            <li><strong>Site:</strong> {selected_site}</li>
                            <li><strong>Feature Set:</strong> {feature_set}</li>
                            <li><strong>Parameter:</strong> {water_param}</li>
                            <li><strong>Date Range:</strong> {date_range[0]} to {date_range[1]}</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

        # Start Prediction button
        if st.button("🚀 Start Prediction"):
            # Filter data based on selection
            filtered_df = data_df.copy()
            if selected_site != "All Sites":
                filtered_df = filtered_df[filtered_df["Site"] == selected_site]

            filtered_df = filtered_df[
                (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
                (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
                ]

            filtered_df = filtered_df.sort_values("Date").reset_index(drop=True)

            if filtered_df.empty:
                st.warning("⚠️ No data available for the selected configuration.")
            else:
                # Simulate predictions (Replace this with your actual ML model predictions)
                actual = filtered_df[water_param]
                cnn_pred = actual + (actual * 0.01)  # Simulated +1%
                lstm_pred = actual + (actual * 0.005)  # Simulated +0.5%
                hybrid_pred = actual  # Assume best prediction is actual for now


                # Compute metrics (simple examples)
                def mae(a, b):
                    return (abs(a - b)).mean()


                def rmse(a, b):
                    return ((a - b) ** 2).mean() ** 0.5


                def r2(a, b):
                    return 1 - (((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


                metrics = pd.DataFrame({
                    "Model": ["CNN", "LSTM", "CNN-LSTM"],
                    "MAE": [mae(actual, cnn_pred), mae(actual, lstm_pred), mae(actual, hybrid_pred)],
                    "RMSE": [rmse(actual, cnn_pred), rmse(actual, lstm_pred), rmse(actual, hybrid_pred)],
                    "R2": [r2(actual, cnn_pred), r2(actual, lstm_pred), r2(actual, hybrid_pred)]
                })

                st.markdown('<div class="section-box">📈 Model Evaluation Metrics</div>', unsafe_allow_html=True)
                metric_fig = px.bar(metrics.melt(id_vars="Model", var_name="Metric", value_name="Value"),
                                    x="Model", y="Value", color="Metric", barmode="group",
                                    title="Evaluation Metrics: MAE, RMSE, R²")
                st.plotly_chart(metric_fig, use_container_width=True)

                # Sample predictions dataframe
                pred_df = pd.DataFrame({
                    "Date": filtered_df["Date"],
                    "Actual": actual,
                    "CNN": cnn_pred,
                    "LSTM": lstm_pred,
                    "CNN-LSTM": hybrid_pred
                })
                st.markdown('<div class="section-box">📋 Actual vs Predicted Values</div>', unsafe_allow_html=True)
                st.dataframe(pred_df)

                st.markdown('<div class="section-box">📊 Actual vs Predicted (Scatter Plot)</div>',
                            unsafe_allow_html=True)
                scatter_fig = px.scatter(pred_df, x="Actual", y=["CNN", "LSTM", "CNN-LSTM"],
                                         title="Actual vs Predicted by Model")
                st.plotly_chart(scatter_fig, use_container_width=True)

    # Time-based prediction logic
    else:
        st.subheader("🔮 Time-based Prediction & WQI Calculation")

        site_options = sorted(data_df["Site"].unique())
        selected_time_site = st.selectbox("Select Site", site_options, key="time_based_site")
        # Get available forecastable columns
        forecastable_columns = data_df.columns.difference(["Date", "Site", "Year", "Month"])

        # Safely set default to WQI only if it's available
        default_targets = ["WQI"] if "WQI" in forecastable_columns else []

        target_vars = st.multiselect("Select Parameters to Forecast",
                                     options=forecastable_columns,
                                     default=default_targets)

        if st.button("⏳ Run Time-based Prediction"):
            location_data = data_df[data_df["Site"] == selected_time_site].sort_values("Date").reset_index(drop=True)

            if location_data.empty:
                st.warning("⚠️ No data available for the selected site.")
            else:
                st.success(f"✅ Running Forecast for {selected_time_site}")

                look_back = 10
                predictions = []

                # Standardize the target variables
                scaler = MinMaxScaler()
                df_to_scale = location_data[target_vars]
                scaled_data = scaler.fit_transform(df_to_scale)

                # Initial input
                scaler = MinMaxScaler()
                df_to_scale = location_data[target_vars]
                scaled_data = scaler.fit_transform(df_to_scale)

                # Check if there's enough data before proceeding
                if scaled_data.shape[0] < look_back:
                    st.error(f"❌ Not enough data to make a forecast. At least {look_back} past records are required.")
                else:
                    last_months_data = scaled_data[-look_back:]
                    predictions = []

                    for i in range(12):
                        input_data = last_months_data[-look_back:].reshape(1, look_back, len(target_vars))
                        next_month_pred = CNN_LSTM_model.predict(input_data)[0]
                        predictions.append(next_month_pred)
                        last_months_data = np.vstack([last_months_data, next_month_pred])

                for i in range(12):
                    input_data = last_months_data[-look_back:].reshape(1, look_back, len(target_vars))
                    next_month_pred = CNN_LSTM_model.predict(input_data)[0]
                    predictions.append(next_month_pred)
                    last_months_data = np.vstack([last_months_data, next_month_pred])

                # Format predictions
                predictions_df = pd.DataFrame(predictions, columns=target_vars)

                # Inverse transform to original units
                descaled_predictors_df = pd.DataFrame(
                    scaler.inverse_transform(predictions_df),
                    columns=predictions_df.columns,
                    index=pd.date_range(start=pd.to_datetime("today"), periods=12, freq="M")
                )

                st.markdown("### 📅 Next 12 Months' Forecast")
                st.dataframe(descaled_predictors_df)

                # Evaluation metrics (optional: compare last 12 known vs predicted if real data is available)
                if len(scaled_data) >= 12:
                    Y = scaled_data  # Assuming you have a Y with same structure
                    st.markdown("### 📊 Forecasting Metrics")
                    for column in predictions_df.columns:
                        actual_values = Y[-12:, predictions_df.columns.get_loc(column)]
                        predicted_values = predictions_df[column]

                        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
                        mae = mean_absolute_error(actual_values, predicted_values)

                        st.write(f"**{column}**")
                        st.metric(label="RMSE", value=f"{rmse:.4f}")
                        st.metric(label="MAE", value=f"{mae:.4f}")

                # WQI Classification (if 'WQI' in target_vars)
                if "WQI" in target_vars:
                    st.markdown("### 🧪 WQI Classification")
                    wqi_preds = descaled_predictors_df["WQI"].values
                    predicted_classes = [classify_wqi(wqi) for wqi in wqi_preds]

                    wqi_class_df = pd.DataFrame({
                        "Date": descaled_predictors_df.index,
                        "Predicted WQI": wqi_preds,
                        "WQI Class": predicted_classes
                    })

                    st.dataframe(wqi_class_df)

                    # WQI Class Histogram
                    fig_class = px.histogram(wqi_class_df, x="WQI Class", color="WQI Class",
                                             title="WQI Forecast Distribution")
                    st.plotly_chart(fig_class)

                    # Recommendations based on final WQI class
                    st.subheader("💡 Recommendation for Last Forecast")
                    final_class = predicted_classes[-1]
                    rec_map = {
                        "Excellent": "Maintain regular monitoring and preserve current conditions.",
                        "Good": "Watch for early signs of pollution and sustain protection measures.",
                        "Medium": "Implement early interventions like wastewater treatment.",
                        "Poor": "Investigate causes and launch mitigation strategies immediately.",
                        "Very Poor": "Urgent restoration efforts required! Consider emergency environmental actions."
                    }
                    st.info(rec_map.get(final_class, "No recommendation available."))

                # CSV Export
                export_csv = descaled_predictors_df.copy()
                export_csv["Date"] = export_csv.index
                st.download_button("📥 Download Forecast CSV", export_csv.to_csv(index=False).encode("utf-8"),
                                   "forecast_12_months.csv", "text/csv")

    st.markdown("""
                <div class="section-box" style="font-size: 1.6rem; text-align: center;">
                    💧 Recommendations
                </div>
            """, unsafe_allow_html=True)


    # Define the recommendation function (if not already defined above)
    def get_wqi_recommendation(wqi_class):
        recommendations = {
            "Excellent": "✅ Maintain regular monitoring and protect the area from pollution sources.",
            "Good": "👍 Water quality is acceptable. Continue existing treatment and surveillance.",
            "Medium": "⚠️ Moderate pollution detected. Review wastewater sources and implement basic mitigation.",
            "Poor": "🚨 Water quality is poor. Immediate action needed: reduce pollutants and enhance treatment.",
            "Very Poor": "⛔ Dangerous quality. Avoid human/animal contact. Implement emergency remediation plans."
        }
        return recommendations.get(wqi_class, "No recommendation available.")


    # Show recommendations based on latest WQI class per site
    for index, row in latest.iterrows():
        site = row["Site"]
        wqi_class = row["WQI_Class"]
        recommendation = get_wqi_recommendation(wqi_class)

        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; margin-bottom: 10px;
                        border-left: 5px solid #064635; border-radius: 8px;">
                <h5 style="color:#064635; margin-bottom: 0.3rem;">
                    📍 <strong>Site:</strong> {site}
                </h5>
                <p style="margin: 0.2rem 0;"><strong>WQI Class:</strong> {wqi_class}</p>
                <p style="margin: 0.2rem 0;"><strong>Recommendation:</strong> {recommendation}</p>
            </div>
        """, unsafe_allow_html=True)

# --- ABOUT PAGE ---
elif active_tab == "about":

    def get_video_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def make_circle(image_path, size=(120, 120)):
        try:
            img = PilImage.open(image_path).convert("RGB").resize(size)
            mask = PilImage.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0) + size, fill=255)
            circular_img = PilImage.new("RGB", size)
            circular_img.paste(img, mask=mask)
            return circular_img
        except Exception as e:
            st.error(f"⚠️ Error loading image: {image_path}\n{e}")
            return None

    # Background Video
    with st.container():
        video_file_path = "about-background.mp4"
        if os.path.exists(video_file_path):
            video_base64 = get_video_base64(video_file_path)
            st.markdown(f"""
                <div class="video-container">
                    <video autoplay muted loop style="position:fixed; right:0; bottom:0; min-width:100%; min-height:100%; z-index:-1;">
                        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                    </video>
                    <div style="
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(255, 255, 255, 0.6);
                        z-index: 0;">
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("📹 Video not found. Please check the file path.")

    # Section title
    st.markdown("<h2 style='text-align: center; margin-top: 50px;'>Meet the Developers</h2>", unsafe_allow_html=True)

    developers = [
        {"name": "Atas Alex", "student_no": "202201113", "cvsu_acc": "mai.alex.atas@cvsu.edu.ph", "image": "images/atas.jpg"},
        {"name": "Castillo, Karl Harold", "student_no": "202201719", "cvsu_acc": "main.karlharold.castillo@cvsu.edu.ph", "image": "images/castillo.jpg"},
        {"name": "Magnaye, Jhenn Mariz", "student_no": "202201480", "cvsu_acc": "main.jhennmariz.magnaye@cvsu.edu.ph", "image": "images/magnaye.jpg"},
        {"name": "Politud, Ma. Nicole", "student_no": "202203673", "cvsu_acc": "main.ma.nicole.politud@cvsu.edu.ph", "image": "images/politud.jpg"},
        {"name": "Tabios, Jhaenelle Allyson", "student_no": "202200485", "cvsu_acc": "main.jhaenelleallyson.tabios@cvsu.edu.ph", "image": "images/tabios.jpg"},
    ]

    # Optional: Check if image files exist
    for dev in developers:
        if not os.path.exists(dev["image"]):
            st.warning(f"⚠️ Image not found: {dev['image']}")

    cols = st.columns(5)

    for i, col in enumerate(cols):
        dev = developers[i]
        with col:
            with st.container():
                st.markdown(
                    """
                    <div style='
                        background-color: #b6e7a0;
                        border-radius: 20px;
                        padding: 20px;
                        text-align: center;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                        margin-top: 20px;
                    '>
                    """,
                    unsafe_allow_html=True
                )

                # Circular Image
                circular_image = make_circle(dev["image"], size=(120, 120))
                if circular_image:
                    st.image(circular_image)

                # Info Text
                st.markdown(f"""
                    <h4 style='margin-top: 10px;'>{dev['name']}</h4>
                    <p style='margin: 0;'>{dev['student_no']}</p>
                    <p style='margin: 0; font-size: small;'>{dev['cvsu_acc']}</p>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)






