# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load pretrained model and preprocessor
pipeline = joblib.load('model/pipeline.pkl')

# Load data for prediction and visualization
df = pd.read_csv("data/cleaned_properties.csv")

# Load evaluation data (used in model evaluation tab)
test_df = pd.read_csv("data/test_properties.csv")  # You must have created this during training
X_test = test_df.drop(columns=["price"])
y_test = test_df["price"]

# Tab layout
tab1, tab2 = st.tabs(["🏠 Prediction", "📊 Model Evaluation"])

with tab1:
    st.title("🏡 BA House Price Predictor")
    st.sidebar.header("Enter Property Details")


    # Sidebar inputs
    bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 2)
    bathrooms = st.sidebar.slider("Bathrooms", 0, 5, 1)
    surface_covered = st.sidebar.number_input("Surface Covered (m²)", min_value=10, max_value=1000, value=60)
    surface_total = st.sidebar.number_input("Surface Total (m²)", min_value=10, max_value=1000, value=80)

    # Neighborhood selection
    location = st.sidebar.selectbox("Neighborhood (l3)", sorted(df["l3"].dropna().unique()))

    # Restrict lat/lon range based on selected neighborhood
    selected_df = df[df["l3"] == location]
    min_lat, max_lat = round(float(selected_df["lat"].min()),4), round(float(selected_df["lat"].max()),4)
    min_lon, max_lon = round(float(selected_df["lon"].min()),4), round(float(selected_df["lon"].max()),4)

    latitude = st.sidebar.number_input(
        "Latitude", min_value=float(min_lat), max_value=float(max_lat), value=round(float(selected_df.lat.mean()),4), format='%.4f'
    )
    longitude = st.sidebar.number_input(
        "Longitude", min_value=float(min_lon), max_value=float(max_lon), value=round(float(selected_df.lon.mean()),4), format='%.4f'
    )

    property_type = st.sidebar.selectbox("Property Type", sorted(df["property_type"].dropna().unique()))

    # Predict button
    if st.sidebar.button("Predict Price"):
        input_df = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "surface_covered": surface_covered,
            "surface_total": surface_total,
            "l3": location,
            "property_type": property_type,
            "lat": latitude,
            "lon": longitude
        }])
        
        predicted_price = pipeline.predict(input_df)[0]
        st.session_state["predicted"] = True

        st.subheader(f"💰 Predicted Price: ${predicted_price:,.2f} USD")

    # Property price map
    st.subheader("📍 Property Price Map (Capital Federal)")
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="price",
        size="surface_covered",
        hover_data=["l3", "bedrooms", "bathrooms", "price"],
        color_continuous_scale="viridis",
        size_max=10,
        zoom=10.5,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")

    # Add predicted point to map if available
    if st.session_state.get("predicted", False):
        fig.add_trace(go.Scattermapbox(
            lat=[latitude],
            lon=[longitude],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=10, color="red", symbol="circle"),
            text=["📍 Your Prediction"],
            textposition="top center",
            hoverinfo="text"
        ))

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)

# ----------------------------- #
#     TAB 2: MODEL EVALUATION   #
# ----------------------------- #
with tab2:
    st.header("📊 Model Evaluation")

    # --- Metrics ---
    st.subheader("📈 Performance Metrics")
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown(f"- **RMSE**: ${rmse:,.2f}")
    st.markdown(f"- **MAE**: ${mae:,.2f}")
    st.markdown(f"- **R² Score**: {r2:.3f}")

    # --- Prediction Plot ---
    st.subheader("📉 Actual vs Predicted Prices")
    fig2 = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Price', 'y': 'Predicted Price'},
        title="Actual vs Predicted Price"
    )
    fig2.add_shape(
        type="line", line=dict(dash='dash', color="gray"),
        x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max()
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Feature Importances (if available) ---
    st.subheader("🔍 Feature Importances")
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": pipeline.named_steps.preprocessor.get_feature_names_out(),
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="Feature Importance")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Feature importances are not available for this model type.")