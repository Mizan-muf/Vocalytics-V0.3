# forecast_bp.py
from flask import Blueprint, request, render_template, current_app
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.offline import plot
import os
import uuid
import logging
import sys

from jinja2 import ChoiceLoader, FileSystemLoader

# Define resource_path function
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for cx_Freeze."""
    try:
        # cx_Freeze sets the base_path as sys._MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize Blueprint with dynamic paths
forecast_bp = Blueprint(
    'forecast', 
    __name__, 
    template_folder=resource_path('forecast_templates'), 
    static_folder=resource_path('static')
)

# Configure Jinja2 to use multiple template directories if necessary
forecast_bp.jinja_loader = ChoiceLoader([
    forecast_bp.jinja_loader,
    FileSystemLoader(resource_path('forecast_templates'))
])

# Load the pre-trained LightGBM model using a dynamic path
MODEL_PATH = resource_path("best_lightgbm_model.pkl")
if os.path.exists(MODEL_PATH):
    best_lgbm = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found! Train and save the model first.")

def forecast_revenue(df):
    # Convert date and sort
    df['Trial Start Date'] = pd.to_datetime(df['Trial Start Date'])
    df = df.sort_values(by='Trial Start Date')

    # Aggregate revenue weekly
    df.set_index('Trial Start Date', inplace=True)
    weekly_revenue = df.resample('W')['Revenue'].sum().reset_index()

    # Feature engineering
    weekly_revenue['Week'] = weekly_revenue['Trial Start Date'].dt.isocalendar().week
    weekly_revenue['Year'] = weekly_revenue['Trial Start Date'].dt.year

    # Create lag features
    for i in range(1, 11):
        weekly_revenue[f'Lag_{i}'] = weekly_revenue['Revenue'].shift(i)

    # Moving averages and standard deviation
    weekly_revenue['MA_4'] = weekly_revenue['Revenue'].rolling(window=4).mean()
    weekly_revenue['MA_8'] = weekly_revenue['Revenue'].rolling(window=8).mean()
    weekly_revenue['Std_4'] = weekly_revenue['Revenue'].rolling(window=4).std()

    weekly_revenue.dropna(inplace=True)
    if weekly_revenue.empty:
        raise ValueError("❌ Not enough data for forecasting.")

    # Forecast next 12 weeks
    num_weeks_to_forecast = 12
    future_weeks = pd.DataFrame({
        'Trial Start Date': pd.date_range(start=weekly_revenue['Trial Start Date'].max() + pd.Timedelta(weeks=1),
                                          periods=num_weeks_to_forecast, freq='W'),
    })
    future_weeks['Week'] = future_weeks['Trial Start Date'].dt.isocalendar().week
    future_weeks['Year'] = future_weeks['Trial Start Date'].dt.year

    predicted_revenue_df = future_weeks.copy()
    temp_weekly_revenue = weekly_revenue.copy()

    for i in range(num_weeks_to_forecast):
        latest_lags = [temp_weekly_revenue['Revenue'].iloc[-j] if len(temp_weekly_revenue) >= j else np.nan
                       for j in range(1, 11)]
        ma_4 = temp_weekly_revenue['MA_4'].iloc[-1] if len(temp_weekly_revenue) >= 4 else np.nan
        ma_8 = temp_weekly_revenue['MA_8'].iloc[-1] if len(temp_weekly_revenue) >= 8 else np.nan
        std_4 = temp_weekly_revenue['Std_4'].iloc[-1] if len(temp_weekly_revenue) >= 4 else np.nan

        next_week_data = {
            'Week': future_weeks.iloc[i]['Week'],
            'Year': future_weeks.iloc[i]['Year'],
            **{f'Lag_{j}': latest_lags[j-1] for j in range(1, 11)},
            'MA_4': ma_4,
            'MA_8': ma_8,
            'Std_4': std_4,
        }
        next_week_df = pd.DataFrame([next_week_data])
        future_prediction = best_lgbm.predict(next_week_df)[0]
        predicted_revenue_df.at[i, 'Predicted Revenue'] = future_prediction

        new_row = {
            'Trial Start Date': future_weeks.iloc[i]['Trial Start Date'],
            'Revenue': future_prediction,
            'Week': next_week_data['Week'],
            'Year': next_week_data['Year']
        }
        for key in next_week_data:
            new_row[key] = next_week_data[key]
        temp_weekly_revenue = pd.concat([
            temp_weekly_revenue,
            pd.DataFrame(new_row, index=[0])
        ], ignore_index=True)

        temp_weekly_revenue['MA_4'] = temp_weekly_revenue['Revenue'].rolling(window=4).mean()
        temp_weekly_revenue['MA_8'] = temp_weekly_revenue['Revenue'].rolling(window=8).mean()
        temp_weekly_revenue['Std_4'] = temp_weekly_revenue['Revenue'].rolling(window=4).std()

    return weekly_revenue, predicted_revenue_df

def process_sales_data(file_path):
    # Attempt to read the CSV file with various encodings
    encodings = ["utf-8", "latin-1", "iso-8859-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"CSV file read successfully with encoding: {enc}")
            break
        except Exception as e:
            logger.warning(f"Failed to read CSV with encoding {enc}: {e}")
    else:
        raise ValueError("Could not read CSV file with available encodings.")

    if "Sales Person" not in df.columns:
        raise ValueError("Error: 'Sales Person' column not found in the CSV file.")

    results = {}

    # Process forecast for each salesperson
    unique_sales = df["Sales Person"].unique()
    for person in unique_sales:
        person_df = df[df["Sales Person"] == person].copy()

        # Skip processing if there is insufficient data for this salesperson
        try:
            actual_data, forecasted_data = forecast_revenue(person_df)
        except Exception as e:
            logger.warning(f"Error forecasting for {person}: {e}")
            continue

        # Create Plotly graph for the salesperson
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual_data['Trial Start Date'].iloc[-30:], 
            y=actual_data['Revenue'].iloc[-30:], 
            mode='lines+markers', 
            name='Actual Revenue', 
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=forecasted_data['Trial Start Date'], 
            y=forecasted_data['Predicted Revenue'], 
            mode='lines+markers', 
            name='Predicted Revenue', 
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title=f"Forecast for {person}",
            xaxis_title="Date",
            yaxis_title="Revenue",
            template="plotly_white"
        )

        graph_div = plot(fig, output_type='div', include_plotlyjs=False)
        results[person] = {
            'actual_data': actual_data.to_dict(orient='records'),
            'forecasted_data': forecasted_data.to_dict(orient='records'),
            'graph_div': graph_div
        }

    # Overall Forecasting
    try:
        actual_overall_data, forecasted_overall_data = forecast_revenue(df)
    except Exception as e:
        logger.error(f"Error forecasting overall: {e}")
        return results  # Return only salesperson data if overall forecast fails

    fig_overall = go.Figure()
    fig_overall.add_trace(go.Scatter(
        x=actual_overall_data['Trial Start Date'].iloc[-30:], 
        y=actual_overall_data['Revenue'].iloc[-30:], 
        mode='lines+markers', 
        name='Actual Overall Revenue',
        line=dict(color='green')
    ))
    fig_overall.add_trace(go.Scatter(
        x=forecasted_overall_data['Trial Start Date'], 
        y=forecasted_overall_data['Predicted Revenue'],
        mode='lines+markers', 
        name='Predicted Overall Revenue', 
        line=dict(color='orange', dash='dash')
    ))
    fig_overall.update_layout(
        title="Actual vs Forecasted Weekly Overall Revenue",
        xaxis_title="Date",
        yaxis_title="Revenue",
        legend_title="Legend",
        template="plotly_white"
    )

    overall_graph_div = plot(fig_overall, output_type='div', include_plotlyjs=False)

    # Removed summary generation and TTS functionality
    overall_results = {
        'actual_data': actual_overall_data.to_dict(orient='records'),
        'forecasted_data': forecasted_overall_data.to_dict(orient='records'),
        'graph_div': overall_graph_div
    }

    results['Overall'] = overall_results
    return results

@forecast_bp.route('/', methods=['GET', 'POST'])
def forecast_handler():
    if request.method == 'GET':
        return render_template('forecast_index.html')

    if 'file' not in request.files:
        return "❌ No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "❌ No file selected!", 400

    uploaded_filename = f"uploaded_{uuid.uuid4().hex}.xlsx"
    uploads_dir = resource_path('uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, uploaded_filename)
    file.save(file_path)

    try:
        results = process_sales_data(file_path)
    except ValueError as ve:
        return str(ve), 400
    except Exception as e:
        return f"⚠ An error occurred during processing: {e}", 500

    os.remove(file_path)
    return render_template('forecast_result.html', results=results)
