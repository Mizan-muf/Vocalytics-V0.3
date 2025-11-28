# Vocalytics-V0.3
 Project Explanation This AI-powered sales dashboard acts as an Auto Data Scientist, automatically detecting CSV encodings, cleaning datasets, and engineering features (lags, moving averages) for LightGBM forecasting. It combines voice-driven analytics with Google Gemini to generate dynamic Plotly charts and audio summaries.
## 🚀 Key Features

*   **🤖 Auto Data Scientist Module:**
    *   **Smart Ingestion:** Automatically detects file encodings (UTF-8, Latin-1, etc.) to prevent read errors.
    *   **Auto-Cleaning & Engineering:** Automatically generates rolling windows, standard deviations, and lag features for time-series analysis without user intervention.
    *   **Schema Analysis:** Scans datasets to extract unique values and column types to give the AI perfect context.
*   **🗣️ Voice-Driven Analytics:** Convert natural language voice commands into executable Python code to generate complex Plotly visualizations.
*   **🔮 Automated Forecasting:** Uses a pre-trained LightGBM model to predict weekly revenue for the next 12 weeks, handling all data pre-processing internally.
*   **📊 AI Insight Generation:** Generates text and audio summaries explaining the trends found in the visualizations.
*   **📦 Standalone Deployment:** Bundled via cx_Freeze as a Windows `.exe` requiring no external Python setup.

## 🛠️ Tech Stack

*   **Core:** Python, Flask
*   **Data Intelligence:** Pandas, NumPy, LightGBM, Scikit-learn
*   **Generative AI:** Google Gemini Pro/Flash
*   **Audio/Voice:** SpeechRecognition, gTTS, Pydub, FFmpeg
*   **Visualization:** Plotly (JSON serialization)
*   **Security:** AST-based code validation

## 📂 Project Structure

*   `app.py`: Main controller. Handles the "Auto Data Scientist" logic for AI context generation and secure code execution.
*   `forecast_bp.py`: Handles the automated feature engineering and ML inference pipeline.
*   `setup.py`: Build script for creating the standalone executable.
*   `templates/`: UI templates.
*   `static/`: Assets and generated media.

## ⚙️ Setup & Installation

1.  **Prerequisites:**
    *   Python 3.10+
    *   FFmpeg (placed in `ffmpeg/bin/ffmpeg.exe`)
    *   Google Gemini API Key (configured in `app.py`)

2.  **Install Dependencies:**
    ```bash
    pip install flask pandas numpy joblib plotly speechrecognition pydub google-generativeai gtts cx_Freeze lightgbm chardet pyarrow
    ```

3.  **Required Files:**
    *   `best_lightgbm_model.pkl`: The pre-trained forecasting model.
    *   `descriptions.json`: Metadata storage.

## 🏗️ Building the Executable

To package the Auto Data Scientist module and UI into a single file:

python setup.py build
