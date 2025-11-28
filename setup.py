# setup.py
import sys
from cx_Freeze import setup, Executable
import os

# Define build_exe options
build_exe_options = {
    "packages": [
        "os",
        "sys",
        "flask",
        "jinja2",
        "plotly",
        "pandas",
        "speech_recognition",
        "pydub",
        "google.generativeai",
        "gtts",
        "logging",
        "json",
        "re",
        "tempfile",
        "uuid",
        "dotenv",
        "pyarrow",            # Added pyarrow
        "pyarrow.compute",    # Added pyarrow.compute
        "pyarrow._compute_docstrings",  # Added pyarrow._compute_docstrings
        # Add any other packages your app uses
    ],
    "includes": [
        "http",
        "http.server",
        "html",
        "html.parser",
        "pyarrow",
        "pyarrow.compute",
        "pyarrow._compute_docstrings",
        # Add any other modules that might be missing
    ],
    "include_files": [
        "templates",
        "forecast_templates",
        "static",
        "uploads",
        "descriptions.json",
        os.path.join("ffmpeg", "bin", "ffmpeg.exe"),  # Adjust path if necessary
        "forecast_bp.py",
        "best_lightgbm_hyperparameters.pkl",
        "best_lightgbm_model.pkl",
        "app.log",
        # "app.ico",      # Optional: Your application icon
        # ".env",         # If using a .env file
        # Include any other necessary files or directories
    ],
    "excludes": [
        "tkinter",
        "unittest",
        "email",
        # Ensure "html" and "http" are NOT excluded
        # Add other packages to exclude if necessary
    ],
    "include_msvcr": True,
    "optimize": 2,
}

# Base setup options
base = None
if sys.platform == "win32":
    base = "Console"  # Set to "Console" to enable console window

# Define executables
executables = [
    Executable(
        "app.py",
        base=base,
        icon="app.ico",  # Ensure 'app.ico' exists in your project directory
        target_name="InteractiveSalesDashboard.exe"
    )
]

# Setup configuration
setup(
    name="InteractiveSalesDashboard",
    version="1.0",
    description="A Flask-based interactive sales dashboard application.",
    options={"build_exe": build_exe_options},
    executables=executables
)
