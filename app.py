# app.py
import logging
import os
import uuid
import json
import re
import tempfile
import io
import sys  # Added import for sys module
from flask import Flask, request, jsonify, render_template
from jinja2 import ChoiceLoader, FileSystemLoader  # Added for multiple template directories
from dotenv import load_dotenv  # If using python-dotenv
import plotly
import pandas as pd
from speech_recognition import Recognizer, AudioFile, UnknownValueError, RequestError
from pydub import AudioSegment
import google.generativeai as genai
from gtts import gTTS
import lightgbm
import chardet
import ast  # Added for AST-based validation

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load environment variables from .env if present
load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; consider changing to INFO for production
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(resource_path("app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with dynamic resource paths
app = Flask(
    __name__,
    template_folder=resource_path('templates'),
    static_folder=resource_path('static')
)

# Configure Jinja2 to use multiple template directories
app.jinja_loader = ChoiceLoader([
    app.jinja_loader,
    FileSystemLoader(resource_path('forecast_templates'))
])

# Environment Variables for Configuration
genai.configure(api_key=GENAI_API_KEY)

# Global variables for CSV handling, descriptions
CSV_FILE_PATH = ""
ACTIVE_FILENAME = ""
DESCRIPTIONS_FILE = resource_path("descriptions.json")
descriptions = {}

def load_descriptions():
    global descriptions
    if os.path.exists(DESCRIPTIONS_FILE):
        try:
            with open(DESCRIPTIONS_FILE, "r") as f:
                # Try to load JSON; if file is empty/invalid, fall back to empty dict
                descriptions = json.load(f)
                logger.info("Descriptions loaded successfully.")
        except (json.JSONDecodeError, ValueError):
            # File is empty or contains invalid JSON
            descriptions = {}
            logger.warning("Descriptions file is empty or invalid. Starting with empty descriptions.")
    else:
        descriptions = {}
        logger.info("Descriptions file not found. Starting with empty descriptions.")

def save_descriptions():
    try:
        with open(DESCRIPTIONS_FILE, "w") as f:
            json.dump(descriptions, f)
            logger.info("Descriptions saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save descriptions: {e}")

load_descriptions()

# 📌 **Hardcoding ffmpeg and ffprobe Paths**
# Assuming 'ffmpeg/bin/' is placed alongside the executable

# Determine the directory where the executable resides
if getattr(sys, 'frozen', False):
    # If the application is run as a bundled executable
    application_path = os.path.dirname(sys.executable)
else:
    # If the application is run in a normal Python environment
    application_path = os.path.dirname(os.path.abspath(__file__))

# Define paths to ffmpeg.exe and ffprobe.exe
FFMPEG_PATH = os.path.join(application_path, 'ffmpeg', 'bin', 'ffmpeg.exe')
FFPROBE_PATH = os.path.join(application_path, 'ffmpeg', 'bin', 'ffprobe.exe')

# Verify that ffmpeg.exe exists
if not os.path.isfile(FFMPEG_PATH):
    logger.critical(f"ffmpeg executable not found at {FFMPEG_PATH}.")
    raise FileNotFoundError(f"ffmpeg executable not found at {FFMPEG_PATH}.")

# Verify that ffprobe.exe exists
if not os.path.isfile(FFPROBE_PATH):
    logger.critical(f"ffprobe executable not found at {FFPROBE_PATH}.")
    raise FileNotFoundError(f"ffprobe executable not found at {FFPROBE_PATH}.")

# Set pydub's ffmpeg and ffprobe paths
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

logger.info(f"ffmpeg configured at {FFMPEG_PATH}")
logger.info(f"ffprobe configured at {FFPROBE_PATH}")

# Ensure necessary directories exist
def ensure_directories():
    directories = ['static/graphs', 'uploads']
    for directory in directories:
        path = resource_path(directory)
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory: {path}")

ensure_directories()

recognizer = Recognizer()

def upload_to_gemini(path, mime_type=None):
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        logger.info(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    except Exception as e:
        logger.error(f"Error uploading file to Gemini: {e}")
        return None

# 📌 **Fixing the `restricted_import` Function**
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    allowed_modules = {
        'pandas', 
        'plotly.express', 
        'plotly.graph_objects', 
        'plotly.subplots'
    }
    # Allow exact matches or submodules
    if name in allowed_modules or any(name.startswith(f"{module}.") for module in allowed_modules):
        return __import__(name, globals, locals, fromlist, level)
    else:
        raise ImportError(f"Import of module '{name}' is not allowed.")

def validate_generated_code(code):
    """
    Validates the generated code to ensure it doesn't contain forbidden operations.
    Uses AST to parse and inspect the code structure.
    """
    forbidden_modules = {'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests'}
    forbidden_functions = {'open', 'exec', 'eval', 'compile', '__import__'}
    
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in forbidden_modules:
                        logger.error(f"Forbidden module import detected: {alias.name}")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in forbidden_modules:
                    logger.error(f"Forbidden module import detected: {node.module}")
                    return False
            # Check for forbidden function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_functions:
                    logger.error(f"Forbidden function call detected: {node.func.id}")
                    return False
                elif isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_functions:
                    logger.error(f"Forbidden function call detected: {node.func.attr}")
                    return False
        return True
    except Exception as e:
        logger.exception(f"Error during code validation: {e}")
        return False
def code_generation(query, file_path, description,):
    try:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(100000)
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']
            logger.info(f"Detected CSV encoding: {encoding}")

            df = pd.read_csv(file_path, encoding=encoding)
            buffer = io.StringIO()
            df.info(buf=buffer)
            columns_info = buffer.getvalue()
            sample_rows = df.head().to_string()
            logger.debug("CSV file read successfully for code generation.")
            
            unique_info = ""
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) < 25:
                    unique_info += f"Column '{col}' unique values: {list(unique_vals)}\n"
                    
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            columns_info = "Could not read CSV information."
            sample_rows = ""
            unique_info = ""

        generation_config = {
            "temperature": 0.6,
            "top_p": 0.5,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config,
        )

        system_prompt = f"""I have a dataset loaded from {file_path}, encoding: {encoding} with the following Columns:
{columns_info}

Here are some sample rows from the dataset:
{sample_rows}

Unique values for certain columns:
{unique_info}

Description: {description}

You will be given queries to write Python code using both Plotly to create graphs and dataframe of the graph to answer that only. 
Assign the Plotly figure to variable 'fig'.
Assign the final database for answering 'ans' and save it as 'ans = df.to_string()' do not have indexes in the dataframe.
don't use dict.

sort the database before using it.
sort the graph accordingly.

Do not import any modules outside the following allowed list: pandas, plotly.express, plotly.graph_objects, plotly.subplots.

Provide only the Python code as plain text without any markdown formatting or code block markers. 
Do not include fig.show(), plt.show(), print() in the code. Ensure that all column names are enclosed in quotes. 
Additionally, specify the encoding when reading the CSV file.
"""
        system_prompt += f"\nUser's current query: {query}\n"

        chat_session = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])
        response = chat_session.send_message(query)
        response.resolve()

        cleaned_text = response.text.strip().replace("```python", "").replace("```", "").strip()
        logger.debug(f"Generated Python code:\n{cleaned_text}")
        return cleaned_text

    except Exception as e:
        logger.exception(f"An error occurred during code generation: {e}")
        return ""

def code_generation(query, file_path, description):
    try:
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            buffer = io.StringIO()
            df.info(buf=buffer)
            columns_info = buffer.getvalue()
            sample_rows = df.head().to_string()
            logger.debug("CSV file read successfully for code generation.")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            columns_info = "Could not read CSV information."
            sample_rows = ""

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_output_tokens": 8192,  # Increased token limit for more comprehensive code
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        system_prompt = f"""I have a dataset loaded from {file_path} with the following Columns:
{columns_info}

Here are some sample rows from the dataset:
{sample_rows}

Further description: {description}

You will be given queries to write Python code using both Plotly to create graphs. 
Assign the Plotly figure to variable 'fig'.
don't use dict.

sort the database before using it.
sort the graph accordingly.

Do not import any modules outside the following allowed list: pandas, plotly.express, plotly.graph_objects, plotly.subplots.

Provide only the Python code as plain text without any markdown formatting or code block markers. 
Do not include fig.show(), plt.show() in the code. Ensure that all column names are enclosed in quotes. 
Additionally, specify the encoding when reading the CSV file.
"""

        chat_session = model.start_chat(history=[{"role": "user", "parts": [system_prompt]}])
        response = chat_session.send_message(query)
        response.resolve()

        cleaned_text = response.text.strip().replace("```python", "").replace("```", "").strip()
        logger.debug(f"Generated Python code:\n{cleaned_text}")
        return cleaned_text

    except Exception as e:
        logger.exception(f"An error occurred during code generation: {e}")
        return ""

def execute_dynamic_code(dynamic_code):
    logger.info("Executing dynamically generated Plotly code.")
    exec_namespace = {}
    try:
        # Validate the code first
        if not validate_generated_code(dynamic_code):
            logger.error("Generated code contains forbidden modules or operations. Execution aborted.")
            return None

        allowed_builtins = {
            '__builtins__': {
                'abs': abs,
                'min': min,
                'max': max,
                'range': range,
                '__import__': restricted_import,
            }
        }

        logger.debug(f"Dynamic Code to Execute:\n{dynamic_code}")
        exec(dynamic_code, allowed_builtins, exec_namespace)

        fig = exec_namespace.get('fig', None)

        if fig is None:
            logger.error("No Plotly figure named 'fig' was created in the dynamic code.")
            return None

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        logger.info("Plotly graph successfully created.")

        graphs_dir = resource_path('static/graphs')
        os.makedirs(graphs_dir, exist_ok=True)

        # Save Plotly figure as image
        if fig:
            filename = f"plotly_fig_{uuid.uuid4().hex}.png"
            image_path = os.path.join(graphs_dir, filename)
            fig.write_image(image_path)
            logger.info(f"Plotly figure saved as image at {image_path}.")

        return {
            "graphJSON": graphJSON, 
            "plotly_image_path": image_path
        }

    except Exception as e:
        logger.exception(f"An error occurred while executing the dynamic Plotly code: {e}")
        return None

def generate_summary(image_path):
    try:
        uploaded_file = upload_to_gemini(image_path, mime_type="image/png")
        if uploaded_file is None:
            logger.error("Failed to upload image to Gemini.")
            return None, None

        generation_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        system_prompt = """Act as a Query analyst and answer the given query with respect to the image.
        give me the answer only with little explaination."""
        input_prompt = [uploaded_file, system_prompt]

        chat_session = model.start_chat(history=[{"role": "user", "parts": input_prompt}])
        response = chat_session.send_message("Process this image")
        response.resolve()

        cleaned_text = response.text.strip()
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII
        cleaned_text = cleaned_text.replace('*', '').replace('_', '').replace('•', '')  # Remove markdown symbols
        summary = "\n".join(cleaned_text.splitlines())

        logger.info("Generated summary from image.")
        logger.debug(f"Final text for TTS: {summary}")  # Debugging step

        # Save summary audio using dynamic path
        tts = gTTS(text=summary, lang='en', slow=False, lang_check=False)
        audio_filename = f"summary_audio_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join(resource_path('static'), audio_filename)
        tts.save(audio_path)

        logger.info(f"Summary audio saved at {audio_path}.")
        return summary, audio_path

    except Exception as e:
        logger.exception(f"An error occurred while generating summary: {e}")
        return None, None

def create_ai_graphs(audio_file_path, csv_file_path, description):
    transcribed_text = ""
    try:
        logger.info("Starting audio conversion.")
        audio = AudioSegment.from_file(audio_file_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            converted_audio_path = tmp_wav.name
        logger.info(f"Audio file converted to WAV format: {converted_audio_path}")

        with AudioFile(converted_audio_path) as source:
            audio_data = recognizer.record(source)
            logger.info("Transcribing audio...")
            transcribed_text = recognizer.recognize_google(audio_data)
            logger.info(f"Transcribed Text: {transcribed_text}")

        dynamic_code = code_generation(transcribed_text, csv_file_path, description)
        if not dynamic_code:
            logger.error(f"No code was generated from the transcribed text and description.")
            return {'plot_json': None, 'summary': "Failed to generate graph.", 'audio_summary_url': None}

        execution_result = execute_dynamic_code(dynamic_code)
        if not execution_result:
            logger.error("Failed to execute the generated Plotly code.")
            return {'plot_json': None, 'summary': "Failed to generate graph.", 'audio_summary_url': None}

        plot_json = execution_result.get('graphJSON')
        image_path = execution_result.get('plotly_image_path')  # Adjusted to get Plotly image path

        if not plot_json or not image_path:
            logger.error("Plot JSON or image path is missing.")
            return {'plot_json': None, 'summary': "Incomplete graph generation.", 'audio_summary_url': None}

        summary, audio_path = generate_summary(image_path)
        audio_summary_url = f"/static/{os.path.basename(audio_path)}" if summary and audio_path else None

        logger.info("Dynamic Plotly graph and summary generated successfully.")
        return {'plot_json': plot_json, 'summary': summary, 'audio_summary_url': audio_summary_url, 'image_path': image_path}

    except UnknownValueError:
        logger.error("Google Speech Recognition could not understand audio.")
        return {'plot_json': None, 'summary': "Could not understand the audio input.", 'audio_summary_url': None}
    except RequestError as e:
        logger.error(f"Speech recognition service error; {e}")
        return {'plot_json': None, 'summary': "Speech recognition service error.", 'audio_summary_url': None}
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}. Exception: {e}")
        return {'plot_json': None, 'summary': "Required file not found.", 'audio_summary_url': None}
    except Exception as e:
        logger.exception(f"An unexpected error occurred in create_ai_graphs: {e}")
        return {'plot_json': None, 'summary': "An unexpected error occurred.", 'audio_summary_url': None}
    finally:
        try:
            if 'converted_audio_path' in locals() and os.path.exists(converted_audio_path):
                os.remove(converted_audio_path)
                logger.info(f"Temporary WAV file removed: {converted_audio_path}")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

def create_dashboard(csv_file_path, description):
    try:
        logger.info("Starting dashboard generation process.")
        # Define a system prompt tailored for dashboard generation with explicit subplot type instructions
        dashboard_prompt = (
            "Using the provided dataset located at "
            f"{csv_file_path}, create a comprehensive dashboard using Plotly with subplots. "
            "Only make 3 subplots that are simple to view and understandable."
            "The dashboard should include multiple subplots that provide insights such as trends, distributions, correlations, and comparative analyses based on the description provided. "
            "Ensure that all column names are enclosed in quotes. "
            "When creating subplots, assign appropriate subplot types based on the trace type: use 'domain' for Pie charts, specifying 'row' and 'column', and 'xy' for Scatter, Bar, Histogram, Box, and Heatmap charts. "
            "Include a variety of graph types such as Histogram, Bar Chart, Pie Chart and Time Series Plot where applicable. "
            "Provide only the Python code as plain text without any markdown formatting or code block markers. "
            "Do not include fig.show() in the code."
        )
        
        logger.info("Generating Python code for dashboard using AI.")
        # Generate the Python code for the dashboard
        dashboard_code = code_generation(dashboard_prompt, csv_file_path, description)
        if not dashboard_code:
            logger.error("No dashboard code was generated from the description.")
            return {'dashboard_json': None, 'dashboard_image_path': None}
        
        logger.debug(f"Generated Dashboard Code:\n{dashboard_code}")
        
        # Execute the generated dashboard code
        logger.info("Executing the generated dashboard code.")
        execution_result = execute_dynamic_code(dashboard_code)
        if not execution_result:
            logger.error("Failed to execute the generated dashboard code.")
            return {'dashboard_json': None, 'dashboard_image_path': None}
        
        dashboard_json = execution_result.get('graphJSON')
        dashboard_image_path = execution_result.get('plotly_image_path')  # Adjusted to get Plotly image path
        
        if dashboard_json:
            logger.info("Dashboard JSON generated successfully.")
        else:
            logger.error("Dashboard JSON generation failed.")
        
        if dashboard_image_path:
            logger.info(f"Dashboard image saved at: {dashboard_image_path}")
        else:
            logger.error("Dashboard image path is missing.")
        
        return {'dashboard_json': dashboard_json, 'dashboard_image_path': dashboard_image_path}
    
    except Exception as e:
        logger.exception(f"An error occurred while creating the dashboard: {e}")
        return {'dashboard_json': None, 'dashboard_image_path': None}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global CSV_FILE_PATH, descriptions, ACTIVE_FILENAME
    if 'csv_file' not in request.files:
        logger.warning("No CSV file provided in upload_csv route.")
        return jsonify({'error': 'No CSV file provided.'}), 400

    file = request.files['csv_file']
    if file.filename == '':
        logger.warning("No file selected in upload_csv route.")
        return jsonify({'error': 'No selected file.'}), 400

    uploads_dir = resource_path('uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    try:
        file.save(file_path)
        logger.info(f"CSV file uploaded and saved at {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded CSV file: {e}")
        return jsonify({'error': 'Failed to save uploaded CSV file.'}), 500

    CSV_FILE_PATH = file_path
    ACTIVE_FILENAME = file.filename  # Set the uploaded file as active
    descriptions[file.filename] = ""  
    save_descriptions()

    logger.info(f"Active CSV file set to: {ACTIVE_FILENAME}")
    return jsonify({'message': 'CSV file uploaded successfully.', 'filename': file.filename}), 200  # Include filename in response

@app.route('/list_csv', methods=['GET'])
def list_csv():
    uploads_dir = resource_path('uploads')
    if not os.path.exists(uploads_dir):
        logger.info("No uploads directory found in list_csv route.")
        return jsonify([])
    files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
    logger.info(f"Listing CSV files: {files}")
    return jsonify(files)

@app.route('/set_csv', methods=['POST'])
def set_csv():
    global CSV_FILE_PATH, ACTIVE_FILENAME
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        logger.warning("No filename provided in set_csv route.")
        return jsonify({'error': 'Filename is required.'}), 400

    uploads_dir = resource_path('uploads')
    filepath = os.path.join(uploads_dir, filename)
    if os.path.exists(filepath):
        CSV_FILE_PATH = filepath
        ACTIVE_FILENAME = filename  # Update active filename
        logger.info(f"CSV file set to {filepath}")
        return jsonify({'message': f'Selected {filename} for use.'})
    logger.warning(f"CSV file not found: {filepath}")
    return jsonify({'error': 'File not found.'}), 404

@app.route('/update_description', methods=['POST'])
def update_description():
    global descriptions
    data = request.get_json()
    filename = data.get('filename')
    description = data.get('description')
    if filename and description is not None:
        descriptions[filename] = description
        save_descriptions()
        logger.info(f"Description updated for {filename}.")
        return jsonify({'message': 'Description updated.'})
    logger.warning("Invalid input in update_description route.")
    return jsonify({'error': 'Invalid input.'}), 400

@app.route('/get_description', methods=['GET'])
def get_description():
    filename = request.args.get('filename')
    if filename in descriptions:
        logger.info(f"Retrieved description for {filename}.")
        return jsonify({'description': descriptions[filename]})
    logger.info(f"No description found for {filename}.")
    return jsonify({'description': ''})

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    global CSV_FILE_PATH
    if not CSV_FILE_PATH:
        logger.warning("generate_graph route accessed without uploaded CSV.")
        return jsonify({'error': 'CSV file not uploaded.'}), 400

    description = request.form.get('q1', '')
    if not description:
        logger.warning("No description provided in generate_graph route.")
        return jsonify({'error': 'Description is required.'}), 400

    if 'audio_data' not in request.files:
        logger.error("No audio file provided in the request.")
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        logger.warning("No audio file selected for uploading in generate_graph route.")
        return jsonify({'error': 'No selected file.'}), 400

    temp_audio_filename = f"temp_audio_{uuid.uuid4().hex}.webm"
    temp_audio_path = os.path.join(resource_path('static'), temp_audio_filename)

    try:
        audio_file.save(temp_audio_path)
        logger.info(f"Uploaded audio file saved temporarily at {temp_audio_path}")

        result = create_ai_graphs(temp_audio_path, CSV_FILE_PATH, description)

        if result['plot_json'] and result['summary']:
            response_payload = {
                'plot_json': result['plot_json'],
                'summary': result['summary'],
                'audio_summary_url': result.get('audio_summary_url', None),
                'image_path': result.get('image_path', "")
            }
            logger.info("Dynamic Plotly graph and summary generated successfully.")
            return jsonify(response_payload)
        else:
            logger.error("Failed to generate dynamic Plotly graph and/or summary.")
            return jsonify({'error': 'Failed to generate dynamic graph and summary.'}), 500

    except Exception as e:
        logger.exception(f"An error occurred in /generate_graph route: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

    finally:
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"Temporary audio file removed: {temp_audio_path}")
        except Exception as cleanup_error:
            logger.error(f"Error removing temporary audio file: {cleanup_error}")

@app.route('/generate_dashboard', methods=['POST'])
def generate_dashboard_route():
    global CSV_FILE_PATH, descriptions
    try:
        if not CSV_FILE_PATH:
            logger.warning("Dashboard generation attempted without an uploaded CSV.")
            return jsonify({'error': 'CSV file not uploaded.'}), 400

        description = request.form.get('description', '')
        if not description:
            logger.warning("Dashboard generation attempted without a description.")
            return jsonify({'error': 'Description is required to generate dashboard.'}), 400

        logger.info(f"Generating dashboard for CSV: {CSV_FILE_PATH} with description: {description[:50]}...")

        result = create_dashboard(CSV_FILE_PATH, description)

        if result['dashboard_json'] or result['dashboard_image_path']:
            logger.info("Dashboard generation successful.")
            return jsonify({
                'dashboard_json': result.get('dashboard_json', None),
                'dashboard_image_path': result.get('dashboard_image_path', None)
            })
        else:
            logger.error("Dashboard generation failed.")
            return jsonify({'error': 'Failed to generate dashboard.'}), 500
    except Exception as e:
        logger.exception(f"An error occurred in /generate_dashboard route: {e}")
        return jsonify({'error': 'Internal server error.'}), 500

# Register the forecasting blueprint if exists
from forecast_bp import forecast_bp
app.register_blueprint(forecast_bp, url_prefix='/forecast')

if __name__ == "__main__":
    try:
        logger.info("Starting Flask server.")
        app.run(debug=True, port=8000)
    except Exception as e:
        logger.critical(f"Failed to start Flask server: {e}")

