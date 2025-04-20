from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai
from datetime import datetime
import speech_recognition as sr
from googletrans import Translator
import pyttsx3
import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import asyncio
import python_weather
from bson import ObjectId
from datetime import timedelta
from functools import wraps
import bcrypt
import uuid
from werkzeug.utils import secure_filename
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
from flask import send_file
import joblib
import csv
from io import StringIO
from flask import Response

app = Flask(__name__)
load_dotenv()
client = MongoClient('mongodb://localhost:27017/')

# Add secret key for sessions
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

# Configure MongoDB with better error handling
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
try:
    # Configure MongoDB for local development
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
        retryWrites=True,
        w='majority'
    )
    # Test connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
    raise

db = client['veterinary_db']

# Collections
users = db.users
images = db.images
vaccinations = db.vaccinations
appointments = db.appointments
records = db.veterinary_records
chats = db.chat_history
health_records = db.health_records  # New collection for health records
predictions_collection = db.predictions  # New collection for disease predictions
posts = db.posts  # New collection for discussion forum posts
marketplace_listings = db.marketplace_listings  # New collection for marketplace listings
animals = db.animals  # New collection for animals
#diseases = db.diseases  # New collection for disease reports
diseases_collection = db.diseases

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("No API key found. Please set GOOGLE_API_KEY in .env file")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')
image_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize other services
translator = Translator()

# Initialize text-to-speech engine with fallback
try:
    speech_engine = pyttsx3.init()
    print("Successfully initialized text-to-speech engine")
except Exception as e:
    print(f"Warning: Failed to initialize text-to-speech engine: {str(e)}")
    speech_engine = None

UPLOAD_FOLDER = 'static/uploads/animals'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# Configure upload folder for disease prediction
UPLOAD_FOLDER_PREDICTION = 'uploads'
ALLOWED_EXTENSIONS_PREDICTION = {'png', 'jpg', 'jpeg'}
if not os.path.exists(UPLOAD_FOLDER_PREDICTION):
    os.makedirs(UPLOAD_FOLDER_PREDICTION)

# Configure upload folder for marketplace images
UPLOAD_FOLDER_MARKETPLACE = 'static/marketplace_images'
if not os.path.exists(UPLOAD_FOLDER_MARKETPLACE):
    os.makedirs(UPLOAD_FOLDER_MARKETPLACE)

# Load encoders and models with error handling
try:
    # Load the symptom encoder
    encoder_symptoms = joblib.load('encoder_symptoms.pkl')

    # Load the disease encoder
    encoder_disease = joblib.load('encoder_disease.pkl')

    # Load the prediction model
    disease_model = joblib.load('disease_prediction_model.pkl')
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Symptom list (based on your dataset)
symptom_list = sorted([
    "Fever", "Cough", "Weight Loss", "Swollen Udder", "Low Appetite", "Diarrhea",
    "Lethargy", "Hair Loss", "Nasal Discharge", "Pain in Udder", "Weakness",
    "Loss of Appetite", "Swollen Joints", "Bloody Diarrhea", "Dehydration",
    "High Temperature", "Redness in Eyes", "Poor Coat", "Swollen Hoof",
    "Blood in Milk", "Swollen Abdomen", "Bloody Stool", "Redness in Mouth",
    "Swelling Around Eyes", "Swelling in Neck", "Fatigue"
])

# Disease information dictionary
disease_info = {
    "Tuberculosis": {
        "description": "A bacterial infection that affects the lungs and other organs.",
        "precautions": "Isolate infected animals and test others for exposure.",
        "medication": "Use antibiotics under veterinary guidance.",
        "workouts": "Rest is essential; avoid strenuous activity.",
        "diet": "Provide a high-energy diet to support immune function."
    },
    # ... (rest of the disease info dictionary)
}

# Update Gemini configuration
GOOGLE_API_KEY = 'AIzaSyA2YckBfXX9FwqAQQuHYbhkYoZe4AIn7ws'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Gemini model with the newer version
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                     generation_config=generation_config,
                                     safety_settings=safety_settings)

# Add this near the top of your file with other constants
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", 
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", 
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", 
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh", 
    "Lakshadweep", "Puducherry"
]

# Add a new route to get states
@app.route('/api/states')
def get_states():
    return jsonify(INDIAN_STATES)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password are required'}), 400

        user = users.find_one({'username': username})

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            return jsonify({'success': True})

        return jsonify({'success': False, 'error': 'Invalid username or password'}), 401

    except Exception as e:
        print(f"Login error: {str(e)}")  # Server-side logging
        return jsonify({'success': False, 'error': 'Server error occurred'}), 500


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    try:
        print("Received registration request")
        data = request.get_json()
        print(f"Request data: {data}")

        if not data:
            print("No data provided")
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        required_fields = ['username', 'password', 'fullName', 'email', 'state', 'district']
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            print(f"Missing fields: {missing_fields}")
            return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Check if username or email already exists
        if users.find_one({'username': data['username']}):
            print(f"Username {data['username']} already exists")
            return jsonify({'success': False, 'error': 'Username already exists'}), 409

        if users.find_one({'email': data['email']}):
            print(f"Email {data['email']} already exists")
            return jsonify({'success': False, 'error': 'Email already exists'}), 409

        try:
            # Hash password and convert to string for MongoDB storage
            hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

            # Prepare user data with proper data types for MongoDB
            user_data = {
                'username': str(data['username']),
                'email': str(data['email']),
                'password': hashed_password,
                'full_name': str(data['fullName']),
                'state': str(data['state']),
                'district': str(data['district']),
                'farm_size': float(data.get('farmSize', 0)),
                'livestock': {
                    'total': int(data.get('livestock', {}).get('total', 0)),
                    'cows': int(data.get('livestock', {}).get('cows', 0)),
                    'buffaloes': int(data.get('livestock', {}).get('buffaloes', 0)),
                    'goats': int(data.get('livestock', {}).get('goats', 0)),
                    'sheep': int(data.get('livestock', {}).get('sheep', 0))
                },
                'created_at': datetime.utcnow()
            }
            print(f"Prepared user data (excluding password): {dict(user_data, password='[HIDDEN]')}")

            result = users.insert_one(user_data)
            if result.inserted_id:
                print(f"User successfully inserted with ID: {result.inserted_id}")
                return jsonify({'success': True})
            else:
                print("Failed to insert user - no ID returned")
                return jsonify({'success': False, 'error': 'Failed to create user'}), 500

        except Exception as inner_e:
            print(f"Error in database operation: {str(inner_e)}")
            return jsonify({'success': False, 'error': f'Database operation failed: {str(inner_e)}'}), 500

    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error occurred: {str(e)}'}), 500


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@login_required
def home():
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')
    user.pop('password', None)
    return render_template('index.html', user=user)


@app.route('/symptoms')
@login_required
def symptoms_page():
    return render_template('symptoms.html', symptoms=symptom_list)


@app.route('/add_record', methods=['POST'])
def add_record():
    try:
        data = request.json
        record = {
            'animal_id': data['animal_id'],
            'animal_type': data['animal_type'],
            'breed': data['breed'],
            'age': data['age'],
            'symptoms': data['symptoms'],
            'diagnosis': data['diagnosis'],
            'treatment': data['treatment'],
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        records.insert_one(record)
        return jsonify({"message": "Record added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Convert image to JPEG format
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save to bytes for base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        # Analyze image with Gemini Vision
        prompt = "Analyze this veterinary image and describe any visible symptoms or conditions you can see. If you can identify the animal type, breed, or any health indicators, please mention them."

        try:
            response = image_model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_base64}])
            analysis_text = response.text if response and response.text else "No analysis available"
        except Exception as e:
            analysis_text = f"Image analysis failed: {str(e)}"

        # Store image in database
        image_data = {
            'image': img_base64,
            'analysis': analysis_text,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        images.insert_one(image_data)

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_base64}",
            "analysis": analysis_text
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/voice_input', methods=['POST'])
def voice_input():
    try:
        audio_data = request.files['audio']
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        # Only attempt text-to-speech if engine is available
        if speech_engine is not None:
            speech_engine.say(text)
            speech_engine.runAndWait()

        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        text = request.json['text']
        target_lang = request.json['target_lang']

        translated = translator.translate(text, dest=target_lang)
        return jsonify({"translated": translated.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_weather', methods=['GET'])
async def get_weather():
    try:
        client = python_weather.Client()
        city = request.args.get('city', 'Mumbai')

        weather = await client.get(city)
        await client.close()

        return jsonify({
            "temperature": weather.current.temperature,
            "description": weather.current.description
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_analytics')
@login_required
def get_analytics():
    try:
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        print("User found:", user['_id'])  # Debug print

        # Initialize analytics data
        analytics = {
            'livestock_count': {},
            'health_statistics': {
                'healthy': 0,
                'sick': 0,
                'recovering': 0
            },
            'age_distribution': {
                '0-2': 0,
                '2-5': 0,
                '5-10': 0,
                '10+': 0
            },
            'recent_health_records': [],
            'vaccination_due': 0
        }

        # Process livestock data
        livestock_details = user.get('livestock_details', {})
        for animal_type in ['cow', 'buffalo', 'goat', 'sheep']:
            # Filter out deleted animals
            animals = [animal for animal in livestock_details.get(animal_type, [])
                       if not animal.get('is_deleted', False)]

            analytics['livestock_count'][animal_type] = len(animals)

            for animal in animals:
                # Health status
                health_status = animal.get('health_status', 'healthy').lower()
                if health_status in analytics['health_statistics']:
                    analytics['health_statistics'][health_status] += 1

                # Age distribution
                age = float(animal.get('age', 0))
                if age <= 2:
                    analytics['age_distribution']['0-2'] += 1
                elif age <= 5:
                    analytics['age_distribution']['2-5'] += 1
                elif age <= 10:
                    analytics['age_distribution']['5-10'] += 1
                else:
                    analytics['age_distribution']['10+'] += 1

        print("Processed livestock data:", analytics['livestock_count'])  # Debug print

        # Get recent health records for active animals only
        try:
            # First, get all active animal IDs
            active_animal_ids = set()
            for animal_type in livestock_details:
                for animal in livestock_details[animal_type]:
                    if not animal.get('is_deleted', False):
                        active_animal_ids.add(str(animal['id']))

            # Get health records only for active animals
            recent_records = list(health_records.find({
                'user_id': str(user['_id']),
                'animal_id': {'$in': list(active_animal_ids)}
            }, {
                '_id': 0,
                'animal_id': 1,
                'animal_type': 1,
                'symptoms': 1,
                'condition': 1,
                'timestamp': 1
            }).sort('timestamp', -1).limit(5))

            print(f"Found {len(recent_records)} health records")  # Debug print

            # Format the records
            formatted_records = []
            for record in recent_records:
                try:
                    formatted_record = {
                        'animal_id': record['animal_id'],
                        'animal_type': record['animal_type'],
                        'symptoms': record['symptoms'],
                        'condition': record.get('condition', 'unknown'),
                        'date': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(record['timestamp'],
                                                                                                datetime) else str(
                            record['timestamp'])
                    }
                    formatted_records.append(formatted_record)
                except Exception as e:
                    print(f"Error formatting record: {str(e)}")
                    print("Record data:", record)
                    continue

            analytics['recent_health_records'] = formatted_records
            print("Formatted health records:", formatted_records)  # Debug print

        except Exception as e:
            print(f"Error getting health records: {str(e)}")
            analytics['recent_health_records'] = []

        # Get upcoming vaccinations count for active animals only
        try:
            current_date = datetime.now()
            thirty_days_later = current_date + timedelta(days=30)

            # Debug print current date range
            print(
                f"Checking vaccinations between {current_date.strftime('%Y-%m-%d')} and {thirty_days_later.strftime('%Y-%m-%d')}")

            # Get all appointments
            upcoming_appointments = list(appointments.find({
                'status': 'scheduled'
            }))

            # Debug print all appointments
            print("All appointments:", upcoming_appointments)

            # Count appointments in the next 30 days
            vaccination_count = 0
            for appt in upcoming_appointments:
                try:
                    # Try parsing the appointment date
                    appt_date = datetime.strptime(appt['appointmentDate'], '%Y-%m-%d')
                    if current_date <= appt_date <= thirty_days_later:
                        if 'animal_id' not in appt or str(appt['animal_id']) in active_animal_ids:
                            vaccination_count += 1
                except Exception as e:
                    print(f"Error processing appointment: {str(e)}")
                    print("Appointment data:", appt)
                    continue

            analytics['vaccination_due'] = vaccination_count
            print(f"Found {vaccination_count} upcoming vaccinations")  # Debug print

        except Exception as e:
            print(f"Error getting vaccination count: {str(e)}")
            analytics['vaccination_due'] = 0

        return jsonify(analytics)

    except Exception as e:
        print(f"Error in get_analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        if 'user_id' not in session:
            return jsonify({'response': 'Please login to use the chat feature.'})

        data = request.get_json()
        user_query = data.get('query', '') or data.get('message', '')
        user_query = user_query.strip()

        print(f"Received query: {user_query}")  # Debug log

        # Get current user's data
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        if not user:
            return jsonify({'response': 'User data not found. Please try logging in again.'})

        # First, check if this is an animal-related query
        words = user_query.lower().split()

        # Check each word for potential animal ID
        for word in words:
            # Search through all animal types
            for animal_type in ['cow', 'buffalo', 'goat', 'sheep']:
                animals = user.get('livestock_details', {}).get(animal_type, [])
                for animal in animals:
                    animal_id = str(animal.get('id', '')).lower()
                    if word == animal_id:
                        animal_info = f"""Here are the details for your {animal_type.title()} (ID: {animal['id']}):
• Breed: {animal['breed']}
• Age: {animal['age']} years
• Weight: {animal['weight']} kg
• Color: {animal['color']}
• Gender: {animal['gender']}
• Health Status: {animal['health_status']}"""

                        if animal.get('notes'):
                            animal_info += f"\n• Notes: {animal['notes']}"

                        # Add vaccination info if available
                        vaccinations = list(records.find({
                            'user_id': str(session['user_id']),
                            'animal_type': animal_type,
                            'animal_id': animal['id']
                        }))

                        if vaccinations:
                            animal_info += "\n\nVaccination History:"
                            for vac in vaccinations:
                                animal_info += f"\n• {vac['vaccine_name']} - {vac['date']}"
                        else:
                            animal_info += "\n\nNo vaccination records found for this animal."

                        return jsonify({'response': animal_info})

        # If no animal ID was found, use Gemini AI for response
        try:
            # Create a rich context for Gemini
            context = f"""As a veterinary AI assistant, please help with this query. Here's the relevant context:

User Profile:
- Location: {user.get('state', 'Unknown')}, {user.get('district', 'Unknown')}
- Farm Size: {user.get('farm_size', 'Unknown')} acres

Livestock Information:
- Total Cows: {user.get('livestock', {}).get('cows', 0)}
- Total Buffaloes: {user.get('livestock', {}).get('buffaloes', 0)}
- Total Goats: {user.get('livestock', {}).get('goats', 0)}
- Total Sheep: {user.get('livestock', {}).get('sheep', 0)}

Recent Veterinary Records:"""

            # Add recent vaccination records
            recent_records = list(records.find(
                {'user_id': str(session['user_id'])},
                {'_id': 0}
            ).sort('date', -1).limit(5))

            if recent_records:
                for record in recent_records:
                    context += f"\n- {record['date']}: {record['vaccine_name']} for {record['animal_type']}"
            else:
                context += "\n- No recent vaccination records found"

            # Create the complete prompt for Gemini
            prompt = f"""User Query: {user_query}

Please provide a detailed response that:
1. Addresses the user's specific question
2. Considers their livestock profile
3. Includes relevant veterinary advice
4. Mentions preventive care when applicable
5. Uses simple, clear language
6. Provides actionable recommendations

Response:"""

            # Get response from Gemini
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)

            if response.text:
                return jsonify({'response': response.text})
            else:
                # Fallback response if Gemini fails
                return jsonify({
                    'response': "I understand you're asking about veterinary care. Could you please provide more specific details about your question? This will help me give you more accurate and helpful information."
                })

        except Exception as e:
            print(f"Gemini AI error: {str(e)}")
            # Fallback to database information if Gemini fails
            return jsonify({
                'response': """I'm currently having trouble accessing my advanced features. However, I can help you with:
• Viewing animal details (provide an animal ID)
• Checking vaccination records
• Basic livestock information
Please try asking a more specific question about your animals or their care."""
            })

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({
            'response': 'I apologize, but I encountered an error processing your request. Please try again.'
        })


@app.route('/vaccination')
@login_required
def vaccination_page():
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')
    user.pop('password', None)
    return render_template('vaccination.html', user=user)


@app.route('/get_vaccination_data')
def get_vaccination_data():
    try:
        user_id = str(session['user_id'])
        print(f"Getting vaccination data for user: {user_id}")  # Debug print

        # Get all vaccination records
        vaccination_records = list(vaccinations.find(
            {'user_id': user_id},
            {
                '_id': 0,
                'animal_id': 1,
                'animal_type': 1,
                'vaccinationType': 1,
                'date': 1,
                'nextDueDate': 1
            }
        ))

        # Get all scheduled appointments
        scheduled_appointments = list(appointments.find(
            {
                'user_id': user_id,
                'status': 'scheduled'
            },
            {
                '_id': 1,
                'animal_id': 1,
                'animal_type': 1,
                'vaccinationType': 1,
                'appointmentDate': 1
            }
        ))

        print(f"Found {len(vaccination_records)} vaccination records")  # Debug print
        print(f"Found {len(scheduled_appointments)} scheduled appointments")  # Debug print

        # Format appointments for response
        formatted_appointments = []
        for appt in scheduled_appointments:
            try:
                formatted_appt = {
                    'id': str(appt['_id']),
                    'animal_id': appt['animal_id'],
                    'animal_type': appt['animal_type'],
                    'vaccinationType': appt['vaccinationType'],
                    'appointmentDate': appt['appointmentDate']
                }
                formatted_appointments.append(formatted_appt)
                print(f"Formatted appointment: {formatted_appt}")  # Debug print
            except Exception as e:
                print(f"Error formatting appointment: {str(e)}")
                print(f"Appointment data: {appt}")
                continue

        # Format vaccination records for response
        formatted_records = []
        for record in vaccination_records:
            try:
                formatted_record = {
                    'animal_id': record['animal_id'],
                    'animal_type': record['animal_type'],
                    'vaccinationType': record['vaccinationType'],
                    'date': record['date'],
                    'nextDueDate': record['nextDueDate']
                }
                formatted_records.append(formatted_record)
                print(f"Formatted vaccination record: {formatted_record}")  # Debug print
            except Exception as e:
                print(f"Error formatting vaccination record: {str(e)}")
                print(f"Record data: {record}")
                continue

        return jsonify({
            'success': True,
            'vaccinations': formatted_records,
            'appointments': formatted_appointments
        })

    except Exception as e:
        print(f"Error in get_vaccination_data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/book_vaccination', methods=['POST'])
def book_vaccination():
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Invalid content type. Expected application/json'
        }), 400

    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['animalId', 'animalType', 'breed', 'age', 'vaccinationType', 'appointmentDate']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]

        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Validate age
        try:
            age = float(data['age'])  # Convert to float to handle decimal years
            if age <= 0 or age > 30:  # Reasonable age range for livestock
                return jsonify({
                    'success': False,
                    'error': 'Invalid age. Age must be between 0 and 30 years'
                }), 400
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid age format'
            }), 400

        # Validate appointment date
        try:
            appointment_date = datetime.strptime(data['appointmentDate'], '%Y-%m-%d')
            if appointment_date.date() < datetime.now().date():
                return jsonify({
                    'success': False,
                    'error': 'Appointment date cannot be in the past'
                }), 400
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid date format'
            }), 400

        # Create appointment record with consistent field names
        appointment = {
            'animal_id': str(data['animalId']),  # Changed from animalId to animal_id
            'animal_type': data['animalType'].lower(),  # Changed from animalType and ensure lowercase
            'breed': data['breed'],
            'age': age,
            'vaccinationType': data['vaccinationType'],
            'appointmentDate': data['appointmentDate'],
            'status': 'scheduled',
            'createdAt': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_id': str(session['user_id'])  # Add user_id field
        }

        # Check for duplicate appointments
        existing_appointment = appointments.find_one({
            'animal_id': appointment['animal_id'],
            'vaccinationType': appointment['vaccinationType'],
            'appointmentDate': appointment['appointmentDate'],
            'status': 'scheduled'
        })

        if existing_appointment:
            return jsonify({
                'success': False,
                'error': 'An appointment already exists for this animal on the selected date'
            }), 400

        # Save to database
        result = appointments.insert_one(appointment)

        if result.inserted_id:
            return jsonify({
                'success': True,
                'message': 'Appointment booked successfully',
                'appointmentId': str(result.inserted_id)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save appointment'
            }), 500

    except Exception as e:
        app.logger.error(f"Error in book_vaccination: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500


@app.route('/complete_vaccination', methods=['POST'])
@login_required
def complete_vaccination():
    try:
        data = request.get_json()
        appointment_id = data.get('appointmentId')
        vaccination_date = data.get('vaccinationDate')
        next_due_date = data.get('nextDueDate')

        if not all([appointment_id, vaccination_date, next_due_date]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400

        # Get the appointment
        appointment = appointments.find_one({'_id': ObjectId(appointment_id)})
        if not appointment:
            return jsonify({
                'success': False,
                'error': 'Appointment not found'
            }), 404

        # Create vaccination record
        vaccination_record = {
            'user_id': str(session['user_id']),
            'animal_id': appointment['animal_id'],
            'animal_type': appointment['animal_type'],
            'vaccinationType': appointment['vaccinationType'],
            'date': vaccination_date,
            'nextDueDate': next_due_date,
            'createdAt': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save vaccination record
        vaccinations.insert_one(vaccination_record)

        # Update appointment status
        appointments.update_one(
            {'_id': ObjectId(appointment_id)},
            {'$set': {'status': 'completed'}}
        )

        return jsonify({
            'success': True,
            'message': 'Vaccination completed successfully'
        })

    except Exception as e:
        print(f"Error completing vaccination: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/profile')
@login_required
def profile():
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')

    # Remove password from user data
    user.pop('password', None)
    return render_template('profile.html', user=user)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.get_json()
        user_id = ObjectId(session['user_id'])

        # Calculate total livestock
        livestock = data.get('livestock', {})
        livestock['total'] = sum(livestock.values())

        # Update user data
        update_data = {
            'full_name': data['full_name'],
            'district': data['district'],
            'state': data['state'],
            'farm_size': float(data['farm_size']),
            'livestock': livestock
        }

        users.update_one(
            {'_id': user_id},
            {'$set': update_data}
        )

        return jsonify({'success': True})
    except Exception as e:
        print(f"Profile update error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to update profile'})


@app.route('/livestock')
@login_required
def livestock_management():
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')

    # Initialize livestock details if not present
    if 'livestock_details' not in user:
        user['livestock_details'] = {
            'cow': [],
            'buffalo': [],
            'goat': [],
            'sheep': []
        }

    user.pop('password', None)
    return render_template('livestock_management.html', user=user)


def get_plural_form(animal_type):
    plural_forms = {
        'cow': 'cows',
        'buffalo': 'buffaloes',
        'goat': 'goats',
        'sheep': 'sheep'
    }
    return plural_forms.get(animal_type, animal_type + 's')


@app.route('/save_animal', methods=['POST'])
def save_animal():
    try:
        user_id = ObjectId(session['user_id'])
        user = users.find_one({'_id': user_id})

        animal_type = request.form['type']
        plural_type = get_plural_form(animal_type)

        # Initialize livestock_details if not present
        if 'livestock_details' not in user:
            user['livestock_details'] = {
                'cow': [], 'buffalo': [], 'goat': [], 'sheep': []
            }

        # Create animal data dictionary
        animal_data = {
            'breed': request.form['breed'],
            'age': float(request.form['age']),  # Age is now in years
            'weight': float(request.form['weight']),
            'color': request.form['color'],
            'gender': request.form['gender'],
            'health_status': request.form['health_status'],
            'notes': request.form.get('notes', '')
        }

        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                # Create upload directory if it doesn't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

                # Generate unique filename
                filename = secure_filename(file.filename)
                ext = filename.rsplit('.', 1)[1].lower()
                new_filename = f"{animal_type}_{request.form['id']}.{ext}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

                # Save the file
                file.save(file_path)
                animal_data['image_url'] = f"/static/uploads/animals/{new_filename}"

        # Check if we're editing or adding
        if request.form.get('editId'):
            # Editing existing animal
            animal_list = user['livestock_details'][animal_type]
            animal_index = next((index for (index, d) in enumerate(animal_list) if d["id"] == request.form['editId']),
                                None)

            if animal_index is not None:
                # Keep existing image if not changed
                if 'image_url' not in animal_data and 'image_url' in animal_list[animal_index]:
                    animal_data['image_url'] = animal_list[animal_index]['image_url']

                # Handle image removal
                if request.form.get('removeImage') == 'true':
                    if 'image_url' in animal_list[animal_index]:
                        try:
                            os.remove(os.path.join('static', animal_list[animal_index]['image_url'].lstrip('/')))
                        except:
                            pass
                    animal_data.pop('image_url', None)

                animal_data['id'] = request.form['editId']
                animal_list[animal_index] = animal_data
            else:
                return jsonify({'success': False, 'error': 'Animal not found'})
        else:
            # Adding new animal
            current_count = len(user['livestock_details'][animal_type])
            max_count = user['livestock'][plural_type]

            if current_count >= max_count:
                return jsonify({
                    'success': False,
                    'error': f'Cannot add more {plural_type}. Maximum limit ({max_count}) reached.'
                })

            # Add ID to animal data
            animal_data['id'] = request.form['id']

            # Verify ID is unique
            existing_ids = [animal['id'] for animal in user['livestock_details'][animal_type]]
            if animal_data['id'] in existing_ids:
                return jsonify({
                    'success': False,
                    'error': f'Animal ID {animal_data["id"]} already exists'
                })

            user['livestock_details'][animal_type].append(animal_data)

        # Update user document
        users.update_one(
            {'_id': user_id},
            {'$set': {'livestock_details': user['livestock_details']}}
        )

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error saving animal: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to save animal'})


@app.route('/get_animal/<animal_type>/<animal_id>')
@login_required
def get_animal(animal_type, animal_id):
    try:
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or 'livestock_details' not in user:
            return jsonify({'success': False, 'error': 'No livestock details found'})

        animal_list = user['livestock_details'].get(animal_type, [])
        animal = next((a for a in animal_list if str(a['id']) == str(animal_id)), None)

        if animal:
            return jsonify({'success': True, 'animal': animal})
        return jsonify({'success': False, 'error': 'Animal not found'})

    except Exception as e:
        print(f"Error getting animal: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/delete_animal', methods=['POST'])
@login_required
def delete_animal():
    try:
        data = request.get_json()
        user_id = ObjectId(session['user_id'])
        animal_type = data['type']
        animal_id = data['id']

        # Get user and verify animal exists
        user = users.find_one({'_id': user_id})
        if not user or 'livestock_details' not in user:
            return jsonify({'success': False, 'error': 'No livestock details found'})

        animal_list = user['livestock_details'].get(animal_type, [])
        animal = next((a for a in animal_list if str(a['id']) == str(animal_id)), None)

        if not animal:
            return jsonify({'success': False, 'error': 'Animal not found'})

        # Remove animal's image if it exists
        if 'image_url' in animal:
            try:
                os.remove(os.path.join('static', animal['image_url'].lstrip('/')))
            except Exception as e:
                print(f"Error removing image: {str(e)}")

        # Remove animal from list
        result = users.update_one(
            {'_id': user_id},
            {'$pull': {f'livestock_details.{animal_type}': {'id': animal_id}}}
        )

        if result.modified_count > 0:
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Failed to delete animal'})

    except Exception as e:
        print(f"Error deleting animal: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_animals/<animal_type>')
@login_required
def get_animals(animal_type):
    try:
        # Get current user's data
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or 'livestock_details' not in user:
            print("No livestock details found for user")
            return jsonify([])

        # Get animals of specified type
        animals = user.get('livestock_details', {}).get(animal_type, [])
        print(f"Found {len(animals)} animals of type {animal_type}")

        # Format animals for response
        formatted_animals = []
        for animal in animals:
            try:
                formatted_animal = {
                    'id': str(animal.get('id', '')),
                    'breed': str(animal.get('breed', 'Unknown')),
                    'age_years': "{:.1f}".format(float(animal.get('age', 0))),
                    'weight': str(animal.get('weight', 0))
                }
                formatted_animals.append(formatted_animal)
            except Exception as e:
                print(f"Error formatting animal: {str(e)}")
                continue

        print(f"Returning {len(formatted_animals)} formatted animals")
        return jsonify(formatted_animals)

    except Exception as e:
        print(f"Error in get_animals: {str(e)}")
        return jsonify([])


@app.route('/add_health_record', methods=['POST'])
@login_required
def add_health_record():
    try:
        data = request.get_json()

        # Debug print
        print("Received data:", data)
        print("User ID:", session['user_id'])

        # Validate required fields
        required_fields = ['animal_id', 'animal_type', 'symptoms', 'diagnosis', 'treatment', 'condition', 'location']
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            return jsonify({
                'success': False,
                'error': f"Missing required fields: {', '.join(missing_fields)}"
            })

        # Find the user and their livestock
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or 'livestock_details' not in user:
            return jsonify({
                'success': False,
                'error': "No livestock details found for user"
            })

        # Debug print
        print("User livestock details:", user.get('livestock_details'))

        # Check if animal type exists
        animal_type = data['animal_type'].lower()
        if animal_type not in user['livestock_details']:
            return jsonify({
                'success': False,
                'error': f"No {animal_type} found in your livestock"
            })

        # Find the specific animal
        animal_found = False
        for animal in user['livestock_details'][animal_type]:
            if str(animal['id']) == str(data['animal_id']):
                animal_found = True
                break

        if not animal_found:
            return jsonify({
                'success': False,
                'error': f"Animal ID {data['animal_id']} not found in your {animal_type} livestock"
            })

        # Create health record
        record = {
            'user_id': str(session['user_id']),
            'animal_id': str(data['animal_id']),
            'animal_type': animal_type,
            'symptoms': data['symptoms'].strip(),
            'diagnosis': data['diagnosis'].strip(),
            'treatment': data['treatment'].strip(),
            'condition': data['condition'].strip(),
            'location': data['location'].strip(),
            'notes': data['notes'].strip() if data.get('notes') else '',
            'timestamp': datetime.now()
        }

        # Insert into database
        health_records.insert_one(record)

        # Update the animal's health history
        result = users.update_one(
            {
                '_id': ObjectId(session['user_id']),
                f'livestock_details.{animal_type}': {
                    '$elemMatch': {
                        'id': str(data['animal_id'])
                    }
                }
            },
            {
                '$push': {
                    f'livestock_details.{animal_type}.$.health_history': {
                        'timestamp': datetime.now(),
                        'symptoms': record['symptoms'],
                        'diagnosis': record['diagnosis'],
                        'treatment': record['treatment'],
                        'condition': record['condition'],
                        'location': record['location'],
                        'notes': record['notes']
                    }
                }
            }
        )

        if result.modified_count == 0:
            print("Failed to update animal health history. Result:", result.raw_result)
            return jsonify({
                'success': False,
                'error': "Failed to update animal health history"
            })

        return jsonify({'success': True})

    except Exception as e:
        print(f"Error adding health record: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/get_health_history/<animal_type>/<animal_id>')
@login_required
def get_health_history(animal_type, animal_id):
    try:
        # Get all health records for this animal
        records = list(health_records.find(
            {
                'user_id': str(session['user_id']),
                'animal_type': animal_type,
                'animal_id': animal_id
            },
            {'_id': 0}
        ).sort('timestamp', -1))  # Sort by newest first

        # Format timestamps
        for record in records:
            record['timestamp'] = record['timestamp'].isoformat()

        return jsonify(records)
    except Exception as e:
        print(f"Error getting health history: {str(e)}")
        return jsonify([])


@app.route('/download_animal_record/<animal_type>/<animal_id>')
@login_required
def download_animal_record(animal_type, animal_id):
    try:
        # Find the animal
        user = users.find_one({
            '_id': ObjectId(session['user_id']),
            f'livestock_details.{animal_type}': {
                '$elemMatch': {
                    'id': animal_id
                }
            }
        })

        if not user or 'livestock_details' not in user:
            return "Animal not found", 404

        # Get the animal details
        animal = None
        for a in user['livestock_details'].get(animal_type, []):
            if a['id'] == animal_id:
                animal = a
                break

        if not animal:
            return "Animal not found", 404

        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        elements.append(Paragraph(f"Animal Record - {animal_type.title()} (ID: {animal_id})", title_style))

        # Basic Details
        elements.append(Paragraph("Basic Information", styles['Heading2']))
        basic_info = [
            ["Breed:", animal['breed']],
            ["Age:", f"{animal['age']} years"],
            ["Weight:", f"{animal['weight']} kg"]
        ]
        t = Table(basic_info, colWidths=[100, 300])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Health Records
        elements.append(Paragraph("Health Records", styles['Heading2']))
        if 'health_history' in animal and animal['health_history']:
            for record in sorted(animal['health_history'], key=lambda x: x['timestamp'], reverse=True):
                # Record date
                date_str = record['timestamp'].strftime('%Y-%m-%d %H:%M')
                elements.append(Paragraph(f"Date: {date_str}", styles['Heading3']))

                # Record details
                record_data = [
                    ["Symptoms:", record['symptoms']],
                    ["Diagnosis:", record['diagnosis']],
                    ["Treatment:", record['treatment']],
                    ["Condition:", record['condition']],
                    ["Location:", record['location']]
                ]
                if record.get('notes'):
                    record_data.append(["Notes:", record['notes']])

                t = Table(record_data, colWidths=[100, 300])
                t.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(t)
                elements.append(Spacer(1, 20))
        else:
            elements.append(Paragraph("No health records found.", styles['Normal']))

        # Build PDF
        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            download_name=f'animal_record_{animal_type}_{animal_id}.pdf',
            as_attachment=True,
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return "Error generating PDF", 500


@app.route('/analytics')
@login_required
def analytics_page():
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')
    return render_template('analytics.html', user=user)


@app.route('/disease-map')
@login_required
def view_disease_map():
    # Get disease types for filter dropdown
    disease_types = list(diseases_collection.distinct('disease_name'))
    return render_template('disease_map.html', diseases=disease_types)


@app.route('/api/patient-disease-data')
@login_required
def get_patient_disease_data():
    try:
        # Get filter parameters
        disease = request.args.get('disease', '')
        severity = request.args.get('severity', '')
        animal = request.args.get('animal', '')
        days = request.args.get('days', 'all')

        # Build query
        query = {}
        if disease:
            query['disease_name'] = disease
        if severity:
            query['severity'] = severity
        if animal:
            query['animal_type'] = animal

        # Add date filter if specified
        if days != 'all':
            try:
                days = int(days)
                cutoff_date = datetime.now() - timedelta(days=days)
                query['reported_date'] = {'$gte': cutoff_date}
            except ValueError:
                pass

        # Fetch disease reports with reporter information using aggregation pipeline
        pipeline = [
            {'$match': query},
            {
                '$lookup': {
                    'from': 'users',
                    'let': {'reporter_id': '$reported_by'},
                    'pipeline': [
                        {
                            '$match': {
                                '$expr': {'$eq': ['$_id', '$$reporter_id']}
                            }
                        },
                        {
                            '$project': {
                                'name': 1,
                                'is_doctor': 1,
                                'contact': 1,
                                '_id': 0
                            }
                        }
                    ],
                    'as': 'reporter'
                }
            },
            {
                '$addFields': {
                    'reporter_info': {'$arrayElemAt': ['$reporter', 0]}
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'disease_name': 1,
                    'severity': 1,
                    'animal_type': 1,
                    'latitude': 1,
                    'longitude': 1,
                    'cases': 1,
                    'notes': 1,
                    'reported_date': 1,
                    'reporter_name': '$reporter_info.name',
                    'reporter_type': {
                        '$cond': {
                            'if': '$reporter_info.is_doctor',
                            'then': 'Doctor',
                            'else': 'Owner'
                        }
                    },
                    'contact': '$reporter_info.contact'
                }
            },
            {'$sort': {'reported_date': -1}}  # Sort by newest first
        ]

        # Execute the aggregation pipeline using the 'disease' collection
        disease_data = list(db.disease.aggregate(pipeline))

        # Convert datetime objects to string for JSON serialization
        for report in disease_data:
            if 'reported_date' in report:
                report['reported_date'] = report['reported_date'].isoformat()

        # Log the number of reports found
        app.logger.info(f"Found {len(disease_data)} disease reports matching the criteria")

        return jsonify(disease_data)

    except Exception as e:
        app.logger.error(f"Error in get_patient_disease_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Get list of all diseases for the dropdown"""
    try:
        diseases = list(db.diseases.find({}, {'_id': 0, 'disease_name': 1}))
        return jsonify([d['disease_name'] for d in diseases])
    except Exception as e:
        app.logger.error(f"Error fetching diseases: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Get selected symptoms
        selected_symptoms = request.form.getlist('symptoms')

        if not selected_symptoms:
            return jsonify({
                'success': False,
                'error': "Please select at least one symptom"
            }), 400

        # Ensure we have exactly 3 symptoms (pad with None if needed)
        while len(selected_symptoms) < 3:
            selected_symptoms.append(None)
        # Take only the first 3 symptoms if more are selected
        selected_symptoms = selected_symptoms[:3]

        try:
            # Encode each symptom (handle None values)
            encoded_symptoms = []
            for symptom in selected_symptoms:
                if symptom is None:
                    # Use the first symptom in the encoder as a default
                    encoded_symptoms.append(0)
                else:
                    encoded_symptoms.append(encoder_symptoms.transform([symptom])[0])

            # Make prediction using the disease prediction model
            prediction = disease_model.predict([encoded_symptoms])[0]
            predicted_disease = encoder_disease.inverse_transform([prediction])[0]

            # Get disease information
            disease_details = disease_info.get(predicted_disease, {
                "description": "Detailed information not available for this condition.",
                "precautions": [
                    "Isolate the animal",
                    "Contact a veterinarian",
                    "Monitor vital signs",
                    "Keep detailed health records"
                ]
            })

            # Store prediction in MongoDB
            prediction_data = {
                "timestamp": datetime.now(),
                "symptoms": [s for s in selected_symptoms if s is not None],
                "predicted_disease": predicted_disease,
                "details": disease_details
            }

            # Add photo information if available
            if 'last_uploaded_photo' in request.form:
                prediction_data['photo'] = request.form['last_uploaded_photo']

            predictions_collection.insert_one(prediction_data)

            return jsonify({
                'success': True,
                'disease': predicted_disease,
                'symptoms': [s for s in selected_symptoms if s is not None],
                'details': disease_details
            }), 200

        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f"Invalid symptom selected. Please choose from the provided list. Error: {str(e)}"
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'error': 'No photo selected'}), 400

    if photo and allowed_file(photo.filename):
        try:
            # Save the photo
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + photo.filename
            filepath = os.path.join(UPLOAD_FOLDER_PREDICTION, filename)
            photo.save(filepath)

            # Read the image for analysis
            with open(filepath, 'rb') as image_file:
                content = image_file.read()
                image = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(content).decode('utf-8')
                }

            # Create prompt for the model
            prompt = """Analyze this image of livestock/cattle and provide detailed information about any visible health issues. Consider:
1. The animal's overall appearance
2. Any visible symptoms or abnormalities
3. Physical condition and behavior
4. Signs of distress or illness

Format the response as JSON with the following structure:
{
    "animal_type": "type of animal",
    "visible_symptoms": ["detailed symptom 1", "detailed symptom 2", ...],
    "condition": "detailed assessment of overall condition",
    "health_concerns": ["specific health concern 1", "specific health concern 2", ...]
}"""

            # Generate content using Gemini model
            response = gemini_model.generate_content(
                contents=[prompt, image],
                stream=False
            )

            try:
                # Extract JSON from response
                analysis_text = response.text
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    analysis_json = json.loads(analysis_text[json_start:json_end])
                else:
                    # If JSON parsing fails, create structured response from text
                    lines = analysis_text.split('\n')
                    analysis_json = {
                        "animal_type": "Cattle",
                        "visible_symptoms": [line.strip('- ') for line in lines if line.strip().startswith('-')],
                        "condition": "Analysis completed",
                        "health_concerns": []
                    }

                # Match symptoms with our symptom list
                matched_symptoms = []
                for ai_symptom in analysis_json.get('visible_symptoms', []):
                    for known_symptom in symptom_list:
                        if (ai_symptom.lower() in known_symptom.lower() or
                                known_symptom.lower() in ai_symptom.lower()):
                            matched_symptoms.append(known_symptom)

                analysis_json['matched_symptoms'] = list(set(matched_symptoms))

                return jsonify({
                    'success': True,
                    'filename': filename,
                    'analysis': analysis_json,
                    'all_symptoms': symptom_list
                }), 200

            except Exception as e:
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'analysis': {
                        'animal_type': 'Cattle',
                        'visible_symptoms': [],
                        'condition': 'Analysis completed',
                        'health_concerns': [],
                        'matched_symptoms': []
                    },
                    'all_symptoms': symptom_list,
                    'error': str(e)
                }), 200

        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image_path' not in request.json:
            return jsonify({'error': 'No image path provided'}), 400

        image_path = request.json['image_path']

        # Read the image file
        with open(os.path.join(UPLOAD_FOLDER_PREDICTION, image_path), 'rb') as image_file:
            content = image_file.read()
            image = {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(content).decode('utf-8')
            }

        # Use Gemini to analyze the image
        prompt = """Analyze this veterinary image and identify:
        1. Any visible symptoms or abnormalities
        2. The location of these symptoms on the animal
        3. The confidence level of each detection
        
        Format the response as a list of detections, each with:
        - Description of the symptom/condition
        - Location (as normalized coordinates: left, top, width, height)
        - Confidence score (0-1)"""

        response = image_model.generate_content([prompt, image])
        result = response.text

        # Process the model's response to extract detections
        detections = []
        try:
            # Parse the model's response to extract detections
            # This is a simplified example - you'll need to adjust based on actual response format
            for line in result.split('\n'):
                if ':' in line and '(' in line and ')' in line:
                    # Extract description and coordinates
                    desc_part = line.split('(')[0].strip()
                    coord_part = line.split('(')[1].split(')')[0]
                    
                    # Parse coordinates (assuming format: left, top, width, height)
                    coords = [float(x) for x in coord_part.split(',')]
                    
                    detection = {
                        'name': desc_part,
                        'boundingBox': {
                            'left': coords[0],
                            'top': coords[1],
                            'width': coords[2],
                            'height': coords[3]
                        },
                        'confidence': 0.9  # You might get this from the model
                    }
                    detections.append(detection)
        except Exception as e:
            print(f"Error parsing detections: {str(e)}")
            # Fallback to basic analysis if parsing fails
            detections = []

        # Get general analysis from Gemini
        analysis_prompt = "What symptoms or health conditions do you observe in this veterinary image? Please provide a detailed analysis."
        analysis_response = image_model.generate_content([analysis_prompt, image])
        analysis_text = analysis_response.text

        return jsonify({
            'success': True,
            'objects': detections,
            'analysis': analysis_text
        })

    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/records')
@login_required
def records_page():
    if 'user_id' not in session:
        return redirect('/login')
    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        return redirect('/logout')
    user.pop('password', None)
    return render_template('records.html', user=user)


@app.route('/get_all_records')
@login_required
def get_all_records():
    try:
        user_id = str(session['user_id'])

        print(f"Looking for records for user: {user_id}")  # Debug print

        # Get user's livestock details to check for deleted animals
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        livestock_details = user.get('livestock_details', {})

        # Create a set of active animal IDs
        active_animal_ids = set()
        for animal_type in livestock_details:
            for animal in livestock_details[animal_type]:
                if not animal.get('is_deleted', False):
                    active_animal_ids.add(str(animal['id']))

        print(f"Found {len(active_animal_ids)} active animals")  # Debug print
        print("Active animal IDs:", active_animal_ids)  # Debug print

        # Get health records for active animals only
        health_records_list = list(health_records.find(
            {
                'user_id': user_id,
                'animal_id': {'$in': list(active_animal_ids)}
            },
            {
                '_id': 0,
                'animal_id': 1,
                'animal_type': 1,
                'symptoms': 1,
                'condition': 1,
                'timestamp': 1
            }
        ).sort('timestamp', -1))  # Sort by newest first

        # Format health records
        formatted_health_records = []
        for record in health_records_list:
            try:
                formatted_record = {
                    'animal_id': record['animal_id'],
                    'animal_type': record['animal_type'],
                    'record_type': 'health',
                    'details': record.get('symptoms', ''),
                    'status': record.get('condition', 'Unknown'),
                    'timestamp': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(record['timestamp'],
                                                                                                datetime) else str(
                        record['timestamp'])
                }
                formatted_health_records.append(formatted_record)
            except Exception as e:
                print(f"Error formatting health record: {str(e)}")
                print("Record data:", record)
                continue

        print(f"Found {len(formatted_health_records)} health records")  # Debug print

        # Get vaccination records for active animals only
        vaccination_records = list(vaccinations.find(
            {
                'user_id': user_id,
                'animal_id': {'$in': list(active_animal_ids)}
            },
            {
                '_id': 0,
                'animal_id': 1,
                'animal_type': 1,
                'vaccinationType': 1,
                'date': 1,
                'nextDueDate': 1
            }
        ).sort('date', -1))

        # Format vaccination records
        formatted_vaccination_records = []
        for record in vaccination_records:
            try:
                formatted_record = {
                    'animal_id': record['animal_id'],
                    'animal_type': record['animal_type'],
                    'record_type': 'vaccination',
                    'details': f"Vaccine: {record['vaccinationType']}, Next Due: {record['nextDueDate']}",
                    'status': 'Completed',
                    'date': record['date']
                }
                formatted_vaccination_records.append(formatted_record)
            except Exception as e:
                print(f"Error formatting vaccination record: {str(e)}")
                print("Record data:", record)
                continue

        print(f"Found {len(formatted_vaccination_records)} vaccination records")  # Debug print

        # Get upcoming vaccination appointments for active animals only
        current_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Current date: {current_date}")  # Debug print

        # Get upcoming appointments with proper field names
        upcoming_appointments = list(appointments.find({
            'user_id': user_id,
            'status': 'scheduled'
        }).sort('appointmentDate', 1))

        print(f"\nFound {len(upcoming_appointments)} scheduled appointments")
        for appt in upcoming_appointments:
            print(f"Scheduled appointment: {appt}")
            print(f"Appointment date: {appt.get('appointmentDate')}")
            print(f"Animal ID: {appt.get('animal_id')}")  # Updated field name
            print(f"Status: {appt.get('status')}")
            print("---")

        # Format appointment records
        formatted_appointments = []
        for appt in upcoming_appointments:
            try:
                # Debug prints for each appointment
                print(f"\nProcessing appointment:")
                print(f"Raw appointment data: {appt}")

                # Get and validate appointment date
                appt_date_str = appt.get('appointmentDate')
                print(f"Appointment date string: {appt_date_str}")

                if not appt_date_str:
                    print("Skipping appointment - no date")
                    continue

                # Parse the appointment date
                try:
                    appt_date = datetime.strptime(appt_date_str, '%Y-%m-%d')
                    formatted_date = appt_date.strftime('%Y-%m-%d')
                    print(f"Parsed appointment date: {formatted_date}")
                except ValueError as e:
                    print(f"Error parsing date {appt_date_str}: {e}")
                    continue

                # Get animal details
                animal_id = str(appt.get('animal_id', ''))  # Updated field name
                animal_type = appt.get('animal_type', '').lower()  # Updated field name
                print(f"Animal ID: {animal_id}, Type: {animal_type}")

                # Check if this is a future appointment
                if appt_date.date() >= datetime.now().date():
                    formatted_record = {
                        'animal_id': animal_id,
                        'animal_type': animal_type,
                        'record_type': 'vaccination',
                        'details': f"Scheduled vaccination: {appt.get('vaccinationType', 'Unknown')}",
                        'status': 'Scheduled',
                        'date': formatted_date
                    }
                    formatted_appointments.append(formatted_record)
                    print(f"Added appointment to records: {formatted_record}")
                else:
                    print(f"Skipping past appointment: {formatted_date}")

            except Exception as e:
                print(f"Error processing appointment: {str(e)}")
                print(f"Problematic appointment data: {appt}")
                continue

        print(f"\nFinal formatted appointments: {len(formatted_appointments)}")
        for appt in formatted_appointments:
            print(f"Formatted appointment: {appt}")

        # Combine all records and sort by date
        all_records = formatted_health_records + formatted_vaccination_records + formatted_appointments

        # Sort records by date, handling both timestamp and date fields
        def get_record_date(record):
            date_str = record.get('date') or record.get('timestamp')
            try:
                if isinstance(date_str, str):
                    # Try different date formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                return datetime.now()  # Default to current date if parsing fails
            except Exception as e:
                print(f"Error parsing date {date_str}: {str(e)}")
                return datetime.now()

        all_records.sort(key=get_record_date, reverse=True)

        print(f"\nTotal records being returned: {len(all_records)}")

        return jsonify({
            'success': True,
            'records': all_records,
            'total_records': len(all_records)
        })

    except Exception as e:
        print(f"Error in get_all_records: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/export_records')
@login_required
def export_records():
    try:
        user_id = str(session['user_id'])

        # Create a StringIO object to write CSV data
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Date', 'Animal ID', 'Animal Type', 'Record Type', 'Details', 'Status'])

        # Get all records using the existing function
        records_response = get_all_records()
        records_data = json.loads(records_response.get_data(as_text=True))

        if records_data['success']:
            # Write records to CSV
            for record in records_data['records']:
                writer.writerow([
                    record.get('date') or record.get('timestamp'),
                    record['animal_id'],
                    record['animal_type'],
                    record['record_type'],
                    record['details'],
                    record.get('status', '')
                ])

            # Prepare the response
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={
                    'Content-Disposition': 'attachment; filename=animal_records.csv',
                    'Content-Type': 'text/csv'
                }
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to get records'
            }), 500

    except Exception as e:
        print(f"Error in export_records: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/debug_appointments')
@login_required
def debug_appointments():
    try:
        user_id = str(session['user_id'])
        print(f"Debugging appointments for user: {user_id}")

        # Get all appointments
        all_appointments = list(appointments.find({}))
        print("\nAll appointments in database:")
        for appt in all_appointments:
            print(f"Appointment: {appt}")

        # Get user's appointments
        user_appointments = list(appointments.find({'user_id': user_id}))
        print(f"\nAppointments for user {user_id}:")
        for appt in user_appointments:
            print(f"User appointment: {appt}")

        # Get scheduled appointments
        scheduled_appointments = list(appointments.find({
            'user_id': user_id,
            'status': 'scheduled'
        }))
        print(f"\nScheduled appointments for user {user_id}:")
        for appt in scheduled_appointments:
            print(f"Scheduled appointment: {appt}")

        return jsonify({
            'all_appointments': all_appointments,
            'user_appointments': user_appointments,
            'scheduled_appointments': scheduled_appointments
        })
    except Exception as e:
        print(f"Error in debug_appointments: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/marketplace')
@login_required
def marketplace():
    try:
        # Get all listings
        listings = list(marketplace_listings.find().sort('created_at', -1))
        for listing in listings:
            listing['_id'] = str(listing['_id'])

        # Get user's animals from the users collection
        user = users.find_one({'_id': ObjectId(session['user_id'])})
        user_animals = []

        if user and 'livestock_details' in user:
            # Flatten the livestock_details structure into a list of animals
            for animal_type, animals in user['livestock_details'].items():
                for animal in animals:
                    animal_copy = animal.copy()
                    animal_copy['type'] = animal_type
                    animal_copy['_id'] = f"{animal_type}|{animal['id']}"  # Create a composite ID
                    user_animals.append(animal_copy)

        print(f"\nFound {len(user_animals)} animals for user {session['user_id']}")
        print("User animals:", user_animals)

        return render_template('marketplace.html', listings=listings, user_animals=user_animals)
    except Exception as e:
        print(f"Error in marketplace route: {str(e)}")
        flash(f'Error loading marketplace: {str(e)}', 'danger')
        return redirect(url_for('home'))


@app.route('/api/marketplace/listings', methods=['POST'])
@login_required
def create_listing():
    try:
        # Handle image upload
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                filepath = os.path.join(UPLOAD_FOLDER_MARKETPLACE, filename)
                file.save(filepath)
                image_url = f"/static/marketplace_images/{filename}"

        # Get animal details from the user's livestock_details
        animal_data = request.form.get('animal_id').split('|')
        if len(animal_data) != 2:
            return jsonify({'message': 'Invalid animal selection'}), 400

        animal_type, animal_id = animal_data
        user = users.find_one({'_id': ObjectId(session['user_id'])})

        if not user or 'livestock_details' not in user:
            return jsonify({'message': 'User has no livestock details'}), 404

        # Find the animal in the user's livestock_details
        animal = None
        for a in user['livestock_details'].get(animal_type, []):
            if a['id'] == animal_id:
                animal = a
                break

        if not animal:
            return jsonify({'message': 'Animal not found or you do not own this animal'}), 404

        # Create listing with initialized likes and shares arrays
        listing = {
            'animal_id': animal['id'],
            'animal_type': animal_type,
            'price': float(request.form.get('price')),
            'phone_number': request.form.get('phone_number'),
            'description': request.form.get('description'),
            'image_url': image_url,
            'owner_id': session['user_id'],
            'owner_name': session['username'],
            'created_at': datetime.now(),
            'animal_details': {
                'breed': animal.get('breed'),
                'age': animal.get('age'),
                'weight': animal.get('weight'),
                'gender': animal.get('gender')
            },
            'likes': [],  # Initialize empty likes array
            'shares': 0   # Initialize shares count
        }

        result = marketplace_listings.insert_one(listing)
        return jsonify({'message': 'Listing created successfully', 'listing_id': str(result.inserted_id)}), 201
    except Exception as e:
        print(f"Error in create_listing: {str(e)}")
        return jsonify({'message': f'Error creating listing: {str(e)}'}), 500


@app.route('/api/marketplace/listings/<listing_id>', methods=['DELETE'])
@login_required
def delete_listing(listing_id):
    try:
        # Verify the listing belongs to the current user
        listing = marketplace_listings.find_one({'_id': ObjectId(listing_id)})
        if not listing:
            return jsonify({'message': 'Listing not found'}), 404

        if listing['owner_id'] != session['user_id']:
            return jsonify({'message': 'Unauthorized to delete this listing'}), 403

        # Delete the image if it exists
        if listing.get('image_url'):
            image_path = os.path.join(app.root_path, listing['image_url'].lstrip('/'))
            if os.path.exists(image_path):
                os.remove(image_path)

        # Delete the listing
        result = marketplace_listings.delete_one({'_id': ObjectId(listing_id)})
        return jsonify({'message': 'Listing deleted successfully'}), 200
    except Exception as e:
        return jsonify({'message': f'Error deleting listing: {str(e)}'}), 500


@app.route('/discussion_forum')
@login_required
def discussion_forum():
    try:
        # Get all posts with their comments, sorted by creation time (newest first)
        posts_list = list(posts.find().sort('created_at', -1))
        current_user_id = str(session['user_id'])

        for post in posts_list:
            post['_id'] = str(post['_id'])
            post['created_at'] = post['created_at'].strftime('%Y-%m-%d %H:%M')
            # Add likes information
            post['likes'] = post.get('likes', [])
            post['like_count'] = len(post['likes'])
            post['is_liked'] = current_user_id in post['likes']
            
            for comment in post.get('comments', []):
                comment['created_at'] = comment['created_at'].strftime('%Y-%m-%d %H:%M')

        return render_template('discussion_forum.html', posts=posts_list)
    except Exception as e:
        flash(f'Error loading discussion forum: {str(e)}', 'danger')
        return redirect(url_for('home'))


@app.route('/api/discussion_forum/posts', methods=['POST'])
@login_required
def create_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        post = {
            'title': data.get('title'),
            'content': data.get('content'),
            'author': session.get('username'),
            'author_id': session.get('user_id'),
            'created_at': datetime.now(),
            'comments': [],
            'likes': []  # Initialize empty likes array
        }

        result = posts.insert_one(post)
        return jsonify({'success': True, 'post_id': str(result.inserted_id)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/discussion_forum/posts/<post_id>/comments', methods=['POST'])
@login_required
def add_comment(post_id):
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'message': 'Missing comment content'}), 400

        comment = {
            'content': data['content'],
            'author': session['username'],
            'created_at': datetime.now()
        }

        result = db.posts.update_one(
            {'_id': ObjectId(post_id)},
            {'$push': {'comments': comment}}
        )

        if result.modified_count == 0:
            return jsonify({'message': 'Post not found'}), 404

        return jsonify({'message': 'Comment added successfully'}), 201
    except Exception as e:
        return jsonify({'message': f'Error adding comment: {str(e)}'}), 500


@app.route('/api/discussion_forum/posts/<post_id>', methods=['DELETE'])
@login_required
def delete_post(post_id):
    try:
        # Verify the post belongs to the current user
        post = db.posts.find_one({'_id': ObjectId(post_id)})
        if not post:
            return jsonify({'message': 'Post not found'}), 404

        if post['author'] != session['username']:
            return jsonify({'message': 'Unauthorized to delete this post'}), 403

        result = db.posts.delete_one({'_id': ObjectId(post_id)})
        return jsonify({'message': 'Post deleted successfully'}), 200
    except Exception as e:
        return jsonify({'message': f'Error deleting post: {str(e)}'}), 500


@app.route('/api/disease-data')
def get_disease_data():
    try:
        # Get filter parameters
        disease = request.args.get('disease', '')
        severity = request.args.get('severity', '')
        state = request.args.get('state', '')
        days = request.args.get('days', 'all')

        # Build query
        query = {}
        if disease:
            query['disease_name'] = disease
        if severity:
            query['severity'] = severity
        if state:
            query['state'] = state  # Add state to query if selected

        # Add date filter if specified
        if days != 'all':
            try:
                days = int(days)
                cutoff_date = datetime.now() - timedelta(days=days)
                query['reported_date'] = {'$gte': cutoff_date}
            except ValueError:
                pass

        # Get disease reports from the diseases collection
        reports = list(diseases_collection.find(query))

        # Format the response data
        formatted_reports = []
        for report in reports:
            formatted_report = {
                'lat': report.get('latitude'),
                'lng': report.get('longitude'),
                'diagnosis': report.get('disease_name'),
                'condition': report.get('severity'),
                'cases': report.get('cases', 1),
                'timestamp': report.get('reported_date'),
                'location': report.get('location', 'Unknown'),
                'state': report.get('state', 'Unknown')  # Include state in response
            }
            formatted_reports.append(formatted_report)

        print(f"Found {len(formatted_reports)} disease reports")  # Debug print
        return jsonify(formatted_reports)

    except Exception as e:
        print(f"Error in get_disease_data: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500


@app.route('/api/report-disease', methods=['POST'])
@login_required
def report_disease():
    try:
        # Check if user is a doctor
        if not session.get('is_doctor'):
            return jsonify({'success': False, 'error': 'Only doctors can report diseases'}), 403

        # Check if the request has JSON data
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Invalid request format'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['disease_name', 'animal_type', 'severity', 'cases', 'latitude', 'longitude']
        missing_fields = [field for field in required_fields if not data.get(field)]

        if missing_fields:
            return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Validate data types
        try:
            cases = int(data['cases'])
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid number format for cases, latitude, or longitude'}), 400

        # Validate value ranges
        if cases < 1:
            return jsonify({'success': False, 'error': 'Number of cases must be at least 1'}), 400
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return jsonify({'success': False, 'error': 'Invalid latitude or longitude values'}), 400

        # Create disease report
        disease_report = {
            'disease_name': data['disease_name'],
            'animal_type': data['animal_type'],
            'severity': data['severity'],
            'cases': cases,
            'latitude': latitude,
            'longitude': longitude,
            'notes': data.get('notes', ''),
            'reported_date': datetime.now(),
            'reported_by': ObjectId(session['user_id'])
        }

        # Insert the report
        result = db.disease.insert_one(disease_report)

        if result.inserted_id:
            return jsonify({
                'success': True,
                'message': 'Disease case reported successfully',
                'report_id': str(result.inserted_id)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save report'}), 500

    except Exception as e:
        app.logger.error(f"Error in report_disease: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


# Add this route temporarily to check your data
@app.route('/check_diseases')
def check_diseases():
    try:
        all_diseases = list(diseases_collection.find())
        formatted_diseases = []
        for disease in all_diseases:
            disease['_id'] = str(disease['_id'])  # Convert ObjectId to string
            if isinstance(disease.get('reported_date'), datetime):
                disease['reported_date'] = disease['reported_date'].isoformat()
            formatted_diseases.append(disease)
        return jsonify({
            'count': len(formatted_diseases),
            'diseases': formatted_diseases
        })
    except Exception as e:
        return jsonify({'error': str(e)})


# Add this route temporarily to insert test data
@app.route('/add_test_disease')
def add_test_disease():
    try:
        test_disease = {
            'disease_name': 'Test Disease',
            'severity': 'Severe',
            'latitude': 20.5937,
            'longitude': 78.9629,
            'cases': 5,
            'reported_date': datetime.now(),
            'location': 'Test Location',
            'animal_type': 'cow',
            'state': 'Maharashtra'  # Add state to test data
        }
        result = diseases_collection.insert_one(test_disease)
        return jsonify({
            'success': True,
            'message': 'Test disease added',
            'id': str(result.inserted_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/marketplace/listings/<listing_id>/like', methods=['POST'])
@login_required
def like_listing(listing_id):
    try:
        user_id = str(session['user_id'])
        
        # Check if user has already liked
        listing = marketplace_listings.find_one({'_id': ObjectId(listing_id)})
        if not listing:
            return jsonify({'success': False, 'error': 'Listing not found'}), 404

        # Initialize likes array if it doesn't exist
        likes = listing.get('likes', [])
        
        if user_id in likes:
            # Unlike
            marketplace_listings.update_one(
                {'_id': ObjectId(listing_id)},
                {
                    '$pull': {'likes': user_id},
                    '$setOnInsert': {'shares': 0}  # Ensure shares field exists
                }
            )
            return jsonify({
                'success': True, 
                'liked': False, 
                'count': len(likes) - 1
            })
        else:
            # Like
            marketplace_listings.update_one(
                {'_id': ObjectId(listing_id)},
                {
                    '$addToSet': {'likes': user_id},  # Use addToSet instead of push to avoid duplicates
                    '$setOnInsert': {'shares': 0}  # Ensure shares field exists
                }
            )
            return jsonify({
                'success': True, 
                'liked': True, 
                'count': len(likes) + 1
            })

    except Exception as e:
        print(f"Error in like_listing: {str(e)}")  # Add debug print
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketplace/listings/<listing_id>/share', methods=['POST'])
@login_required
def share_listing(listing_id):
    try:
        listing = marketplace_listings.find_one({'_id': ObjectId(listing_id)})
        if not listing:
            return jsonify({'success': False, 'error': 'Listing not found'}), 404

        # Increment share count
        shares = listing.get('shares', 0) + 1
        marketplace_listings.update_one(
            {'_id': ObjectId(listing_id)},
            {'$set': {'shares': shares}}
        )

        # Generate share URL
        share_url = request.host_url + f"marketplace/listings/{listing_id}"
        
        return jsonify({
            'success': True, 
            'shares': shares,
            'shareUrl': share_url
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/discussion_forum/posts/<post_id>/like', methods=['POST'])
@login_required
def like_post(post_id):
    try:
        user_id = str(session['user_id'])
        
        # Check if post exists
        post = posts.find_one({'_id': ObjectId(post_id)})
        if not post:
            return jsonify({'success': False, 'error': 'Post not found'}), 404

        # Initialize likes array if it doesn't exist
        likes = post.get('likes', [])
        
        if user_id in likes:
            # Unlike
            posts.update_one(
                {'_id': ObjectId(post_id)},
                {
                    '$pull': {'likes': user_id}
                }
            )
            return jsonify({
                'success': True, 
                'liked': False, 
                'count': len(likes) - 1
            })
        else:
            # Like
            posts.update_one(
                {'_id': ObjectId(post_id)},
                {
                    '$addToSet': {'likes': user_id}  # Use addToSet to avoid duplicates
                }
            )
            return jsonify({
                'success': True, 
                'liked': True, 
                'count': len(likes) + 1
            })

    except Exception as e:
        print(f"Error in like_post: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/discussion_forum/posts/<post_id>')
@login_required
def view_post(post_id):
    try:
        post = posts.find_one({'_id': ObjectId(post_id)})
        if not post:
            flash('Post not found', 'error')
            return redirect(url_for('discussion_forum'))
            
        # Format post data
        post['_id'] = str(post['_id'])
        post['created_at'] = post['created_at'].strftime('%Y-%m-%d %H:%M')
        post['likes'] = post.get('likes', [])
        post['like_count'] = len(post['likes'])
        post['is_liked'] = str(session['user_id']) in post['likes']
        
        for comment in post.get('comments', []):
            comment['created_at'] = comment['created_at'].strftime('%Y-%m-%d %H:%M')
            
        return render_template('post_detail.html', post=post)
    except Exception as e:
        flash(f'Error loading post: {str(e)}', 'error')
        return redirect(url_for('discussion_forum'))




if __name__ == '__main__':
    app.run(debug=False)

