# from flask import Flask, request, jsonify
# from transformers import VisionEncoderDecoderModel, TrOCRProcessor
# from PIL import Image
# import torch
# import io

# # Initialize Flask app
# app = Flask(__name__)

# # Set device (CUDA if available, else CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the trained model and processor
# model_path = "best_trocr_model"
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
# model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# @app.route('/')
# def home():
#     return app.send_static_file('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse request
#         if 'image' not in request.files:
#             return jsonify({"error": "No image file provided"}), 400

#         image_file = request.files['image']
#         txt_file = request.files.get('txt', None)  # Optional txt file for bounding boxes

#         # Load and preprocess the image
#         image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

#         results = []

#         if txt_file:
#             txt_content = txt_file.read().decode('utf-8')
#             for line in txt_content.strip().split('\n'):
#                 parts = line.split('####')
#                 if len(parts) != 2:
#                     continue
#                 coords, label = parts
                
#                 print(coords)
#                 coords = [int(x) for x in coords.split(',')]
#                 x1, y1 = min(coords[::2]), min(coords[1::2])
#                 x2, y2 = max(coords[::2]), max(coords[1::2])
#                 cropped_image = image.crop((x1, y1, x2, y2))
#                 print(cropped_image)

#                 # Process the cropped image
#                 pixel_values = processor(cropped_image, return_tensors="pt").pixel_values.to(device)

#                 # Generate text prediction
#                 output_ids = model.generate(pixel_values, max_length=64, num_beams=4, early_stopping=True)
#                 predicted_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

#                 results.append({"label": label, "predicted_text": predicted_text})
#         else:
#             # Process the entire image if no bounding box txt file provided
#             pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
#             output_ids = model.generate(pixel_values, max_length=64, num_beams=4, early_stopping=True)
#             predicted_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
#             results.append({"predicted_text": predicted_text})

#         # Return the result
#         return jsonify(results)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


#ORIGINAL
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from flask_mysqldb import MySQL
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import io
import os

app = Flask(__name__)

# Secret key for session management
app.secret_key = 'your_secret_key'

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Vvit@123'
app.config['MYSQL_DB'] = 'ocr_app'
app.config['MYSQL_PORT'] = 3306

mysql = MySQL(app)

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load TrOCR model
model_path = "best_trocr_model"
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        cursor.close()

        if user:
            session['username'] = username
            return render_template('index.html', username=session['username'])
        else:
            flash('Invalid Credentials!')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        # Check if username or email already exists
        cursor.execute("SELECT * FROM users WHERE username=%s OR email=%s", (username, email))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            flash('You already have an account! Please login.')
            return redirect(url_for('login'))
        else:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
            mysql.connection.commit()
            cursor.close()
            flash('Account created successfully! Please login.')
            return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

# @app.route('/')
# def home():
#     return render_template('index.html')
@app.route('/about')
def about():
    return app.send_static_file('AboutUs.html')

@app.route('/feedback')
def feedback():
    return app.send_static_file('FeedbackForm.html')

# Serve video files from the static folder
@app.route('/avatars/<filename>')
def serve_avatar(filename):
    return send_from_directory('avatars', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'username' not in session:
            return redirect(url_for('login'))
    
        # Parse request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        txt_file = request.files.get('txt', None)  # Optional txt file for bounding boxes

        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        results = []

        if txt_file:
            txt_content = txt_file.read().decode('utf-8')
            for line in txt_content.strip().split('\n'):
                parts = line.split('####')
                if len(parts) != 2:
                    continue
                coords, label = parts
                coords = coords.strip()
                label = label.strip()

                print(f"Raw coordinates: {coords}")
                try:
                    # Convert coordinates to integers, skipping invalid or empty values
                    coords = [int(x) for x in coords.split(',') if x.strip().isdigit()]
                except ValueError as ve:
                    print(f"Invalid coordinates: {coords}. Error: {ve}")
                    continue  # Skip this line if coordinates are invalid

                if len(coords) < 4:
                    print(f"Insufficient coordinates: {coords}")
                    continue  # Ensure at least four values are present

                # Calculate bounding box
                x1, y1 = min(coords[::2]), min(coords[1::2])
                x2, y2 = max(coords[::2]), max(coords[1::2])
                cropped_image = image.crop((x1, y1, x2, y2))
                print(f"Cropped image bounds: {(x1, y1, x2, y2)}")

                # Process the cropped image
                pixel_values = processor(cropped_image, return_tensors="pt").pixel_values.to(device)

                # Generate text prediction
                output_ids = model.generate(pixel_values, max_length=64, num_beams=4, early_stopping=True)
                predicted_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                results.append(predicted_text)
        else:
            # Process the entire image if no bounding box txt file provided
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            output_ids = model.generate(pixel_values, max_length=64, num_beams=4, early_stopping=True)
            predicted_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append(predicted_text)
            
        # Return the result
        return "\n".join(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

