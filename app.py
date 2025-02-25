from flask import Flask, render_template, jsonify, request
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import torch
import io

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Set device (CUDA if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model and processor
model_path = "best_trocr_model"
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

# Routes for rendering HTML pages
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('AboutUs.html')

@app.route('/feedback')
def feedback():
    return render_template('FeedbackForm.html')

# Image processing and OCR prediction
@app.route('/submit', methods=['POST'])
def submit():
    try:
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
