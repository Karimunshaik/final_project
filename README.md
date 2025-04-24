📌 Project Overview
Project Title: Scene Text Recognition Using Vision Transformer and Transformer Decoder
Goal: Recognize and transcribe text from natural scene images using an end-to-end transformer-based model.
Model Used: TrOCR (VisionEncoderDecoderModel – Vision Transformer + Transformer Decoder)
Interface: Flask-based web application
Input: Images with scene text
Output: Predicted text transcriptions
🛠️ Requirements
Install dependencies using:

pip install -r requirements.txt

If `requirements.txt` is not available:

    pip install flask torch torchvision transformers numpy pandas scikit-learn opencv-python
🗂️ Project Structure
scene_text_recognition/
├── app.py                      # Flask application
├── train_trocr.py              # Model training script
├── trocr_model.pkl             # Pretrained TrOCR model
├── CTW1500_dataset/            # Dataset used for training and testing
├── static/                     # Static resources (if any)
├── templates/
│   ├── index.html              # Upload and result page
├── utils.py                    # Preprocessing and helper functions
└── README.md                   # Project documentation
🚀 How to Run the Project
1. Clone and Navigate
    cd scene_text_recognition

2. Install Dependencies

3. Train the Model (optional)
    python train_trocr.py

4. Run the Flask App
    flask --app app run

5. Open in Browser
    Visit http://127.0.0.1:5000/ to upload an image and get text predictions.
🔄 To Train the Model (Optional)
Retrain the model using:
    python train_trocr.py
📊 Sample Input/Output
Input: Upload image of a street sign
Model: TrOCR
Output: "Welcome to VVIT Campus"
🤖 Model Description
Architecture:
- Encoder: Vision Transformer (ViT)
- Decoder: Transformer-based language model
- Preprocessor: TrOCRProcessor (image normalization + text tokenization)

Advantages:
- End-to-end trainable
- Handles curved and distorted text
- Superior to CNN-RNN models in scene text recognition
📁 Dataset
Dataset Used: CTW1500
- 1500 natural scene images
- Polygon annotations for curved text
- Real-world images with diverse orientations and noise
📈 Performance Metrics
- Accuracy: 87.10%
- Validation Accuracy: 55.79%
- F1 Score: 18.45%
- CER and WER also evaluated
🔧 Tools & Libraries
- Python, PyTorch, HuggingFace Transformers
- NumPy, OpenCV, Flask, Matplotlib
🙌 Credits
Developed as a major B.Tech project by
Shaik Nikhat Anjum, Shaik Karimun, Sandipogu Chandrika, and Vanipenta Mounika
Under the guidance of Dr. P. Srinivasa Rao, VVIT.
