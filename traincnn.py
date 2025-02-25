from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import xml.etree.ElementTree as ET
import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt

# Set the device (CUDA if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Dataset Class for CTW1500
class CTW1500_Dataset(Dataset):
    def __init__(self, img_dir, label_dir, processor, max_target_length=128, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform
        self.image_data = []
        self._load_image_data()
    
    def _load_image_data(self):
        for img_file in sorted(os.listdir(self.img_dir)):
            img_path = os.path.join(self.img_dir, img_file)
            img_file_name = os.path.splitext(img_file)[0]
            
            # Parse the corresponding XML file
            xml_file = img_file_name + '.xml'
            xml_path = os.path.join(self.label_dir, xml_file)
            if not os.path.exists(xml_path):
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for box in root.findall(".//box"):
                label = box.find("label").text
                if label == '' or label == '###':
                    continue
                
                left = int(box.attrib["left"])
                top = int(box.attrib["top"])
                width = int(box.attrib["width"])
                height = int(box.attrib["height"])
                
                bbox = [left, top, left + width, top + height]
                
                self.image_data.append({
                    "img_path": img_path,
                    "bbox": bbox,
                    "label": label
                })
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        
        # Get image and crop it according to the bbox
        img_path = data["img_path"]
        bbox = data["bbox"]
        cropped_img = Image.open(img_path).convert('RGB').crop(bbox)
        
        # Apply any transformations (e.g., resizing, normalization)
        if self.transform:
            cropped_img = self.transform(cropped_img)
        
        # Use the processor to get the pixel values in tensor form
        pixel_values = self.processor(cropped_img, return_tensors="pt", do_rescale=False).pixel_values
        
        # Tokenize the label text
        label = data["label"]
        labels = self.processor.tokenizer(label,
                                         padding="max_length",
                                         max_length=self.max_target_length,
                                         truncation=True).input_ids
        
        # Ignore padding tokens during training (set to -100)
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        
        return encoding

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to fixed dimensions
    transforms.ToTensor(),          # Convert to tensor
])

# Initialize processor and dataset
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
dset = CTW1500_Dataset(img_dir=r'data/train_images/train_images',
                       label_dir=r'data/train_labels/ctw1500_train_labels',
                       processor=processor,
                       transform=transform)

print(f"There are {len(dset)} samples in this dataset")

# Split dataset into training and validation sets
train_size = round(len(dset) * 0.8)
val_size = len(dset) - train_size

train_set, val_set = random_split(dset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# Load the pre-trained TrOCR model for fine-tuning
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

# Configure model
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss(ignore_index=-100)

# Train the model with validation
def train_model_with_validation(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=3):
    best_val_loss = float("inf")
    training_loss_history = []
    validation_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_loss_history.append(avg_train_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("best_trocr_model")
            print("Model saved with improved validation loss.")

    return training_loss_history, validation_loss_history

# Plot training and validation loss
def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

# Main training loop
if __name__ == '__main__':
    num_epochs = 10

    # Train the model
    train_loss, val_loss = train_model_with_validation(
        model, train_loader, val_loader, optimizer, criterion, device, num_epochs
    )

    # Plot the loss
    plot_loss(train_loss, val_loss)

    # Save the final model
    model.save_pretrained("final_trocr_model")