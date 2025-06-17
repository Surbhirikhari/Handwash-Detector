import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2

# Simple CNN model
class HandWashingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Dataset class
class HandWashingDataset(Dataset):
    def __init__(self, washing_dir, not_washing_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load washing hands images
        for img_name in os.listdir(washing_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(washing_dir, img_name))
                self.labels.append(1)
        
        # Load not washing hands images
        for img_name in os.listdir(not_washing_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(not_washing_dir, img_name))
                self.labels.append(0)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def train_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load data
    dataset = HandWashingDataset('WASHING_HANDS', 'NOT_WASHING_HANDS', transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = HandWashingCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("Training model...")
    model.train()
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(10):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/10, Loss: {epoch_loss:.4f}")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict()
            print(f"New best model saved with loss: {best_loss:.4f}")
    
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_hand_washing_model.pth')
        print(f"Best model saved with loss: {best_loss:.4f}")
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return model, device, transform

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model = HandWashingCNN().to(device)
    if os.path.exists('best_hand_washing_model.pth'):
        model.load_state_dict(torch.load('best_hand_washing_model.pth'))
        print("Loaded best saved model")
    else:
        print("No saved model found, training new model...")
        model, device, transform = train_model()
    
    return model, device, transform

def process_video(video_path, model, device, transform, skip_frames=1, delay=30):
    cap = cv2.VideoCapture(video_path)
    model.eval()
    frame_count = 0
    
    print(f"\nProcessing video: {video_path}")
    print("Press 'q' to quit")
    
    # For tracking predictions
    total_frames = 0
    washing_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            cv2.imshow('Hand Washing Detection', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            continue
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor).squeeze()
            confidence = output.item()
            # Using threshold of 0.6 for washing hands detection
            prediction = confidence > 0.6
        
        # Update counters
        total_frames += 1
        if prediction:
            washing_frames += 1
        
        # Add prediction to frame
        result = "Washing Hands" if prediction else "Not Washing Hands"
        color = (0, 255, 0) if prediction else (0, 0, 255)
        text = f"{result}: {confidence:.2%}"
        
        # Add text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 35), font, 1, color, 2)
        
        cv2.imshow('Hand Washing Detection', frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Print video summary
    washing_percentage = (washing_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"Video Summary:")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with hand washing: {washing_frames}")
    print(f"Percentage of frames with hand washing: {washing_percentage:.2f}%")
    
    cap.release()
    cv2.destroyAllWindows()

def test_single_image(image_path, model, device, transform):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor).squeeze()
        confidence = output.item()
        # Using threshold of 0.6 for washing hands detection
        prediction = confidence > 0.6
    
    # Print results
    result = "Washing Hands" if prediction else "Not Washing Hands"
    print(f"\nPrediction for {image_path}:")
    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Threshold used: 0.6")

if __name__ == "__main__":
    # Load or train model
    model, device, transform = load_model()
    
    # Test images
    print("\nTesting images:")
    test_single_image("test images/test.jpeg", model, device, transform)
    test_single_image("test images/test1.jpeg", model, device, transform)
    
    # Process videos with slower frame rate
    print("\nProcessing videos:")
    process_video("test video/Testvideo1.mp4", model, device, transform, skip_frames=1, delay=30)
