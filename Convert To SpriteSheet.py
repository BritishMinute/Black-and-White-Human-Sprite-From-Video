import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import os
import concurrent.futures

# Load the DeepLabV3 model with the updated API
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights).to(device).eval()

# Transform for the input images
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_frame(frame_path, output_path, resize_factor=0.5):
    try:
        # Load the image and resize
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))

        # Apply the transformation and add a batch dimension
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Perform the segmentation
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)

        # Create a mask for the person class (class 15 in COCO dataset used by DeepLabV3)
        person_mask = output_predictions == 15

        # Convert mask to numpy array
        person_mask_np = person_mask.byte().cpu().numpy()

        # Create a new image with a transparent background
        result = Image.new("RGBA", img.size, (0, 0, 0, 0))

        # Create a white outline
        outline = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(outline)
        contours, _ = cv2.findContours(person_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = contour[:, 0, :]  # Reshape contour to correct format
            for i in range(len(contour) - 1):
                draw.line([tuple(contour[i]), tuple(contour[i + 1])], fill=255, width=3)  # Adjust the width for outline thickness

        # Fill the person area with black
        person_img = Image.fromarray(person_mask_np.astype("uint8") * 255, mode='L')
        result.paste((0, 0, 0, 255), mask=person_img)

        # Save the processed frame
        result.save(output_path)
    except Exception as e:
        print(f"Error processing {frame_path}: {e}")

def extract_frames(video_path, output_folder, interval=13, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frames_to_save = min(max_frames, total_frames // interval)
    print(f"Extracting up to {frames_to_save} frames at interval {interval}.")
    
    for i in tqdm(range(frames_to_save), desc="Extracting frames"):
        frame_no = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, frame)
    cap.release()

def process_video(video_path, output_folder):
    frame_folder = os.path.join(output_folder, "frames")
    processed_folder = os.path.join(output_folder, "processed")
    
    # Step 1: Extract frames
    extract_frames(video_path, frame_folder)

    # Step 2: Process each frame
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    frame_files = sorted(os.listdir(frame_folder))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda f: process_frame(os.path.join(frame_folder, f), os.path.join(processed_folder, f"processed_{f}")), frame_files), total=len(frame_files), desc="Processing frames"))

    print(f"All frames processed and saved to {processed_folder}")

# Run the processing
video_path = input("Enter the path to the MP4 file: ")
output_folder = os.path.join(os.path.dirname(video_path), "Output")
process_video(video_path, output_folder)
