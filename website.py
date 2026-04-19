import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from huggingface_hub import login
import cv2
import tempfile
import os


# 1. Download the weights file from the Hugging Face Hub
# NOTE: Replace "best.pt" if the repository uses a different filename for the weights 
# (e.g., "model.pt" or "yolo26s.pt").
model_path = hf_hub_download(repo_id="Perception365/VehicleNet-Y26s", filename="weights/best.pt")

# 2. Initialize YOLO with the downloaded local file path
model = YOLO(model_path)

# Optional: Standalone test inference
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)

def predict_image(img, conf_threshold, iou_threshold, filter_text):
    if img is None:
        return None
    
    predict_args = {
        "source": img,
        "conf": conf_threshold,
        "iou": iou_threshold,
        "verbose": False
    }

    text = str(filter_text).strip() if filter_text else ""

    if text:
        search_terms = [term.strip().lower() for term in text.split(",") if term.strip()]

        if search_terms:
            target_ids = [
                cls_id for cls_id, cls_name in model.names.items()
                if any(term in cls_name.lower() for term in search_terms)
            ]

            predict_args["classes"] = target_ids if target_ids else [-1]

    results = model.predict(**predict_args)
    
    # if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
    if not results or len(results) == 0 or len(results[0].boxes) == 0:
        return img
        
    # plot() generates a BGR numpy array
    annotated_bgr = results[0].plot(labels=True, conf=True)
    
    # Convert BGR to RGB so Gradio displays the colors correctly
    annotated_rgb = annotated_bgr[..., ::-1] 
    
    return annotated_rgb


def predict_video(video_path, conf_threshold, iou_threshold, filter_text):
    if not video_path:
        return None

    # Setup the exact same filter logic as the image function
    predict_args = {
        "conf": conf_threshold,
        "iou": iou_threshold,
        "verbose": False
    }

    text = str(filter_text).strip() if filter_text else ""
    if text:
        search_terms = [term.strip().lower() for term in text.split(",") if term.strip()]
        if search_terms:
            target_ids = [
                cls_id for cls_id, cls_name in model.names.items()
                if any(term in cls_name.lower() for term in search_terms)
            ]
            predict_args["classes"] = target_ids if target_ids else [-1]

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a temporary file to save the processed video
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "output.webm")
    
    # Setup the video writer (mp4v codec works well for Gradio)
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Loop through the video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video
            
        # Pass the single frame to YOLO
        predict_args["source"] = frame
        results = model.predict(**predict_args)
        
        # Annotate or keep original
        if not results or len(results) == 0 or len(results[0].boxes) == 0:
            out.write(frame)
        else:
            annotated_frame = results[0].plot(labels=True, conf=True)
            out.write(annotated_frame) # Write the BGR frame directly to video

    # Clean up
    cap.release()
    out.release()
    return output_path


with gr.Blocks(title="Ultralytics YOLO26 App") as app:
    gr.Markdown("# 🚗 VehicleNet Object Detection")
    gr.Markdown("Upload an image or a video to detect vehicles. Use the filters to isolate specific classes.")
    
    # Global settings shared across both tabs
    with gr.Row():
        conf_slider = gr.Slider(0, 1, value=0.25, label="Confidence threshold")
        iou_slider = gr.Slider(0, 1, value=0.45, label="IoU threshold")
        filter_box = gr.Textbox(label="Filter by label, seperate with commas ('Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck', 'Three-wheeler', 'Two-wheeler', 'LCV', 'Mini-bus', 'Tempo-traveller', 'Bicycle', 'Van', 'Others')", placeholder="Leave empty to detect all classes")

    # The Tabs interface
    with gr.Tabs():
        # --- TAB 1: IMAGE ---
        with gr.TabItem("🖼️ Image Detection"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image")
                img_output = gr.Image(type="numpy", label="Result")
            img_button = gr.Button("Process Image", variant="primary")
            
            # Wire up the button
            img_button.click(
                fn=predict_image,
                inputs=[img_input, conf_slider, iou_slider, filter_box],
                outputs=img_output
            )

        # --- TAB 2: VIDEO ---
        with gr.TabItem("🎥 Video Detection"):
            with gr.Row():
                vid_input = gr.Video(label="Upload Video")
                vid_output = gr.Video(label="Result Video")
            vid_button = gr.Button("Process Video", variant="primary")
            
            # Wire up the button
            vid_button.click(
                fn=predict_video,
                inputs=[vid_input, conf_slider, iou_slider, filter_box],
                outputs=vid_output
            )


# Print labels to console for debugging
print(model.names)

# Launch the app
app.launch()