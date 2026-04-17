import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from huggingface_hub import login


# 1. Download the weights file from the Hugging Face Hub
# NOTE: Replace "best.pt" if the repository uses a different filename for the weights 
# (e.g., "model.pt" or "yolo26s.pt").
model_path = hf_hub_download(repo_id="Perception365/VehicleNet-Y26s", filename="weights/best.pt")

# 2. Initialize YOLO with the downloaded local file path
model = YOLO(model_path)

# Optional: Standalone test inference
source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)

def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold
    )
    
    if not results:
        return None
        
    # plot() generates a BGR numpy array
    annotated_bgr = results[0].plot(labels=True, conf=True)
    
    # Convert BGR to RGB so Gradio displays the colors correctly
    annotated_rgb = annotated_bgr[..., ::-1] 
    
    return annotated_rgb

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="numpy", label="Result"),
    title="Ultralytics Gradio YOLO26",
    description="Upload images for YOLO26 object detection.",
)

iface.launch()
