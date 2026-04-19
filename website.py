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

iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Textbox(label="Filter by label('Hatchback', 'Sedan', 'SUV', 'MUV', 'Bus', 'Truck', 'Three-wheeler', 'Two-wheeler', 'LCV', 'Mini-bus', 'Tempo-traveller', 'Bicycle', 'Van', 'Others')", placeholder="Leave empty to detect all classes")
    ],
    outputs=gr.Image(type="numpy", label="Result"),
    title="Ultralytics Gradio YOLO26",
    description="Upload images for YOLO26 object detection.",
)

iface.launch()