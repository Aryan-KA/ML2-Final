import gradio as gr

def reverse_text(input_text):
    return input_text[::-1]

iface = gr.Interface(
    fn=reverse_text,
    inputs="text",
    outputs="text",
    title="Text Reverser",
    description="Enter text, and this app will reverse it.",
)

iface.launch()
