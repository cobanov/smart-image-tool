from utils.midas_utils import depth
from utils.blip_utils import inference
from utils.png_utils import chunk
import gradio as gr


midas_inputs = [
    gr.Image(type="pil", label="Original Image"),
    gr.Dropdown(
        [
            "DPTDepthModel",
            "DPT_Hybrid",
            "DPT_Large",
            "MiDaS",
            "MiDaS_small",
            "MidasNet",
            "MidasNet_small",
        ],
        value="MiDaS",
    ),
]
midas_outputs = [
    gr.Image(
        type="pil",
        interactive=False,
    )
]

blip_inputs = [
    gr.Image(type="pil"),
    gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        type="value",
        default="Nucleus sampling",
        label="Caption Decoding Strategy",
    ),
]
blip_outputs = gr.outputs.Textbox(label="Output")

png_inputs = [
    gr.Image(type="pil"),
]
png_outputs = gr.outputs.Textbox(label="Output")


with gr.Blocks() as demo:
    with gr.Tab("MiDaS"):
        gr.Interface(
            depth,
            midas_inputs,
            midas_outputs,
        )
    with gr.Tab("Caption"):
        gr.Interface(
    inference,
    blip_inputs,
    blip_outputs,
)
    with gr.Tab("PNG Analyze"):
        gr.Interface(
    inference,
    png_inputs,
    png_outputs,
)
if __name__ == "__main__":
    demo.launch()
