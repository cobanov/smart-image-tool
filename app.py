from utils.midas_utils import depth
# from utils.blip_utils import inference
from utils.png_utils import chunk
from utils.video2img_utils import extract
from utils import utils
import gradio as gr

# MiDaS Depth Map
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


# BLIP Image Captioning
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


## PNG Chunk
png_inputs = [
    gr.Image(type="pil"),
]
png_outputs = gr.outputs.Textbox(label="Output")


## Video2PNG
inputs = [
    gr.File(label="Please select video"),
    gr.Number(
        value=2,
    ),
    gr.Textbox(label="Output Directory"),
]
outputs = gr.Text(label="Output Directory")




with gr.Blocks() as demo:
    with gr.Tab("MiDaS"):
        gr.Interface(
            depth,
            midas_inputs,
            midas_outputs,
        )
    with gr.Tab("Caption"):
        gr.Interface(
            utils.null_function,
            blip_inputs,
            blip_outputs,
        )
    with gr.Tab("PNG Analyze"):
        gr.Interface(
            chunk,
            png_inputs,
            png_outputs,
        )
    with gr.Tab("Video2PNG"):
        gr.Interface(extract, inputs, outputs)

if __name__ == "__main__":
    demo.launch()
