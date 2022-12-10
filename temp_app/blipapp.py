import gradio as gr
from utils.blip_utils import inference

blip_inputs = [
    gr.inputs.Image(type="pil"),
    gr.inputs.Radio(
        choices=["Beam search", "Nucleus sampling"],
        type="value",
        default="Nucleus sampling",
        label="Caption Decoding Strategy",
    ),
]
blip_outputs = gr.outputs.Textbox(label="Output")


demo = gr.Interface(
    inference,
    blip_inputs,
    blip_outputs,
)

if __name__ == "__main__":
    demo.launch()
