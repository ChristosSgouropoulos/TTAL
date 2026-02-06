import gradio as gr
import torchaudio
import click
import tempfile
from tangoflux import TangoFluxInference

model = TangoFluxInference(name="declare-lab/TangoFlux")


def generate_audio(prompt, duration, steps):
    audio = model.generate(prompt, steps=steps, duration=duration)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, audio, sample_rate=44100)
        return f.name


examples = [
    ["Hammer slowly hitting the wooden table", 10, 50],
    ["Gentle rain falling on a tin roof", 15, 50],
    ["Wind chimes tinkling in a light breeze", 10, 50],
    ["Rhythmic wooden table tapping overlaid with steady water pouring sound", 10, 50],
]

with gr.Blocks(title="TangoFlux Text-to-Audio Generation") as demo:
    gr.Markdown("# TangoFlux Text-to-Audio Generation")
    gr.Markdown("Generate audio from text descriptions using TangoFlux")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Text Prompt", placeholder="Enter your audio description..."
            )
            duration = gr.Slider(
                minimum=1, maximum=30, value=10, step=1, label="Duration (seconds)"
            )
            steps = gr.Slider(
                minimum=10, maximum=100, value=50, step=10, label="Number of Steps"
            )
            generate_btn = gr.Button("Generate Audio")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")

    generate_btn.click(
        fn=generate_audio, inputs=[prompt, duration, steps], outputs=audio_output
    )

    gr.Examples(
        examples=examples,
        inputs=[prompt, duration, steps],
        outputs=audio_output,
        fn=generate_audio,
    )

@click.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=None, help='Port to bind to')
@click.option('--share', is_flag=True, help='Enable sharing via Gradio')
def main(host, port, share):
    demo.queue().launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    main()
