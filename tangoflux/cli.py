import click
import torchaudio
from tangoflux import TangoFluxInference

@click.command()
@click.argument('prompt')
@click.argument('output_file')
@click.option('--duration', default=10, type=int, help='Duration in seconds (1-30)')
@click.option('--steps', default=50, type=int, help='Number of inference steps (10-100)')
def main(prompt: str, output_file: str, duration: int, steps: int):
    """Generate audio from text using TangoFlux.
    
    Args:
        prompt: Text description of the audio to generate
        output_file: Path to save the generated audio file
        duration: Duration of generated audio in seconds (default: 10)
        steps: Number of inference steps (default: 50)
    """
    if not 1 <= duration <= 30:
        raise click.BadParameter('Duration must be between 1 and 30 seconds')
    if not 10 <= steps <= 100:
        raise click.BadParameter('Steps must be between 10 and 100')

    model = TangoFluxInference(name="declare-lab/TangoFlux")
    audio = model.generate(prompt, steps=steps, duration=duration)
    torchaudio.save(output_file, audio, sample_rate=44100)

if __name__ == '__main__':
    main()
