from models.audiosep import AudioSep
from utils import get_ss_model
import torch
from pipeline import separate_audio

device = torch.device('cpu')

ss_model = get_ss_model('config/audiosep_base.yaml')

model = AudioSep.from_pretrained("nielsr/audiosep-demo", ss_model=ss_model)

audio_file = 'audios/test.wav'
text = 'synthesizer sound'
output_file = 'separated_audio.wav'

# AudioSep processes the audio at 32 kHz sampling rate
separate_audio(model, audio_file, text, output_file, device)
