"""Exercise baked packages, CUDA extension and real audio format decoders."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio
import lhotse

out = Path(sys.argv[1])
site = Path('/opt/venv/lib/python3.12/site-packages')
for module in (torchaudio, lhotse):
    assert Path(module.__file__).is_relative_to(site), module.__file__
manifest = json.loads((out / 'package-manifest.json').read_text())
expected_lhotse = next(p['version'] for p in manifest['packages'] if p.get('name') == 'lhotse')
assert metadata.version('lhotse') == expected_lhotse
assert metadata.version('torchaudio') == '2.11.0+nv26.8.cuda13.3'
assert torch.__version__ == '2.13.0a0+8145d630e8.nv26.06'
assert torch.version.cuda == '13.3' and torch.backends.cudnn.version() == 92401
assert torch.ops._torchaudio.cuda_version() == 13030
assert torch.cuda.is_available()
signal = torch.sin(torch.arange(4800, dtype=torch.float32)[None] * 0.03)
cpu = torchaudio.transforms.Resample(48000, 24000)(signal)
gpu = torchaudio.transforms.Resample(48000, 24000).cuda()(signal.cuda())
torch.testing.assert_close(cpu, gpu.cpu(), atol=1e-5, rtol=1e-5)
a, b = torch.tensor([1.0, -0.25]), torch.tensor([0.75, 0.0])
cpu_filter = torchaudio.functional.lfilter(signal, a, b, clamp=False)
gpu_filter = torchaudio.functional.lfilter(signal.cuda(), a.cuda(), b.cuda(), clamp=False)
torch.testing.assert_close(cpu_filter, gpu_filter.cpu(), atol=1e-5, rtol=1e-5)
conv = torch.nn.Conv1d(1, 8, 3, padding=1).cuda()(gpu[:, None])
assert torch.isfinite(conv).all()
torch.cuda.synchronize()

fixtures = out / 'installed-wheel-tests/test/fixtures'
decoded = []
for name in ('mono_c0.wav', 'common_voice_en_651325.mp3', 'mono_c0.opus'):
    path = fixtures / name
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
    payload = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le', '-acodec', 'pcm_f32le', '-ac', '1', '-ar', '24000', 'pipe:1'])
    audio = np.frombuffer(payload, dtype='<f4')
    assert len(audio) > 0 and np.isfinite(audio).all()
    decoded.append({'file': name, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'codec': info['streams'][0]['codec_name'], 'decoded_samples_24k': len(audio)})
wav = out / 'roundtrip.wav'
flac = out / 'roundtrip.flac'
sf.write(wav, signal[0].numpy(), 48000, subtype='PCM_16')
subprocess.run(['ffmpeg', '-v', 'error', '-i', str(wav), str(flac)], check=True)
wave_audio, wave_rate = sf.read(wav, dtype='float32')
flac_audio, flac_rate = sf.read(flac, dtype='float32')
assert wave_rate == flac_rate == 48000 and np.array_equal(wave_audio, flac_audio)
resampled = soxr.resample(flac_audio, 48000, 24000)
assert len(resampled) == 2400 and np.isfinite(resampled).all()
report = {'ok': True, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
          'torchaudio': metadata.version('torchaudio'), 'lhotse': metadata.version('lhotse'),
          'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
          'baked_module_paths': {'lhotse': lhotse.__file__, 'torchaudio': torchaudio.__file__},
          'cpu_gpu_resampling_and_compiled_filter': True, 'ffmpeg_decodes': decoded,
          'wav_flac_lossless_roundtrip': True, 'soxr_resample': True}
(out / 'runtime-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
