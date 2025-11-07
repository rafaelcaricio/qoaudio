# QOA Audio Examples

This directory contains examples demonstrating how to use the qoaudio library.

## encode.rs

Encodes raw PCM audio data to QOA format.

### Usage

```bash
cargo run --example encode -- <input_raw_pcm> <output_qoa> <channels> <sample_rate> <samples>
```

### Arguments

- `input_raw_pcm` - Path to raw PCM audio file (16-bit signed little-endian)
- `output_qoa` - Path to output QOA file
- `channels` - Number of channels (1 for mono, 2 for stereo, up to 8)
- `sample_rate` - Sample rate in Hz (e.g., 44100)
- `samples` - Number of samples per channel

### Example

Encode a stereo file with 44.1kHz sample rate:

```bash
cargo run --example encode -- input.raw output.qoa 2 44100 88200
```

### Creating Raw PCM Files

You can create raw PCM files from various audio formats using tools like ffmpeg:

```bash
# Convert MP3 to raw PCM (stereo, 44.1kHz)
ffmpeg -i input.mp3 -f s16le -acodec pcm_s16le output.raw

# Convert WAV to raw PCM
ffmpeg -i input.wav -f s16le -acodec pcm_s16le output.raw
```

To get the audio parameters:
```bash
ffprobe input.wav
```

### Testing the Example

Create a simple test file with Python:

```python
import struct
import math

sample_rate = 44100
duration = 1.0  # 1 second
frequency = 440  # A4 note
samples_per_channel = int(sample_rate * duration)
channels = 2

with open('test_audio.raw', 'wb') as f:
    for i in range(samples_per_channel):
        t = i / sample_rate
        # Left channel: sine wave
        left = int(math.sin(2 * math.pi * frequency * t) * 16384)
        # Right channel: cosine wave
        right = int(math.cos(2 * math.pi * frequency * t) * 16384)

        # Write as 16-bit little-endian integers
        f.write(struct.pack('<h', left))
        f.write(struct.pack('<h', right))
```

Then encode it:
```bash
cargo run --example encode -- test_audio.raw test_audio.qoa 2 44100 44100
```

And play it back:
```bash
cargo run --example play --features rodio -- test_audio.qoa
```

## play.rs

Plays QOA audio files using the rodio library.

### Usage

```bash
cargo run --example play --features rodio -- <path_to_qoa_file>
```

### Example

```bash
cargo run --example play --features rodio -- audio.qoa
```

Note: The play example requires the `rodio` feature to be enabled.
