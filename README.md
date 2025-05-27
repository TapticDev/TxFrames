# TxFrames - AI Frame Interpolation Tool

TxFrames is a powerful video frame interpolation tool that uses AI to generate smooth slow-motion effects or increase frame rates while maintaining high visual quality.

## Features

- 🚀 **AI-Powered Interpolation**: Uses advanced neural networks to generate intermediate frames
- ⚡ **GPU Acceleration**: Supports DirectML for hardware-accelerated processing
- 🎞️ **Multiple Modes**:
  - Frame multiplication (2x, 3x, 4x, 8x, 16x)
  - Slow motion (2x, 4x, 8x)
- 🎥 **Video Support**: Works with MP4, MKV, AVI, MOV, WEBM, FLV, GIF
- 🔊 **Audio Handling**: Preserves original audio (except in slow motion mode)
- 🛠️ **Customizable Settings**: Adjustable quality, GPU selection, and output options

## Installation

### Windows (Coming soon!)
1. Download the latest release from [Releases Page](https://github.com/TapticDev/TxFrames/releases)
2. Extract the ZIP file
3. Run `TxFrames.exe`

### Requirements
- Windows 10/11 (64-bit)
- Python 3.10+ (for source version)
- FFmpeg (included in package)
- ONNX Runtime with DirectML support

## Usage

1. **Select Input Files**: Drag and drop or click to select video files
2. **Choose Settings**:
   - AI Model: Select between quality vs speed
   - Interpolation: Choose frame multiplication or slow motion
   - Output: Set destination and codec options
3. **Start Processing**: Click "START PROCESSING" button
4. **Get Results**: Find your processed videos in the output folder

## Configuration

Settings are saved automatically in:
`C:\Users\<username>\Documents\TxFrames_config.json`

Key options:
- `ai_model`: "txfsmall" (faster) or "txflarge" (better quality)
- `interpolation`: "2x", "4x", "slow2x", etc.
- `gpu`: "Auto" or specific GPU selection
- `video_codec`: Output codec (H.264, H.265, etc.)
- `keep_frames`: Keep temporary frame images (true/false)

## Building from Source

# Clone the repository
git clone https://github.com/TapticDev/TxFrames.git
cd TxFrames

# Install dependencies
pip install -r requirements.txt

# Run the application
python TxFrames.py
