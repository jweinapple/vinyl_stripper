# Optimized Spleeter Real-Time Processing

## Overview

This is an optimized implementation of Spleeter for real-time vocal removal, achieving **0.5x real-time performance** (processing is 2x faster than audio playback).

## Performance

- **Real-time ratio**: 0.5x (processes audio in half the time it takes to play)
- **Processing speed**: ~1.5s per 3.0s chunk
- **Speedup**: 2x faster than real-time
- **Platform**: Optimized for Mac (Apple Silicon), portable to Raspberry Pi 5

## Quick Start

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (if not already installed)
pip install spleeter tensorflow==2.13.0 tensorflow-estimator sounddevice numpy
```

### Run the Optimized Script

```bash
# Basic usage (auto-detects UA input and Mac speakers)
python3 test_spleeter_new.py

# The script will:
# 1. Auto-detect Universal Audio device as input
# 2. Auto-detect Mac speakers as output
# 3. Build an 8-second buffer
# 4. Switch to processed audio (vocals removed)
```

## How It Works

### Processing Pipeline

1. **Audio Input**: Captures audio from Universal Audio Thunderbolt interface
2. **Chunking**: Processes audio in 3.0-second chunks with 100ms overlap
3. **Stem Separation**: Uses Spleeter 2-stems model (vocals/accompaniment)
4. **Vocal Removal**: Extracts accompaniment (removes vocals)
5. **Crossfading**: Smooth transitions between chunks
6. **Audio Output**: Plays processed audio through Mac speakers

### Optimizations Applied

1. **Large Chunk Size** (3.0s): Reduces per-chunk overhead
2. **TensorFlow Optimizations**: oneDNN, multi-threading, optimized session config
3. **Model Pre-loading**: Eliminates first-call overhead
4. **Memory Optimization**: Contiguous arrays, efficient access patterns
5. **Buffer Management**: 8s minimum buffer with 2s fallback threshold
6. **Crossfading**: 100ms overlap prevents clicks/pops

## Configuration

### Chunk Size

The script uses 3.0-second chunks for optimal performance. This balances:
- Processing efficiency (larger = less overhead)
- Latency (smaller = lower latency)
- Buffer requirements (larger = more buffer needed)

### Buffer Settings

- **Minimum buffer**: 8 seconds (before switching to processed audio)
- **Low threshold**: 2 seconds (falls back to passthrough if below)
- **Pre-buffer**: Builds buffer during passthrough mode

## Performance Monitoring

The script logs performance metrics every 5 seconds:

```
[PROCESSED] Buffer: 7.5s | Queue: 0 | Processed: 10 chunks | Avg: 1.50s/chunk
```

- **Buffer**: Current output buffer size in seconds
- **Queue**: Number of chunks waiting to be processed
- **Processed**: Total chunks processed
- **Avg**: Average processing time per chunk

## Troubleshooting

### Audio Dropouts

If you experience dropouts:
1. Check buffer levels in logs (should stay above 3s)
2. Increase `min_buffer_samples` if needed
3. Reduce chunk size if processing can't keep up

### Processing Too Slow

If processing is slower than real-time:
1. Check CPU usage (should use all cores)
2. Verify TensorFlow optimizations are enabled
3. Consider reducing chunk size (but may increase overhead)

### Device Not Found

If devices aren't detected:
1. Run with `--list-devices` to see available devices
2. Check device permissions (macOS may need microphone access)
3. Verify devices are connected and powered on

## Benchmarking

Use the benchmark script to test performance:

```bash
python3 benchmark_spleeter.py
```

This will:
- Generate test audio
- Process multiple chunks
- Report average processing time and real-time ratio
- Compare against target (0.5x real-time)

## Platform Differences

### Mac (Current Implementation)
- Chunk size: 3.0s
- Real-time ratio: ~0.5x
- Buffer: 8s minimum

### Raspberry Pi 5 (Expected)
- Chunk size: 2.0-2.5s (may need adjustment)
- Real-time ratio: ~2-4x (slower than Mac)
- Buffer: 15-30s minimum (larger buffer needed)

## Technical Details

### Model
- **Spleeter**: 2-stems model (vocals/accompaniment)
- **Format**: TensorFlow Estimator API
- **Sample Rate**: 44.1 kHz
- **Channels**: Stereo (2 channels)

### TensorFlow Configuration
- **Version**: 2.13.0 (compatible with Spleeter)
- **Threading**: All CPU cores
- **Optimizations**: oneDNN enabled
- **Session**: Optimized for inference

### Audio Processing
- **Input**: Universal Audio Thunderbolt (device 5)
- **Output**: MacBook Pro Speakers (device 2)
- **Format**: Float32, 44.1kHz, Stereo
- **Block Size**: 2048 samples

## Future Improvements

- [ ] Port to Raspberry Pi 5 with platform-specific optimizations
- [ ] Add Hailo AI Kit support for hardware acceleration
- [ ] Implement batch processing for even better throughput
- [ ] Add support for 4-stems model (vocals, drums, bass, other)
- [ ] Real-time monitoring dashboard

## See Also

- `GIT_WORKFLOW.md`: Git workflow for Mac/Pi development
- `vinyl_stripper_pi.py`: Pi-optimized version (different implementation)
- `benchmark_spleeter.py`: Performance benchmarking tool

