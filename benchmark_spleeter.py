#!/usr/bin/env python3
"""
Quick benchmark script to test Spleeter performance improvements
"""

import sys
import os
import time
import numpy as np
import logging

# Suppress most logging
logging.basicConfig(level=logging.WARNING)

# Import the processor
sys.path.insert(0, os.path.dirname(__file__))
from test_spleeter_new import SpleeterProcessor, find_universal_audio_device, find_mac_speakers

def benchmark():
    """Run a quick benchmark test."""
    print("=" * 60)
    print("SPLEETER PERFORMANCE BENCHMARK")
    print("=" * 60)
    print()
    
    # Find devices
    input_device = find_universal_audio_device()
    output_device = find_mac_speakers()
    
    if input_device is None or output_device is None:
        print("Error: Could not find audio devices")
        return
    
    print(f"Input:  Device {input_device}")
    print(f"Output: Device {output_device}")
    print()
    
    # Create processor
    processor = SpleeterProcessor(
        input_device=input_device,
        output_device=output_device,
        remove_vocals=True
    )
    
    print(f"Configuration:")
    print(f"  - Chunk size: {processor.chunk_duration}s")
    print(f"  - Sample rate: {processor.sample_rate} Hz")
    print(f"  - Overlap: {processor.overlap_samples / processor.sample_rate * 1000:.0f}ms")
    print()
    
    # Generate test audio (sine wave)
    print("Generating test audio...")
    duration = 3.0  # 3 seconds of test audio
    samples = int(duration * processor.sample_rate)
    t = np.linspace(0, duration, samples)
    # Stereo sine wave
    test_audio = np.zeros((samples, 2), dtype=np.float32)
    test_audio[:, 0] = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
    test_audio[:, 1] = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Process multiple chunks
    print("Processing test chunks...")
    print()
    
    num_chunks = 5
    processing_times = []
    
    for i in range(num_chunks):
        chunk_start = i * processor.hop_samples
        chunk_end = min(chunk_start + processor.chunk_samples, samples)
        chunk = test_audio[chunk_start:chunk_end].copy()
        
        # Pad to exact chunk size if needed
        if chunk.shape[0] < processor.chunk_samples:
            padding = np.zeros((processor.chunk_samples - chunk.shape[0], 2), dtype=np.float32)
            chunk = np.vstack([chunk, padding])
        
        start_time = time.time()
        result = processor.process_chunk(chunk)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        realtime_ratio = processing_time / processor.chunk_duration
        print(f"Chunk {i+1}: {processing_time:.3f}s ({realtime_ratio:.2f}x real-time)")
    
    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print()
    
    avg_time = np.mean(processing_times)
    min_time = np.min(processing_times)
    max_time = np.max(processing_times)
    std_time = np.std(processing_times)
    
    print(f"Processing Times:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min:     {min_time:.3f}s")
    print(f"  Max:     {max_time:.3f}s")
    print(f"  Std Dev: {std_time:.3f}s")
    print()
    
    realtime_ratio = avg_time / processor.chunk_duration
    speedup = 1.0 / realtime_ratio if realtime_ratio > 0 else 0
    
    print(f"Real-time Performance:")
    print(f"  Real-time ratio: {realtime_ratio:.3f}x")
    print(f"  Speedup: {speedup:.2f}x faster than real-time")
    print()
    
    if realtime_ratio <= 0.5:
        print("✓ TARGET ACHIEVED! Processing is 2x+ faster than real-time")
    elif realtime_ratio <= 0.7:
        print("✓ EXCELLENT! Processing is significantly faster than real-time")
    elif realtime_ratio < 1.0:
        print("✓ GOOD! Processing is faster than real-time")
    else:
        print("⚠ Processing is slower than real-time")
    
    print()
    print(f"Target: 0.5x real-time")
    print(f"Actual: {realtime_ratio:.3f}x real-time")
    print(f"Status: {'✓ ACHIEVED' if realtime_ratio <= 0.5 else '⚠ Not yet achieved'}")
    print()

if __name__ == "__main__":
    try:
        benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

