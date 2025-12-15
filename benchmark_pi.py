#!/usr/bin/env python3
"""
Benchmark script to test stem separation speed on different backends.
Run this on your Pi 5 to determine which backend works best.

Usage:
    python3 benchmark_pi.py
    python3 benchmark_pi.py --duration 5  # Test with 5 seconds of audio
"""

import argparse
import sys
import os
import time
import numpy as np

# Platform info
IS_PI = os.path.exists('/proc/device-tree/model')
PI_MODEL = ""
if IS_PI:
    try:
        with open('/proc/device-tree/model') as f:
            PI_MODEL = f.read().strip()
    except:
        pass

def get_memory_info():
    """Get current memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent': mem.percent
        }
    except:
        return None

def benchmark_backend(backend_name: str, audio: np.ndarray, sample_rate: int, num_runs: int = 3):
    """Benchmark a single backend."""
    print(f"\n{'='*50}")
    print(f"Testing: {backend_name}")
    print(f"{'='*50}")

    try:
        # Import backend
        if backend_name == "demucs":
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"

            print(f"Device: {device}")
            print("Loading model...")

            model = get_model("htdemucs")
            model.to(device)
            model.eval()

            # Quantize for CPU
            if device == "cpu":
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("Model quantized")
                except:
                    pass

            def separate(audio_in):
                audio_tensor = torch.from_numpy(audio_in.T).unsqueeze(0).float().to(device)
                ref_std = audio_tensor.std()
                audio_tensor = audio_tensor / (ref_std + 1e-8)

                with torch.no_grad():
                    sources = apply_model(model, audio_tensor, device=device,
                                        segment=None, split=False, shifts=1, overlap=0.0)

                return sources[0].cpu().numpy()

        elif backend_name == "spleeter":
            from spleeter.separator import Separator

            print("Loading model...")
            separator = Separator('spleeter:4stems')
            separator._get_prediction_generator()  # Pre-load

            def separate(audio_in):
                return separator.separate(audio_in)

        elif backend_name == "onnx":
            import onnxruntime as ort

            print(f"ONNX providers: {ort.get_available_providers()}")
            print("Note: ONNX requires pre-converted model")

            # This is a placeholder - actual ONNX model needed
            def separate(audio_in):
                return {"dummy": audio_in}

            print("ONNX model not available - skipping")
            return None

        else:
            print(f"Unknown backend: {backend_name}")
            return None

    except ImportError as e:
        print(f"Backend not available: {e}")
        return None
    except Exception as e:
        print(f"Error loading backend: {e}")
        return None

    # Warmup run
    print("Warming up...")
    try:
        _ = separate(audio)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None

    # Benchmark runs
    print(f"Running {num_runs} benchmark iterations...")
    times = []
    mem_before = get_memory_info()

    for i in range(num_runs):
        start = time.time()
        _ = separate(audio)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    mem_after = get_memory_info()

    # Results
    avg_time = np.mean(times)
    std_time = np.std(times)
    audio_duration = len(audio) / sample_rate
    realtime_factor = avg_time / audio_duration

    results = {
        'backend': backend_name,
        'avg_time': avg_time,
        'std_time': std_time,
        'audio_duration': audio_duration,
        'realtime_factor': realtime_factor,
        'can_realtime': realtime_factor < 1.0,
    }

    print(f"\nResults for {backend_name}:")
    print(f"  Average time: {avg_time:.2f}s (+/- {std_time:.2f}s)")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  Real-time factor: {realtime_factor:.1f}x")
    print(f"  Can do real-time: {'YES' if realtime_factor < 1.0 else 'NO'}")

    if mem_before and mem_after:
        print(f"  Memory: {mem_before['percent']:.1f}% -> {mem_after['percent']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark stem separation backends")
    parser.add_argument("--duration", type=float, default=3.0, help="Audio duration to test (seconds)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--backends", nargs="+", default=["spleeter", "demucs"],
                       help="Backends to test")
    args = parser.parse_args()

    print("="*60)
    print("  Vinyl Stripper - Backend Benchmark")
    print("="*60)

    # Platform info
    print(f"\nPlatform: {PI_MODEL if PI_MODEL else 'Desktop/Unknown'}")

    import multiprocessing
    print(f"CPU cores: {multiprocessing.cpu_count()}")

    mem = get_memory_info()
    if mem:
        print(f"Memory: {mem['available_gb']:.1f}GB / {mem['total_gb']:.1f}GB available")

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("GPU: Apple Silicon (MPS)")
        else:
            print("GPU: None (CPU only)")
    except:
        print("GPU: Unknown (torch not available)")

    # Generate test audio (sine wave + noise)
    print(f"\nGenerating {args.duration}s test audio...")
    t = np.linspace(0, args.duration, int(args.duration * args.sample_rate))
    audio = np.column_stack([
        0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t)),
        0.5 * np.sin(2 * np.pi * 880 * t) + 0.1 * np.random.randn(len(t)),
    ]).astype(np.float32)

    # Run benchmarks
    results = []
    for backend in args.backends:
        result = benchmark_backend(backend, audio, args.sample_rate)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    if not results:
        print("\nNo backends successfully tested.")
        print("Install backends with:")
        print("  pip install spleeter  # Fast, recommended")
        print("  pip install torch demucs  # High quality, slow")
        return

    # Sort by speed
    results.sort(key=lambda x: x['realtime_factor'])

    print(f"\n{'Backend':<15} {'Time':<10} {'RT Factor':<12} {'Real-time?':<10}")
    print("-" * 50)

    for r in results:
        rt_str = "YES" if r['can_realtime'] else "NO"
        print(f"{r['backend']:<15} {r['avg_time']:.2f}s     {r['realtime_factor']:.1f}x         {rt_str}")

    # Recommendation
    print("\n" + "="*60)
    print("  RECOMMENDATION")
    print("="*60)

    best = results[0]
    if best['can_realtime']:
        print(f"\n{best['backend'].upper()} can process in real-time!")
        print(f"Use: python3 vinyl_stripper_pi.py --backend {best['backend']}")
    else:
        print(f"\nNo backend can do real-time on this hardware.")
        print(f"Best option: {best['backend']} ({best['realtime_factor']:.1f}x real-time)")
        print(f"\nRecommendations:")
        print(f"  1. Use high-latency mode with {int(best['realtime_factor'] * 30)}s pre-buffer")
        print(f"  2. Process audio files offline instead of real-time")
        print(f"  3. Consider Raspberry Pi AI Kit (Hailo-8) for acceleration")

        # Calculate required pre-buffer for continuous playback
        # Need enough buffer so that while playing X seconds, we can process X seconds
        required_workers = int(np.ceil(best['realtime_factor']))
        print(f"\n  With {required_workers} workers and {int(best['realtime_factor'] * 60)}s pre-buffer,")
        print(f"  continuous playback may be possible.")


if __name__ == "__main__":
    main()
