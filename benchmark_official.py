#!/usr/bin/env python3
"""
Comprehensive benchmark for Qwen3-TTS Official Backend
Measures performance across multiple text lengths and provides detailed metrics
"""

import time
import json
import subprocess
from pathlib import Path
from openai import OpenAI

# Test cases with varying lengths
TEST_CASES = [
    {
        "name": "Ultra Short",
        "text": "Hi there",
        "expected_words": 2,
    },
    {
        "name": "Short",
        "text": "Hello world",
        "expected_words": 2,
    },
    {
        "name": "Medium",
        "text": "The quick brown fox jumps over the lazy dog near the riverbank",
        "expected_words": 12,
    },
    {
        "name": "Medium-Long",
        "text": "In the heart of the bustling city, where skyscrapers touch the clouds and the streets are always alive with activity, there exists a small park that serves as an oasis of tranquility.",
        "expected_words": 33,
    },
    {
        "name": "Long",
        "text": "Artificial intelligence has revolutionized the way we interact with technology, transforming everything from simple voice assistants to complex data analysis systems. Machine learning algorithms can now recognize patterns, make predictions, and even generate creative content that rivals human capabilities.",
        "expected_words": 42,
    },
    {
        "name": "Very Long",
        "text": "The advancement of text-to-speech technology has made it possible to create incredibly natural-sounding voices that can convey emotion, adjust pacing, and maintain consistency across long passages. Modern TTS systems utilize deep learning models trained on vast amounts of audio data, enabling them to produce speech that is nearly indistinguishable from human narrators. These systems are now being used in applications ranging from accessibility tools for the visually impaired to audiobook narration and virtual assistants.",
        "expected_words": 81,
    },
]

VOICES = ["Vivian", "Ryan"]
OUTPUT_DIR = Path("/tmp/tts_benchmark")


def run_benchmark():
    """Run comprehensive TTS benchmarks"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    client = OpenAI(
        base_url="http://localhost:8880/v1",
        api_key="not-needed"
    )
    
    results = []
    
    print("=" * 80)
    print("QWEN3-TTS OFFICIAL BACKEND BENCHMARK")
    print("=" * 80)
    print()
    
    # Get GPU info
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
        print(f"GPU: {gpu_info}")
    except:
        print("GPU: N/A")
    
    print()
    
    for test_case in TEST_CASES:
        name = test_case["name"]
        text = test_case["text"]
        words = len(text.split())
        chars = len(text)
        
        print(f"Testing: {name} ({words} words, {chars} chars)")
        print(f"Text: {text[:60]}..." if len(text) > 60 else f"Text: {text}")
        print()
        
        test_results = []
        
        for voice in VOICES:
            # Run the test
            output_file = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_{voice.lower()}.mp3"
            
            print(f"  Voice: {voice}...", end=" ", flush=True)
            
            start_time = time.time()
            try:
                response = client.audio.speech.create(
                    model="qwen3-tts",
                    voice=voice,
                    input=text,
                    response_format="mp3",
                    speed=1.0
                )
                
                # Save to file
                response.stream_to_file(str(output_file))
                
                elapsed = time.time() - start_time
                
                # Get file size
                file_size = output_file.stat().st_size
                
                result = {
                    "test_case": name,
                    "voice": voice,
                    "text_length": chars,
                    "word_count": words,
                    "generation_time": round(elapsed, 2),
                    "file_size_kb": round(file_size / 1024, 2),
                    "chars_per_second": round(chars / elapsed, 2),
                    "words_per_second": round(words / elapsed, 2),
                    "success": True
                }
                
                test_results.append(result)
                print(f"{elapsed:.2f}s ✓")
                
            except Exception as e:
                print(f"FAILED: {e}")
                result = {
                    "test_case": name,
                    "voice": voice,
                    "text_length": chars,
                    "word_count": words,
                    "success": False,
                    "error": str(e)
                }
                test_results.append(result)
        
        # Average results for this test case
        successful = [r for r in test_results if r.get("success")]
        if successful:
            avg_time = sum(r["generation_time"] for r in successful) / len(successful)
            avg_cps = sum(r["chars_per_second"] for r in successful) / len(successful)
            avg_wps = sum(r["words_per_second"] for r in successful) / len(successful)
            
            print(f"  Average: {avg_time:.2f}s ({avg_cps:.1f} chars/s, {avg_wps:.1f} words/s)")
        
        print()
        results.extend(test_results)
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    successful = [r for r in results if r.get("success")]
    if successful:
        total_tests = len(results)
        success_rate = len(successful) / total_tests * 100
        
        avg_time = sum(r["generation_time"] for r in successful) / len(successful)
        min_time = min(r["generation_time"] for r in successful)
        max_time = max(r["generation_time"] for r in successful)
        
        avg_cps = sum(r["chars_per_second"] for r in successful) / len(successful)
        avg_wps = sum(r["words_per_second"] for r in successful) / len(successful)
        
        print(f"Total Tests: {total_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Generation Time: {avg_time:.2f}s")
        print(f"Min/Max Time: {min_time:.2f}s / {max_time:.2f}s")
        print(f"Average Throughput: {avg_cps:.1f} chars/s ({avg_wps:.1f} words/s)")
        print()
        
        # Group by test case
        print("By Test Case:")
        print("-" * 80)
        for test_case in TEST_CASES:
            case_results = [r for r in successful if r["test_case"] == test_case["name"]]
            if case_results:
                avg = sum(r["generation_time"] for r in case_results) / len(case_results)
                words = case_results[0]["word_count"]
                print(f"{test_case['name']:15} ({words:2} words): {avg:6.2f}s avg")
    
    # Save results
    results_file = OUTPUT_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "official",
            "device": "GPU",
            "results": results
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {results_file}")
    print(f"Audio files saved to: {OUTPUT_DIR}")
    print()
    
    return results


if __name__ == "__main__":
    run_benchmark()
