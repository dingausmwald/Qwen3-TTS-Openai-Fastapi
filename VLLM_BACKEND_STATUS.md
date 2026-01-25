# vLLM Backend Status Report

## Summary

As of January 25, 2026, the vLLM backend implementation in this repository is **NOT FUNCTIONAL**. The code references a `vllm.Omni` class that does not exist in the current vLLM library (v0.14.1).

## Investigation Findings

### vLLM API Issue

The vLLM backend code (`api/backends/vllm_omni_qwen3_tts.py`) attempts to import:
```python
from vllm import Omni
```

However, vLLM 0.14.1 does not expose an `Omni` class. Available classes in vLLM include:
- `LLM` - Main inference class
- `MultiModalRegistry` - For multi-modal support
- Various other utilities

The vLLM library does have multi-modal capabilities, but they don't include a ready-to-use audio generation interface called `Omni`.

### Status of vLLM-Omni

The term "vLLM-Omni" appears to refer to:
1. A future feature planned for vLLM
2. A separate fork or extension not yet in the main vLLM repository
3. A placeholder implementation awaiting proper vLLM audio support

As of this date, standard vLLM does not have native TTS/audio generation capabilities that can be directly used for Qwen3-TTS.

## Recommendations

### Short-term

1. **Use Official Backend**: The official backend (`TTS_BACKEND=official`) works correctly and provides excellent performance
2. **Remove vLLM Claims**: Update documentation to remove claims about vLLM backend functionality until it's properly implemented
3. **Mark as Experimental**: Clearly label the vLLM backend as "planned" or "experimental" rather than functional

### Long-term

To implement a working vLLM backend, one of these approaches is needed:

1. **Wait for vLLM-Omni**: Monitor vLLM development for native TTS support
2. **Custom Implementation**: Build a custom adapter that uses vLLM's multi-modal capabilities to integrate Qwen3-TTS
3. **Alternative Optimization**: Use other optimization techniques like:
   - ONNX Runtime
   - TensorRT
   - Optimum
   - BetterTransformer / FlashAttention (already supported)

## Docker/GPU Issues

Additionally, during testing we encountered GPU passthrough issues with the current Docker configuration. The container can access `nvidia-smi` but PyTorch reports "No CUDA GPUs are available". This appears to be a container build/configuration issue unrelated to the vLLM backend problem.

### Working Configuration

For GPU inference, users should ensure:
- Proper NVIDIA Container Toolkit installation
- Correct docker-compose GPU device configuration
- Matching CUDA versions between container and host driver

## Conclusion

The vLLM backend should be considered **non-functional** and **not ready for use**. The official backend remains the recommended option for all deployment scenarios.

---

*Report generated: January 25, 2026*
*vLLM version tested: 0.14.1*
*PyTorch version: 2.5.1+cu121*
