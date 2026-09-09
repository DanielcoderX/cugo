//go:build windows || linux

package driver

import (
	"github.com/DanielcoderX/cugo/internal/nvapi"
)

// ProfilerStart starts CUDA profiling data collection for the current process.
// Typically used when running under NVIDIA Nsight Systems or Nsight Compute.
func ProfilerStart() error {
	return nvapi.CuProfilerStart()
}

// ProfilerStop stops CUDA profiling data collection for the current process.
func ProfilerStop() error {
	return nvapi.CuProfilerStop()
}
