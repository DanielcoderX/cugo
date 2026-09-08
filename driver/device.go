//go:build windows || linux

package driver

import (
	"bytes"
	"fmt"

	"github.com/cugo/cugo/internal/nvapi"
)

// Device represents a handle to a CUDA compute device.
type Device struct {
	handle nvapi.CUdevice
}

// Init initializes the CUDA driver API. Must be called before calling any other driver methods.
// Optional flags can be provided (default 0).
func Init(flags ...uint32) error {
	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}
	return nvapi.CuInit(f)
}

// DriverVersion returns the CUDA driver version supported by the installed driver.
// E.g. 13040 corresponds to CUDA 13.4.
func DriverVersion() (int, error) {
	var v int32
	if err := nvapi.CuDriverGetVersion(&v); err != nil {
		return 0, fmt.Errorf("cugo: cuDriverGetVersion: %w", err)
	}
	return int(v), nil
}

// DeviceCount returns the number of compute-capable devices available.
func DeviceCount() (int, error) {
	var c int32
	if err := nvapi.CuDeviceGetCount(&c); err != nil {
		return 0, fmt.Errorf("cugo: cuDeviceGetCount: %w", err)
	}
	return int(c), nil
}

// Devices returns all available compute devices on the system.
func Devices() ([]Device, error) {
	count, err := DeviceCount()
	if err != nil {
		return nil, err
	}
	devs := make([]Device, count)
	for i := 0; i < count; i++ {
		d, err := GetDevice(i)
		if err != nil {
			return nil, err
		}
		devs[i] = d
	}
	return devs, nil
}

// GetDevice returns a handle to a compute device by its ordinal index.
func GetDevice(ordinal int) (Device, error) {
	var dev nvapi.CUdevice
	if err := nvapi.CuDeviceGet(&dev, int32(ordinal)); err != nil {
		return Device{}, fmt.Errorf("cugo: cuDeviceGet(ordinal=%d): %w", ordinal, err)
	}
	return Device{handle: dev}, nil
}

// Handle returns the underlying raw CUdevice handle.
func (d Device) Handle() int32 {
	return int32(d.handle)
}

// Name returns the device's model name (e.g. "NVIDIA GeForce RTX 4060").
func (d Device) Name() (string, error) {
	buf := make([]byte, 256)
	if err := nvapi.CuDeviceGetName(buf, d.handle); err != nil {
		return "", fmt.Errorf("cugo: cuDeviceGetName: %w", err)
	}
	n := bytes.IndexByte(buf, 0)
	if n < 0 {
		n = len(buf)
	}
	return string(buf[:n]), nil
}

// ComputeCapability returns the major and minor compute capability version numbers.
func (d Device) ComputeCapability() (major, minor int, err error) {
	var maj, min int32
	if err := nvapi.CuDeviceGetAttribute(&maj, nvapi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d.handle); err != nil {
		return 0, 0, fmt.Errorf("cugo: compute capability major: %w", err)
	}
	if err := nvapi.CuDeviceGetAttribute(&min, nvapi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d.handle); err != nil {
		return 0, 0, fmt.Errorf("cugo: compute capability minor: %w", err)
	}
	return int(maj), int(min), nil
}

// TotalMemory returns the total amount of global memory on the device in bytes.
func (d Device) TotalMemory() (uint64, error) {
	var total uint64
	if err := nvapi.CuDeviceTotalMem(&total, d.handle); err != nil {
		return 0, fmt.Errorf("cugo: cuDeviceTotalMem: %w", err)
	}
	return total, nil
}

// Attribute queries a specific device attribute from the driver.
func (d Device) Attribute(attr nvapi.CUdevice_attribute) (int, error) {
	var val int32
	if err := nvapi.CuDeviceGetAttribute(&val, attr, d.handle); err != nil {
		return 0, fmt.Errorf("cugo: cuDeviceGetAttribute(%d): %w", attr, err)
	}
	return int(val), nil
}
