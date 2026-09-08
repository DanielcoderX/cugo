//go:build windows

package nvapi

import (
	"bytes"
	"testing"
)

func TestCuInitAndDeviceEnum(t *testing.T) {
	if err := CheckDriver(); err != nil {
		t.Skipf("skipped: no CUDA driver: %v", err)
	}

	if err := CuInit(0); err != nil {
		t.Skipf("skipped: cuInit failed (no GPU device?): %v", err)
	}

	var version int32
	if err := CuDriverGetVersion(&version); err != nil {
		t.Fatalf("cuDriverGetVersion failed: %v", err)
	}
	t.Logf("CUDA Driver Version: %d", version)
	if version <= 0 {
		t.Fatalf("invalid driver version: %d", version)
	}

	var count int32
	if err := CuDeviceGetCount(&count); err != nil {
		t.Fatalf("cuDeviceGetCount failed: %v", err)
	}
	t.Logf("CUDA Device Count: %d", count)
	if count <= 0 {
		t.Skip("skipped: no CUDA devices detected")
	}

	for i := int32(0); i < count; i++ {
		var dev CUdevice
		if err := CuDeviceGet(&dev, i); err != nil {
			t.Fatalf("cuDeviceGet(ordinal=%d) failed: %v", i, err)
		}

		nameBuf := make([]byte, 256)
		if err := CuDeviceGetName(nameBuf, dev); err != nil {
			t.Fatalf("cuDeviceGetName failed: %v", err)
		}
		name := string(bytes.TrimRight(nameBuf, "\x00"))
		t.Logf("Device %d: %s (handle: %d)", i, name, dev)

		var totalMem uint64
		if err := CuDeviceTotalMem(&totalMem, dev); err != nil {
			t.Fatalf("cuDeviceTotalMem failed: %v", err)
		}
		t.Logf("Device %d Total Memory: %d bytes (%.2f GiB)", i, totalMem, float64(totalMem)/(1024*1024*1024))

		var ccMajor, ccMinor int32
		if err := CuDeviceGetAttribute(&ccMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev); err != nil {
			t.Fatalf("cuDeviceGetAttribute(CC_MAJOR) failed: %v", err)
		}
		if err := CuDeviceGetAttribute(&ccMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev); err != nil {
			t.Fatalf("cuDeviceGetAttribute(CC_MINOR) failed: %v", err)
		}
		t.Logf("Device %d Compute Capability: %d.%d", i, ccMajor, ccMinor)

		if ccMajor == 0 {
			t.Fatalf("unexpected compute capability 0.0")
		}
	}
}
