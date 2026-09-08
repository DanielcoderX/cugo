//go:build windows || linux

package driver_test

import (
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
)

func TestDeviceEnumeration(t *testing.T) {
	if err := nvapi.CheckDriver(); err != nil {
		t.Skipf("skipped: no CUDA driver: %v", err)
	}

	if err := driver.Init(); err != nil {
		t.Skipf("skipped: driver.Init failed: %v", err)
	}

	v, err := driver.DriverVersion()
	if err != nil {
		t.Fatalf("driver.DriverVersion failed: %v", err)
	}
	t.Logf("Driver Version: %d", v)

	devs, err := driver.Devices()
	if err != nil {
		t.Fatalf("driver.Devices failed: %v", err)
	}
	if len(devs) == 0 {
		t.Skip("skipped: no CUDA devices found")
	}

	for i, d := range devs {
		name, err := d.Name()
		if err != nil {
			t.Fatalf("dev[%d].Name failed: %v", i, err)
		}
		major, minor, err := d.ComputeCapability()
		if err != nil {
			t.Fatalf("dev[%d].ComputeCapability failed: %v", i, err)
		}
		mem, err := d.TotalMemory()
		if err != nil {
			t.Fatalf("dev[%d].TotalMemory failed: %v", i, err)
		}
		smCount, err := d.Attribute(nvapi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
		if err != nil {
			t.Fatalf("dev[%d].Attribute(MULTIPROCESSOR_COUNT) failed: %v", i, err)
		}

		t.Logf("GPU %d: %s | CC %d.%d | Memory: %.2f GiB | SMs: %d",
			i, name, major, minor, float64(mem)/(1024*1024*1024), smCount)

		if major <= 0 {
			t.Errorf("expected major compute capability > 0, got %d", major)
		}
	}
}
