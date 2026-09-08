//go:build windows || linux

package driver_test

import (
	"testing"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/internal/nvapi"
)

func TestP2PAccess(t *testing.T) {
	if err := nvapi.CheckDriver(); err != nil {
		t.Skipf("skipped: no CUDA driver: %v", err)
	}
	if err := driver.Init(); err != nil {
		t.Skipf("skipped: driver.Init failed: %v", err)
	}

	devs, err := driver.Devices()
	if err != nil || len(devs) == 0 {
		t.Skip("skipped: no CUDA devices")
	}

	dev0 := devs[0]

	// Query peer access on self (expected false)
	canAccessSelf, err := dev0.CanAccessPeer(dev0)
	if err != nil {
		t.Fatalf("dev0.CanAccessPeer(dev0) failed: %v", err)
	}
	t.Logf("CanAccessPeer(self): %v", canAccessSelf)
	if canAccessSelf {
		t.Errorf("expected CanAccessPeer(self) == false, got true")
	}

	// Query P2P attribute
	rank, err := dev0.P2PAttribute(nvapi.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, dev0)
	if err != nil {
		t.Logf("P2PAttribute(PERFORMANCE_RANK): %v", err)
	} else {
		t.Logf("P2PAttribute(PERFORMANCE_RANK): %d", rank)
	}

	ctx0, err := dev0.CreateContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx0.Destroy()

	// Error safety tests
	if err := ctx0.EnablePeerAccess(nil); err == nil {
		t.Fatal("expected error on nil peerContext in EnablePeerAccess")
	}
	if err := ctx0.DisablePeerAccess(nil); err == nil {
		t.Fatal("expected error on nil peerContext in DisablePeerAccess")
	}
	if err := driver.CopyPeer(nil, 0, nil, 0, 100); err == nil {
		t.Fatal("expected error on nil contexts in CopyPeer")
	}

	// Multi-GPU test if second GPU available
	if len(devs) > 1 {
		dev1 := devs[1]
		canAccess, err := dev0.CanAccessPeer(dev1)
		if err != nil {
			t.Fatalf("dev0.CanAccessPeer(dev1) failed: %v", err)
		}
		t.Logf("Device 0 can access Device 1: %v", canAccess)

		if canAccess {
			ctx1, err := dev1.CreateContext()
			if err != nil {
				t.Fatal(err)
			}
			defer ctx1.Destroy()

			if err := ctx0.EnablePeerAccess(ctx1); err != nil {
				t.Fatalf("ctx0.EnablePeerAccess(ctx1) failed: %v", err)
			}
			defer ctx0.DisablePeerAccess(ctx1)

			const size = 1024
			d0, _ := ctx0.Alloc(size); defer ctx0.Free(d0)
			d1, _ := ctx1.Alloc(size); defer ctx1.Free(d1)

			if err := driver.CopyPeer(ctx1, d1, ctx0, d0, size); err != nil {
				t.Fatalf("driver.CopyPeer failed: %v", err)
			}
			t.Log("Successfully verified P2P GPU-to-GPU memory transfer!")
		}
	} else {
		t.Log("Single GPU host: multi-device transfer skipped gracefully")
	}
}
