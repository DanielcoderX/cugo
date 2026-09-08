//go:build windows || linux

package driver_test

import (
	"math"
	"testing"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/vecadd"
)

func TestGraphNodeDynamicParamsUpdate(t *testing.T) {
	if err := driver.Init(); err != nil {
		t.Fatalf("driver.Init failed: %v", err)
	}

	count, err := driver.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := driver.GetDevice(0)
	if err != nil {
		t.Fatalf("GetDevice(0) failed: %v", err)
	}

	ctx, err := dev.CreateContext()
	if err != nil {
		t.Fatalf("CreateContext failed: %v", err)
	}
	defer ctx.Destroy()

	mod, err := ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		t.Fatalf("LoadModuleData failed: %v", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		t.Fatalf("mod.Function failed: %v", err)
	}


	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}
	defer stream.Destroy()

	const n = 1024
	const sizeBytes = n * 4

	hA := make([]float32, n)
	hB1 := make([]float32, n)
	hB2 := make([]float32, n)
	for i := range hA {
		hA[i] = 10.0
		hB1[i] = 20.0
		hB2[i] = 50.0
	}

	dA, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("ctx.Alloc dA failed: %v", err)
	}
	defer ctx.Free(dA)

	dB1, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("ctx.Alloc dB1 failed: %v", err)
	}
	defer ctx.Free(dB1)

	dB2, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("ctx.Alloc dB2 failed: %v", err)
	}
	defer ctx.Free(dB2)

	dC, err := ctx.Alloc(sizeBytes)
	if err != nil {
		t.Fatalf("ctx.Alloc dC failed: %v", err)
	}
	defer ctx.Free(dC)

	if err := ctx.CopyHtoD(dA, unsafe.Slice((*byte)(unsafe.Pointer(&hA[0])), sizeBytes)); err != nil {
		t.Fatalf("CopyHtoD dA failed: %v", err)
	}
	if err := ctx.CopyHtoD(dB1, unsafe.Slice((*byte)(unsafe.Pointer(&hB1[0])), sizeBytes)); err != nil {
		t.Fatalf("CopyHtoD dB1 failed: %v", err)
	}
	if err := ctx.CopyHtoD(dB2, unsafe.Slice((*byte)(unsafe.Pointer(&hB2[0])), sizeBytes)); err != nil {
		t.Fatalf("CopyHtoD dB2 failed: %v", err)
	}

	// 1. Capture graph with vecAdd(dA, dB1, dC, n) -> 10 + 20 = 30
	if err := stream.BeginCapture(); err != nil {
		t.Fatalf("BeginCapture failed: %v", err)
	}

	cfg := driver.LaunchConfig{
		GridDimX:  4,
		BlockDimX: 256,
		Stream:    stream,
	}

	if err := fn.Launch(cfg, dA, dB1, dC, int32(n)); err != nil {
		t.Fatalf("fn.Launch inside capture failed: %v", err)
	}

	graph, err := stream.EndCapture()
	if err != nil {
		t.Fatalf("EndCapture failed: %v", err)
	}
	defer graph.Destroy()

	// 2. Inspect graph nodes
	nodes, err := graph.Nodes()
	if err != nil {
		t.Fatalf("graph.Nodes failed: %v", err)
	}
	if len(nodes) == 0 {
		t.Fatalf("expected at least 1 node in graph, got 0")
	}

	var kernelNode *driver.GraphNode
	for _, node := range nodes {
		ntype, err := node.Type()
		if err != nil {
			t.Fatalf("node.Type failed: %v", err)
		}
		if ntype == driver.GraphNodeTypeKernel {
			kernelNode = node
			break
		}
	}
	if kernelNode == nil {
		t.Fatalf("expected to find a kernel node in graph")
	}

	// 3. Instantiate and run initial graph
	graphExec, err := graph.Instantiate()
	if err != nil {
		t.Fatalf("graph.Instantiate failed: %v", err)
	}
	defer graphExec.Destroy()

	if err := graphExec.Launch(stream); err != nil {
		t.Fatalf("graphExec.Launch failed: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	hResult := make([]float32, n)
	resBytes := unsafe.Slice((*byte)(unsafe.Pointer(&hResult[0])), sizeBytes)
	if err := ctx.CopyDtoH(resBytes, dC); err != nil {
		t.Fatalf("CopyDtoH failed: %v", err)
	}
	for i := 0; i < n; i++ {
		if math.Abs(float64(hResult[i]-30.0)) > 1e-4 {
			t.Fatalf("initial launch element %d: got %f, expected 30.0", i, hResult[i])
		}
	}

	// 4. Update kernel node parameters in-place (switch dB1 to dB2 -> 10 + 50 = 60)
	if err := graphExec.SetKernelNodeParams(kernelNode, fn, cfg, dA, dB2, dC, int32(n)); err != nil {
		t.Fatalf("SetKernelNodeParams failed: %v", err)
	}

	// 5. Replay graph without re-instantiation
	if err := graphExec.Launch(stream); err != nil {
		t.Fatalf("graphExec.Launch after param update failed: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	if err := ctx.CopyDtoH(resBytes, dC); err != nil {
		t.Fatalf("CopyDtoH failed: %v", err)
	}

	for i := 0; i < n; i++ {
		if math.Abs(float64(hResult[i]-60.0)) > 1e-4 {
			t.Fatalf("updated launch element %d: got %f, expected 60.0", i, hResult[i])
		}
	}


	t.Logf("Successfully updated kernel node parameters in-place on instantiated GraphExec!")
}
