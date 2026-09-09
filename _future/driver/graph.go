//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrGraphNil           = errors.New("cugo: nil graph handle")
	ErrGraphExecNil       = errors.New("cugo: nil executable graph handle")
	ErrGraphDestroyed     = errors.New("cugo: graph already destroyed")
	ErrGraphExecDestroyed = errors.New("cugo: graph exec already destroyed")
)

// GraphNodeType indicates the operational type of a graph node.
type GraphNodeType int32

const (
	GraphNodeTypeKernel GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_KERNEL)
	GraphNodeTypeMemcpy GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_MEMCPY)
	GraphNodeTypeMemset GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_MEMSET)
	GraphNodeTypeHost   GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_HOST)
	GraphNodeTypeGraph  GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_GRAPH)
	GraphNodeTypeEmpty  GraphNodeType = GraphNodeType(nvapi.CU_GRAPH_NODE_TYPE_EMPTY)
)

// GraphNode represents an individual execution or memory node within a CUDA graph.
type GraphNode struct {
	handle nvapi.CUgraphNode
}

// Handle returns the underlying driver node handle.
func (n *GraphNode) Handle() uintptr {
	if n == nil {
		return 0
	}
	return uintptr(n.handle)
}

// Type returns the node's operational type (kernel, memcpy, etc.).
func (n *GraphNode) Type() (GraphNodeType, error) {
	if n == nil || n.handle == 0 {
		return 0, errors.New("cugo: nil graph node")
	}
	var t nvapi.CUgraphNodeType
	if err := nvapi.CuGraphNodeGetType(n.handle, &t); err != nil {
		return 0, err
	}
	return GraphNodeType(t), nil
}

// StreamCaptureMode specifies how work dispatched during stream capture is tracked.
type StreamCaptureMode int32

const (
	CaptureModeGlobal      StreamCaptureMode = StreamCaptureMode(nvapi.CU_STREAM_CAPTURE_MODE_GLOBAL)
	CaptureModeThreadLocal StreamCaptureMode = StreamCaptureMode(nvapi.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
	CaptureModeRelaxed     StreamCaptureMode = StreamCaptureMode(nvapi.CU_STREAM_CAPTURE_MODE_RELAXED)
)

// BeginCapture initiates stream capture on the stream using default global mode.
// Any operations queued on this stream until EndCapture are captured into a graph rather than executed immediately.
func (s *Stream) BeginCapture() error {
	return s.BeginCaptureMode(CaptureModeGlobal)
}

// BeginCaptureMode initiates stream capture using the specified capture mode.
func (s *Stream) BeginCaptureMode(mode StreamCaptureMode) error {
	if s == nil || s.handle == 0 {
		return ErrStreamDestroyed
	}
	return nvapi.CuStreamBeginCapture(s.handle, nvapi.CUstreamCaptureMode(mode))
}

// EndCapture ends stream capture and returns the captured Graph.
func (s *Stream) EndCapture() (*Graph, error) {
	if s == nil || s.handle == 0 {
		return nil, ErrStreamDestroyed
	}
	var hGraph nvapi.CUgraph
	if err := nvapi.CuStreamEndCapture(s.handle, &hGraph); err != nil {
		return nil, err
	}
	return &Graph{handle: hGraph}, nil
}

// IsCapturing returns true if the stream is actively in stream capture mode.
func (s *Stream) IsCapturing() (bool, error) {
	if s == nil || s.handle == 0 {
		return false, ErrStreamDestroyed
	}
	var status nvapi.CUstreamCaptureStatus
	if err := nvapi.CuStreamIsCapturing(s.handle, &status); err != nil {
		return false, err
	}
	return status == nvapi.CU_STREAM_CAPTURE_STATUS_ACTIVE, nil
}

// Graph represents an uninstantiated CUDA graph DAG of compute and memory operations.
type Graph struct {
	mu        sync.Mutex
	handle    nvapi.CUgraph
	destroyed bool
}

// Nodes returns all nodes contained within the graph DAG.
func (g *Graph) Nodes() ([]*GraphNode, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.destroyed || g.handle == 0 {
		return nil, ErrGraphDestroyed
	}

	var numNodes uint64
	if err := nvapi.CuGraphGetNodes(g.handle, nil, &numNodes); err != nil {
		return nil, err
	}
	if numNodes == 0 {
		return nil, nil
	}
	rawNodes := make([]nvapi.CUgraphNode, numNodes)
	if err := nvapi.CuGraphGetNodes(g.handle, &rawNodes[0], &numNodes); err != nil {
		return nil, err
	}
	nodes := make([]*GraphNode, numNodes)
	for i := range nodes {
		nodes[i] = &GraphNode{handle: rawNodes[i]}
	}
	return nodes, nil
}

// Instantiate validates the graph topology and creates an executable GraphExec.
func (g *Graph) Instantiate() (*GraphExec, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.destroyed || g.handle == 0 {
		return nil, ErrGraphDestroyed
	}
	var hExec nvapi.CUgraphExec
	if err := nvapi.CuGraphInstantiate(&hExec, g.handle); err != nil {
		return nil, err
	}
	return &GraphExec{handle: hExec}, nil
}

// Destroy frees the CUDA graph.
func (g *Graph) Destroy() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.destroyed {
		return ErrGraphDestroyed
	}
	g.destroyed = true
	if g.handle != 0 {
		h := g.handle
		g.handle = 0
		return nvapi.CuGraphDestroy(h)
	}
	return nil
}

// GraphExec represents an instantiated, executable CUDA graph ready for low-latency launching.
type GraphExec struct {
	mu        sync.Mutex
	handle    nvapi.CUgraphExec
	destroyed bool
}

// SetKernelNodeParams updates the kernel parameters, launch bounds, and arguments
// of an existing kernel node within an instantiated executable graph.
func (ge *GraphExec) SetKernelNodeParams(node *GraphNode, fn *Function, cfg LaunchConfig, args ...any) error {
	ge.mu.Lock()
	defer ge.mu.Unlock()

	if ge.destroyed || ge.handle == 0 {
		return ErrGraphExecDestroyed
	}
	if node == nil || node.handle == 0 {
		return errors.New("cugo: nil graph node")
	}
	if fn == nil || fn.handle == 0 {
		return errors.New("cugo: nil or uninitialized Function")
	}

	gridX := cfg.GridDimX
	if gridX == 0 {
		gridX = 1
	}
	gridY := cfg.GridDimY
	if gridY == 0 {
		gridY = 1
	}
	gridZ := cfg.GridDimZ
	if gridZ == 0 {
		gridZ = 1
	}

	blockX := cfg.BlockDimX
	if blockX == 0 {
		blockX = 1
	}
	blockY := cfg.BlockDimY
	if blockY == 0 {
		blockY = 1
	}
	blockZ := cfg.BlockDimZ
	if blockZ == 0 {
		blockZ = 1
	}

	var kernelParams unsafe.Pointer
	var paramPtrs []unsafe.Pointer
	var kargs []KernelArg

	if len(args) > 0 {
		var err error
		paramPtrs, kargs, err = buildKernelArgs(args)
		if err != nil {
			return err
		}
		kernelParams = unsafe.Pointer(&paramPtrs[0])
	}

	nodeParams := nvapi.CUDA_KERNEL_NODE_PARAMS{
		Func:           fn.handle,
		GridDimX:       gridX,
		GridDimY:       gridY,
		GridDimZ:       gridZ,
		BlockDimX:      blockX,
		BlockDimY:      blockY,
		BlockDimZ:      blockZ,
		SharedMemBytes: cfg.SharedMemBytes,
		KernelParams:   (*unsafe.Pointer)(kernelParams),
		Extra:          nil,
	}

	err := nvapi.CuGraphExecKernelNodeSetParams(ge.handle, node.handle, &nodeParams)
	runtime.KeepAlive(kargs)
	runtime.KeepAlive(paramPtrs)
	if err != nil {
		return fmt.Errorf("cugo: cuGraphExecKernelNodeSetParams: %w", err)
	}
	return nil
}

// Launch enqueues execution of the entire graph onto the given stream.
// If stream is nil, the default stream is used.
func (ge *GraphExec) Launch(stream *Stream) error {
	ge.mu.Lock()
	defer ge.mu.Unlock()

	if ge.destroyed || ge.handle == 0 {
		return ErrGraphExecDestroyed
	}
	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}
	return nvapi.CuGraphLaunch(ge.handle, hStream)
}

// Destroy destroys the executable graph.
func (ge *GraphExec) Destroy() error {
	ge.mu.Lock()
	defer ge.mu.Unlock()

	if ge.destroyed {
		return ErrGraphExecDestroyed
	}
	ge.destroyed = true
	if ge.handle != 0 {
		h := ge.handle
		ge.handle = 0
		return nvapi.CuGraphExecDestroy(h)
	}
	return nil
}

