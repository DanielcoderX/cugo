//go:build windows

package driver

import (
	"errors"
	"sync"

	"github.com/cugo/cugo/internal/nvapi"
)

var (
	ErrGraphNil        = errors.New("cugo: nil graph handle")
	ErrGraphExecNil    = errors.New("cugo: nil executable graph handle")
	ErrGraphDestroyed  = errors.New("cugo: graph already destroyed")
	ErrGraphExecDestroyed = errors.New("cugo: graph exec already destroyed")
)

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
