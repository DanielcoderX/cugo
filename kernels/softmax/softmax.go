package softmax

import (
	_ "embed"
	"fmt"

	"github.com/cugo/cugo/driver"
)

//go:embed softmax.ptx
var PTX []byte

// Execute runs the warp-shuffle online softmax kernel on an input matrix of shape [rows, cols].
func Execute(ctx *driver.Context, stream *driver.Stream, dInput, dOutput driver.DevicePtr, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cugo/softmax: invalid dimensions rows=%d, cols=%d", rows, cols)
	}

	mod, err := ctx.LoadModuleData(PTX)
	if err != nil {
		return fmt.Errorf("cugo/softmax: LoadModuleData: %w", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("softmax_forward_kernel")
	if err != nil {
		return fmt.Errorf("cugo/softmax: Function: %w", err)
	}

	// 256 threads per block (8 warps)
	blockSize := uint32(256)
	if cols < 256 {
		// round up to nearest multiple of 32
		blockSize = uint32(((cols + 31) / 32) * 32)
		if blockSize < 32 {
			blockSize = 32
		}
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32(rows),
		BlockDimX: blockSize,
		Stream:    stream,
	}

	return fn.Launch(cfg, dInput, dOutput, int32(rows), int32(cols))
}
