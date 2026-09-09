package layernorm

import (
	_ "embed"
	"fmt"

	"github.com/DanielcoderX/cugo/driver"
)

//go:embed rmsnorm.ptx
var PTX []byte

// ExecuteRMSNorm executes the fused warp-shuffle RMSNorm kernel on an input matrix of shape [rows, cols].
// If dWeight is 0, uniform scaling (1.0) is assumed without extra memory reads.
func ExecuteRMSNorm(ctx *driver.Context, stream *driver.Stream, dInput, dOutput, dWeight driver.DevicePtr, rows, cols int, eps float32) error {
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("cugo/layernorm: invalid dimensions rows=%d, cols=%d", rows, cols)
	}

	mod, err := ctx.LoadModuleData(PTX)
	if err != nil {
		return fmt.Errorf("cugo/layernorm: LoadModuleData: %w", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("rmsnorm_forward_kernel")
	if err != nil {
		return fmt.Errorf("cugo/layernorm: Function: %w", err)
	}

	blockSize := uint32(256)
	if cols < 256 {
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

	return fn.Launch(cfg, dInput, dOutput, dWeight, int32(rows), int32(cols), eps)
}
