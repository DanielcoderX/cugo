package gemv

import (
	_ "embed"
	"fmt"

	"github.com/DanielcoderX/cugo/driver"
)

//go:embed gemv.ptx
var PTX []byte

// ExecuteGEMV computes y = alpha * A * x + beta * y on matrix A of shape [m, n].
func ExecuteGEMV(ctx *driver.Context, stream *driver.Stream, dA, dX, dY driver.DevicePtr, m, n int, alpha, beta float32) error {
	if m <= 0 || n <= 0 {
		return fmt.Errorf("cugo/gemv: invalid dimensions m=%d, n=%d", m, n)
	}

	mod, err := ctx.LoadModuleData(PTX)
	if err != nil {
		return fmt.Errorf("cugo/gemv: LoadModuleData: %w", err)
	}
	defer mod.Unload()

	fn, err := mod.Function("gemv_forward_kernel")
	if err != nil {
		return fmt.Errorf("cugo/gemv: Function: %w", err)
	}

	// 128 threads per block = 4 warps = 4 rows per block
	const threadsPerBlock = 128
	const warpsPerBlock = threadsPerBlock / 32
	gridDim := uint32((m + warpsPerBlock - 1) / warpsPerBlock)

	cfg := driver.LaunchConfig{
		GridDimX:  gridDim,
		BlockDimX: threadsPerBlock,
		Stream:    stream,
	}

	return fn.Launch(cfg, dA, dX, dY, int32(m), int32(n), alpha, beta)
}
