//go:build windows || linux

package driver

import (
	"errors"

	"github.com/DanielcoderX/cugo/internal/nvapi"
)

var (
	ErrFunctionNil = errors.New("cugo: nil function handle")
)

// MaxActiveBlocksPerMultiprocessor returns the maximum active blocks per multiprocessor
// for the function given the specified block size and dynamic shared memory size.
func (f *Function) MaxActiveBlocksPerMultiprocessor(blockSize int, dynamicSMemSize uint64) (int, error) {
	if f == nil || f.handle == 0 {
		return 0, ErrFunctionNil
	}
	var numBlocks int32
	if err := nvapi.CuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, f.handle, int32(blockSize), dynamicSMemSize); err != nil {
		return 0, err
	}
	return int(numBlocks), nil
}

// SuggestBlockSize computes a launch configuration (minimum grid size and optimal block size)
// that achieves maximum potential occupancy on the multiprocessor.
// If blockSizeLimit is <= 0, no limit is imposed.
func (f *Function) SuggestBlockSize(dynamicSMemSize uint64, blockSizeLimit int) (minGridSize, optimalBlockSize int, err error) {
	if f == nil || f.handle == 0 {
		return 0, 0, ErrFunctionNil
	}
	var minGrid, block int32
	if err := nvapi.CuOccupancyMaxPotentialBlockSize(
		&minGrid,
		&block,
		f.handle,
		0,
		dynamicSMemSize,
		int32(blockSizeLimit),
	); err != nil {
		return 0, 0, err
	}
	return int(minGrid), int(block), nil
}
