//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/DanielcoderX/cugo/internal/nvapi"
)

var (
	ErrArrayNil       = errors.New("cugo: nil array handle")
	ErrArrayDestroyed = errors.New("cugo: array already destroyed")
)

// PitchedPtr represents a 2D pitched memory allocation on the GPU device.
type PitchedPtr struct {
	Ptr          DevicePtr
	Pitch        uint64
	WidthInBytes uint64
	Height       uint64
}

// AllocPitch allocates a 2D pitched array of linear memory on the GPU device.
// Pitch alignment ensures optimal coalesced memory access across rows.
func (c *Context) AllocPitch(widthInBytes, height uint64, elementSizeBytes uint32) (PitchedPtr, error) {
	if c == nil || c.handle == 0 {
		return PitchedPtr{}, ErrContextDestroyed
	}
	if widthInBytes == 0 || height == 0 {
		return PitchedPtr{}, ErrZeroSizeAlloc
	}
	if err := c.EnsureCurrent(); err != nil {
		return PitchedPtr{}, err
	}

	var dptr nvapi.CUdeviceptr
	var pitch uint64
	if err := nvapi.CuMemAllocPitch(&dptr, &pitch, widthInBytes, height, elementSizeBytes); err != nil {
		return PitchedPtr{}, fmt.Errorf("cugo: cuMemAllocPitch: %w", err)
	}

	return PitchedPtr{
		Ptr:          DevicePtr(dptr),
		Pitch:        pitch,
		WidthInBytes: widthInBytes,
		Height:       height,
	}, nil
}

// MemType represents source or destination memory type for 2D copies.
type MemType int32

const (
	MemTypeHost    MemType = MemType(nvapi.CU_MEMORYTYPE_HOST)
	MemTypeDevice  MemType = MemType(nvapi.CU_MEMORYTYPE_DEVICE)
	MemTypeArray   MemType = MemType(nvapi.CU_MEMORYTYPE_ARRAY)
	MemTypeUnified MemType = MemType(nvapi.CU_MEMORYTYPE_UNIFIED)
)

// Copy2DParams defines coordinates and layout for a 2D rectangular memory transfer.
type Copy2DParams struct {
	SrcXInBytes   uint64
	SrcY          uint64
	SrcMemoryType MemType
	SrcHost       unsafe.Pointer
	SrcDevice     DevicePtr
	SrcArray      *Array2D
	SrcPitch      uint64

	DstXInBytes   uint64
	DstY          uint64
	DstMemoryType MemType
	DstHost       unsafe.Pointer
	DstDevice     DevicePtr
	DstArray      *Array2D
	DstPitch      uint64

	WidthInBytes uint64
	Height       uint64
}

func (p *Copy2DParams) toNVAPI() nvapi.CUDA_MEMCPY2D {
	cp := nvapi.CUDA_MEMCPY2D{
		SrcXInBytes:   p.SrcXInBytes,
		SrcY:          p.SrcY,
		SrcMemoryType: nvapi.CUmemorytype(p.SrcMemoryType),
		SrcHost:       p.SrcHost,
		SrcDevice:     nvapi.CUdeviceptr(p.SrcDevice),
		SrcPitch:      p.SrcPitch,

		DstXInBytes:   p.DstXInBytes,
		DstY:          p.DstY,
		DstMemoryType: nvapi.CUmemorytype(p.DstMemoryType),
		DstHost:       p.DstHost,
		DstDevice:     nvapi.CUdeviceptr(p.DstDevice),
		DstPitch:      p.DstPitch,

		WidthInBytes: p.WidthInBytes,
		Height:       p.Height,
	}
	if p.SrcArray != nil {
		cp.SrcArray = p.SrcArray.handle
	}
	if p.DstArray != nil {
		cp.DstArray = p.DstArray.handle
	}
	return cp
}

// Copy2D performs a synchronous 2D matrix memory transfer.
func (c *Context) Copy2D(params Copy2DParams) error {
	if c == nil || c.handle == 0 {
		return ErrContextDestroyed
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}
	rawCopy := params.toNVAPI()
	return nvapi.CuMemcpy2D(&rawCopy)
}

// Copy2DAsync performs an asynchronous 2D matrix memory transfer on the stream.
func (s *Stream) Copy2DAsync(params Copy2DParams) error {
	if s == nil || s.handle == 0 {
		return ErrStreamDestroyed
	}
	rawCopy := params.toNVAPI()
	return nvapi.CuMemcpy2DAsync(&rawCopy, s.handle)
}

// ArrayFormat specifies data format for CUDA 2D/3D arrays.
type ArrayFormat int32

const (
	ArrayFormatUint8  ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_UNSIGNED_INT8)
	ArrayFormatUint16 ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_UNSIGNED_INT16)
	ArrayFormatUint32 ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_UNSIGNED_INT32)
	ArrayFormatInt8   ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_SIGNED_INT8)
	ArrayFormatInt16  ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_SIGNED_INT16)
	ArrayFormatInt32  ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_SIGNED_INT32)
	ArrayFormatHalf   ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_HALF)
	ArrayFormatFloat  ArrayFormat = ArrayFormat(nvapi.CU_AD_FORMAT_FLOAT)
)

// Array2D wraps a 2D CUDA hardware array (e.g. for textures or surfaces).
type Array2D struct {
	mu        sync.Mutex
	handle    nvapi.CUarray
	destroyed bool
	Width     uint64
	Height    uint64
	Format    ArrayFormat
	Channels  uint32
}

// CreateArray2D creates a 2D CUDA hardware array.
func (c *Context) CreateArray2D(width, height uint64, format ArrayFormat, numChannels uint32) (*Array2D, error) {
	if c == nil || c.handle == 0 {
		return nil, ErrContextDestroyed
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	desc := nvapi.CUDA_ARRAY_DESCRIPTOR{
		Width:       width,
		Height:      height,
		Format:      nvapi.CUarray_format(format),
		NumChannels: numChannels,
	}

	var hArray nvapi.CUarray
	if err := nvapi.CuArrayCreate(&hArray, &desc); err != nil {
		return nil, fmt.Errorf("cugo: cuArrayCreate: %w", err)
	}

	return &Array2D{
		handle:   hArray,
		Width:    width,
		Height:   height,
		Format:   format,
		Channels: numChannels,
	}, nil
}

// Destroy destroys the 2D CUDA hardware array.
func (a *Array2D) Destroy() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.destroyed {
		return ErrArrayDestroyed
	}
	a.destroyed = true
	if a.handle != 0 {
		h := a.handle
		a.handle = 0
		return nvapi.CuArrayDestroy(h)
	}
	return nil
}
