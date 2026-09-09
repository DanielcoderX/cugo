//go:build windows || linux

package driver

import (
	"errors"
	"fmt"
	"sync"

	"github.com/DanielcoderX/cugo/internal/nvapi"
)

var (
	ErrTextureNil       = errors.New("cugo: nil texture object handle")
	ErrTextureDestroyed = errors.New("cugo: texture object already destroyed")
	ErrSurfaceNil       = errors.New("cugo: nil surface object handle")
	ErrSurfaceDestroyed = errors.New("cugo: surface object already destroyed")
)

// TextureObject wraps a 64-bit CUDA texture object handle.
type TextureObject struct {
	mu        sync.Mutex
	handle    nvapi.CUtexObject
	destroyed bool
}

// SurfaceObject wraps a 64-bit CUDA surface object handle.
type SurfaceObject struct {
	mu        sync.Mutex
	handle    nvapi.CUsurfObject
	destroyed bool
}

// Handle returns the underlying 64-bit handle.
func (t *TextureObject) Handle() uint64 {
	return uint64(t.handle)
}

// Handle returns the underlying 64-bit handle.
func (s *SurfaceObject) Handle() uint64 {
	return uint64(s.handle)
}

// CreateTextureObject creates a texture object bound to the given 2D array.
func (c *Context) CreateTextureObject(arr *Array2D) (*TextureObject, error) {
	if c == nil || c.handle == 0 {
		return nil, ErrContextDestroyed
	}
	if arr == nil || arr.handle == 0 {
		return nil, ErrArrayNil
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var resDesc nvapi.CUDA_RESOURCE_DESC
	resDesc.ResType = nvapi.CU_RESOURCE_TYPE_ARRAY
	resDesc.ResData[0] = uint64(arr.handle)

	var texDesc nvapi.CUDA_TEXTURE_DESC
	texDesc.AddressMode[0] = nvapi.CU_TR_ADDRESS_MODE_CLAMP
	texDesc.AddressMode[1] = nvapi.CU_TR_ADDRESS_MODE_CLAMP
	texDesc.AddressMode[2] = nvapi.CU_TR_ADDRESS_MODE_CLAMP
	texDesc.FilterMode = nvapi.CU_TR_FILTER_MODE_LINEAR

	var hTex nvapi.CUtexObject
	if err := nvapi.CuTexObjectCreate(&hTex, &resDesc, &texDesc, nil); err != nil {
		return nil, fmt.Errorf("cugo: cuTexObjectCreate: %w", err)
	}

	return &TextureObject{handle: hTex}, nil
}

// Destroy destroys the texture object.
func (t *TextureObject) Destroy() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.destroyed {
		return ErrTextureDestroyed
	}
	t.destroyed = true
	if t.handle != 0 {
		h := t.handle
		t.handle = 0
		return nvapi.CuTexObjectDestroy(h)
	}
	return nil
}

// CreateSurfaceObject creates a surface object bound to the given 2D array.
func (c *Context) CreateSurfaceObject(arr *Array2D) (*SurfaceObject, error) {
	if c == nil || c.handle == 0 {
		return nil, ErrContextDestroyed
	}
	if arr == nil || arr.handle == 0 {
		return nil, ErrArrayNil
	}
	if err := c.EnsureCurrent(); err != nil {
		return nil, err
	}

	var resDesc nvapi.CUDA_RESOURCE_DESC
	resDesc.ResType = nvapi.CU_RESOURCE_TYPE_ARRAY
	resDesc.ResData[0] = uint64(arr.handle)

	var hSurf nvapi.CUsurfObject
	if err := nvapi.CuSurfObjectCreate(&hSurf, &resDesc); err != nil {
		return nil, fmt.Errorf("cugo: cuSurfObjectCreate: %w", err)
	}

	return &SurfaceObject{handle: hSurf}, nil
}

// Destroy destroys the surface object.
func (s *SurfaceObject) Destroy() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.destroyed {
		return ErrSurfaceDestroyed
	}
	s.destroyed = true
	if s.handle != 0 {
		h := s.handle
		s.handle = 0
		return nvapi.CuSurfObjectDestroy(h)
	}
	return nil
}
