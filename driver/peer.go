//go:build windows

package driver

import (
	"errors"
	"fmt"

	"github.com/cugo/cugo/internal/nvapi"
)

// CanAccessPeer queries whether device d can directly access memory allocated on peer.
func (d Device) CanAccessPeer(peer Device) (bool, error) {
	var canAccess int32
	if err := nvapi.CuDeviceCanAccessPeer(&canAccess, d.handle, peer.handle); err != nil {
		return false, fmt.Errorf("cugo: cuDeviceCanAccessPeer: %w", err)
	}
	return canAccess != 0, nil
}

// P2PAttribute queries a specific peer-to-peer connection attribute between device d and peer.
func (d Device) P2PAttribute(attr nvapi.CUdevice_P2PAttribute, peer Device) (int, error) {
	var val int32
	if err := nvapi.CuDeviceGetP2PAttribute(&val, attr, d.handle, peer.handle); err != nil {
		return 0, fmt.Errorf("cugo: cuDeviceGetP2PAttribute: %w", err)
	}
	return int(val), nil
}

// EnablePeerAccess enables direct GPU-to-GPU access to allocations in peerContext.
func (c *Context) EnablePeerAccess(peerContext *Context, flags ...uint32) error {
	if peerContext == nil {
		return errors.New("cugo: peerContext must not be nil")
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}

	var f uint32
	if len(flags) > 0 {
		f = flags[0]
	}

	if err := nvapi.CuCtxEnablePeerAccess(peerContext.handle, f); err != nil {
		return fmt.Errorf("cugo: cuCtxEnablePeerAccess: %w", err)
	}
	return nil
}

// DisablePeerAccess disables direct peer-to-peer access to allocations in peerContext.
func (c *Context) DisablePeerAccess(peerContext *Context) error {
	if peerContext == nil {
		return errors.New("cugo: peerContext must not be nil")
	}
	if err := c.EnsureCurrent(); err != nil {
		return err
	}

	if err := nvapi.CuCtxDisablePeerAccess(peerContext.handle); err != nil {
		return fmt.Errorf("cugo: cuCtxDisablePeerAccess: %w", err)
	}
	return nil
}

// CopyPeer synchronously copies memory between two distinct CUDA contexts.
func CopyPeer(
	dstContext *Context,
	dst DevicePtr,
	srcContext *Context,
	src DevicePtr,
	byteCount uint64,
) error {
	if dstContext == nil || srcContext == nil {
		return errors.New("cugo: dstContext and srcContext must not be nil")
	}
	if dst == 0 || src == 0 {
		return ErrNullPointer
	}
	if byteCount == 0 {
		return nil
	}

	if err := nvapi.CuMemcpyPeer(
		nvapi.CUdeviceptr(dst),
		dstContext.handle,
		nvapi.CUdeviceptr(src),
		srcContext.handle,
		byteCount,
	); err != nil {
		return fmt.Errorf("cugo: cuMemcpyPeer: %w", err)
	}
	return nil
}

// CopyPeerAsync asynchronously copies memory between two distinct CUDA contexts on a stream.
func CopyPeerAsync(
	dstContext *Context,
	dst DevicePtr,
	srcContext *Context,
	src DevicePtr,
	byteCount uint64,
	stream *Stream,
) error {
	if dstContext == nil || srcContext == nil {
		return errors.New("cugo: dstContext and srcContext must not be nil")
	}
	if dst == 0 || src == 0 {
		return ErrNullPointer
	}
	if byteCount == 0 {
		return nil
	}

	var hStream nvapi.CUstream
	if stream != nil {
		hStream = stream.handle
	}

	if err := nvapi.CuMemcpyPeerAsync(
		nvapi.CUdeviceptr(dst),
		dstContext.handle,
		nvapi.CUdeviceptr(src),
		srcContext.handle,
		byteCount,
		hStream,
	); err != nil {
		return fmt.Errorf("cugo: cuMemcpyPeerAsync: %w", err)
	}
	return nil
}
