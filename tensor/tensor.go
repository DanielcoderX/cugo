package tensor

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/cugo/cugo/driver"
	"github.com/cugo/cugo/kernels/vecadd"
)

var (
	ErrShapeMismatch    = errors.New("cugo: tensor shape mismatch")
	ErrNonContiguous    = errors.New("cugo: view cannot be created on non-contiguous tensor (call Contiguous() first)")
	ErrDimensionInvalid = errors.New("cugo: dimension out of bounds")
	ErrIndexOutOfBounds = errors.New("cugo: index out of bounds")
)

// Tensor represents an n-dimensional array on the GPU, modeled directly on PyTorch's at::Tensor.
type Tensor struct {
	storage      *Storage
	offset       uint64  // Byte offset from storage base pointer
	shape        []int64 // Extent along each dimension
	strides      []int64 // Element stride along each dimension
	dtype        DType
	ctx          *driver.Context
	RequiresGrad bool
	Grad         *Tensor
	GradFn       Node
}

// computeContiguousStrides calculates the standard row-major (C-contiguous) strides for a shape.
func computeContiguousStrides(shape []int64) []int64 {
	strides := make([]int64, len(shape))
	if len(shape) == 0 {
		return strides
	}
	var acc int64 = 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = acc
		acc *= shape[i]
	}
	return strides
}

// numel computes the total number of elements in a shape.
func numel(shape []int64) int64 {
	if len(shape) == 0 {
		return 0
	}
	var n int64 = 1
	for _, dim := range shape {
		if dim < 0 {
			return -1
		}
		n *= dim
	}
	return n
}

// New creates an uninitialized Tensor on the GPU with the specified shape and data type.
func New(ctx *driver.Context, dtype DType, shape ...int64) (*Tensor, error) {
	n := numel(shape)
	if n <= 0 {
		return nil, ErrShapeMismatch
	}

	byteSize := uint64(n) * uint64(dtype.Size())
	storage, err := NewStorage(ctx, byteSize)
	if err != nil {
		return nil, err
	}

	shapeCopy := make([]int64, len(shape))
	copy(shapeCopy, shape)

	return &Tensor{
		storage: storage,
		offset:  0,
		shape:   shapeCopy,
		strides: computeContiguousStrides(shapeCopy),
		dtype:   dtype,
		ctx:     ctx,
	}, nil
}

// NewFromFloat32 creates a Float32 Tensor on the GPU initialized with host slice data.
func NewFromFloat32(ctx *driver.Context, data []float32, shape ...int64) (*Tensor, error) {
	n := numel(shape)
	if n != int64(len(data)) {
		return nil, fmt.Errorf("%w: element count %d does not match data len %d", ErrShapeMismatch, n, len(data))
	}

	t, err := New(ctx, Float32, shape...)
	if err != nil {
		return nil, err
	}

	rawBytes := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*4)
	if err := ctx.CopyHtoD(t.DevicePtr(), rawBytes); err != nil {
		_ = t.Close()
		return nil, fmt.Errorf("cugo: CopyHtoD failed: %w", err)
	}

	return t, nil
}

// DevicePtr returns the memory address pointing to the start of this Tensor's data.
func (t *Tensor) DevicePtr() driver.DevicePtr {
	return t.storage.DevicePtr().Offset(t.offset)
}

// Shape returns a copy of the tensor dimensions.
func (t *Tensor) Shape() []int64 {
	cp := make([]int64, len(t.shape))
	copy(cp, t.shape)
	return cp
}

// Strides returns a copy of the tensor element strides.
func (t *Tensor) Strides() []int64 {
	cp := make([]int64, len(t.strides))
	copy(cp, t.strides)
	return cp
}

// DType returns the element data type.
func (t *Tensor) DType() DType {
	return t.dtype
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int64 {
	return numel(t.shape)
}

// Dims returns the rank (number of dimensions).
func (t *Tensor) Dims() int {
	return len(t.shape)
}

// IsContiguous returns true if tensor data is laid out in standard row-major contiguous memory.
func (t *Tensor) IsContiguous() bool {
	expected := computeContiguousStrides(t.shape)
	for i := range expected {
		if t.strides[i] != expected[i] {
			return false
		}
	}
	return true
}

// View returns a new tensor with the same storage and new shape if contiguous.
func (t *Tensor) View(newShape ...int64) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, ErrNonContiguous
	}

	// Resolve -1 inferred dimension
	var inferIdx = -1
	var knownProduct int64 = 1
	for i, d := range newShape {
		if d == -1 {
			if inferIdx != -1 {
				return nil, errors.New("cugo: only one dimension can be inferred (-1)")
			}
			inferIdx = i
		} else if d > 0 {
			knownProduct *= d
		} else {
			return nil, ErrShapeMismatch
		}
	}

	resolvedShape := make([]int64, len(newShape))
	copy(resolvedShape, newShape)

	total := t.Numel()
	if inferIdx != -1 {
		if total%knownProduct != 0 {
			return nil, fmt.Errorf("cugo: cannot reshape tensor of size %d into shape %v", total, newShape)
		}
		resolvedShape[inferIdx] = total / knownProduct
	} else if numel(resolvedShape) != total {
		return nil, fmt.Errorf("cugo: cannot reshape tensor of size %d into shape %v", total, newShape)
	}

	t.storage.Retain()
	return &Tensor{
		storage: t.storage,
		offset:  t.offset,
		shape:   resolvedShape,
		strides: computeContiguousStrides(resolvedShape),
		dtype:   t.dtype,
		ctx:     t.ctx,
	}, nil
}

// Slice returns a zero-copy sub-tensor view along dimension dim from start to end (exclusive).
func (t *Tensor) Slice(dim int, start, end int64) (*Tensor, error) {
	if dim < 0 || dim >= len(t.shape) {
		return nil, ErrDimensionInvalid
	}
	dimSize := t.shape[dim]
	if start < 0 || end > dimSize || start >= end {
		return nil, ErrIndexOutOfBounds
	}

	newShape := make([]int64, len(t.shape))
	copy(newShape, t.shape)
	newShape[dim] = end - start

	elementSize := uint64(t.dtype.Size())
	byteOffsetAdvance := uint64(start * t.strides[dim]) * elementSize

	t.storage.Retain()
	return &Tensor{
		storage: t.storage,
		offset:  t.offset + byteOffsetAdvance,
		shape:   newShape,
		strides: t.Strides(),
		dtype:   t.dtype,
		ctx:     t.ctx,
	}, nil
}

// ToCPUFloat32 copies a contiguous Float32 tensor to host CPU slice.
func (t *Tensor) ToCPUFloat32() ([]float32, error) {
	if t.dtype != Float32 {
		return nil, errors.New("cugo: ToCPUFloat32 called on non-Float32 tensor")
	}
	if !t.IsContiguous() {
		return nil, ErrNonContiguous
	}

	n := t.Numel()
	res := make([]float32, n)
	rawBytes := unsafe.Slice((*byte)(unsafe.Pointer(&res[0])), n*4)

	if err := t.ctx.CopyDtoH(rawBytes, t.DevicePtr()); err != nil {
		return nil, fmt.Errorf("cugo: CopyDtoH failed: %w", err)
	}
	return res, nil
}

// SetRequiresGrad enables or disables gradient tracking for this tensor.
func (t *Tensor) SetRequiresGrad(r bool) *Tensor {
	t.RequiresGrad = r
	return t
}

// Context returns the underlying CUDA context.
func (t *Tensor) Context() *driver.Context {
	return t.ctx
}

// ZeroGrad resets the tensor's gradient to nil or zero.
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		_ = t.Grad.Close()
		t.Grad = nil
	}
}

// Clone creates a deep copy of the tensor with separate device storage.
func (t *Tensor) Clone() (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, ErrNonContiguous
	}
	cp, err := New(t.ctx, t.dtype, t.shape...)
	if err != nil {
		return nil, err
	}
	byteSize := uint64(t.Numel()) * uint64(t.dtype.Size())
	hMem, err := t.ToCPUFloat32()
	if err != nil {
		_ = cp.Close()
		return nil, err
	}
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&hMem[0])), byteSize)
	if err := t.ctx.CopyHtoD(cp.DevicePtr(), raw); err != nil {
		_ = cp.Close()
		return nil, err
	}
	return cp, nil
}

// AccumulateGrad adds incoming gradient into t.Grad.
func (t *Tensor) AccumulateGrad(g *Tensor) error {
	if t.Grad == nil {
		cloned, err := g.Clone()
		if err != nil {
			return err
		}
		t.Grad = cloned
		return nil
	}

	// Add g to t.Grad on GPU
	n := int32(t.Numel())
	mod, err := t.ctx.LoadModuleData(vecadd.PTX)
	if err != nil {
		return err
	}
	defer mod.Unload()

	fn, err := mod.Function("vecAdd")
	if err != nil {
		return err
	}

	cfg := driver.LaunchConfig{
		GridDimX:  uint32((n + 255) / 256),
		BlockDimX: 256,
	}

	return fn.Launch(cfg, t.Grad.DevicePtr(), g.DevicePtr(), t.Grad.DevicePtr(), n)
}

// Close releases the Tensor's claim on its underlying Storage.
func (t *Tensor) Close() error {
	if t.Grad != nil {
		_ = t.Grad.Close()
		t.Grad = nil
	}
	if t.storage == nil {
		return nil
	}
	err := t.storage.Release()
	t.storage = nil
	return err
}
