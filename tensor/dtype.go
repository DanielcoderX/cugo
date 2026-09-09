package tensor

import "fmt"

// DType represents the numeric data type of tensor elements.
type DType uint8

const (
	Float32 DType = iota
	Float64
	Int32
	Int64
	Uint8
)

// Size returns the size in bytes of a single element of this DType.
func (d DType) Size() int {
	switch d {
	case Float32:
		return 4
	case Float64:
		return 8
	case Int32:
		return 4
	case Int64:
		return 8
	case Uint8:
		return 1
	default:
		return 4
	}
}

// String returns the string identifier of the data type.
func (d DType) String() string {
	switch d {
	case Float32:
		return "torch.float32"
	case Float64:
		return "torch.float64"
	case Int32:
		return "torch.int32"
	case Int64:
		return "torch.int64"
	case Uint8:
		return "torch.uint8"
	default:
		return fmt.Sprintf("dtype(%d)", d)
	}
}
