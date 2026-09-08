package vecadd

import (
	_ "embed"
)

//go:generate nvcc -ptx -o vecadd.ptx vecadd.cu

// PTX contains the embedded PTX assembly for the vecAdd kernel.
//
//go:embed vecadd.ptx
var PTX []byte
