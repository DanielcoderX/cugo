package gemm

import _ "embed"

//go:embed gemm.ptx
var PTX []byte
