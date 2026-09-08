# cugo: pure-Go CUDA Driver API bindings (no cgo)

> **Status**: Experimental / Pre-1.0 (Windows-only for v1)

`cugo` exposes the NVIDIA CUDA Driver API directly to Go **without cgo** by dynamically loading `nvcuda.dll` at runtime via `golang.org/x/sys/windows`. While existing Go CUDA bindings require a C compiler, CUDA headers, and the CUDA Toolkit toolchain installed at build time, `cugo` requires **no C compiler to build or ship a Go program that uses the GPU** — only the standard NVIDIA display driver needs to be present on the target host at runtime. This makes cross-compiling GPU-accelerated Go binaries effortless.

## Feature Comparison

| Feature | `cugo` | `gorgonia/cu` |
|---|---|---|
| **CGO Required** | **No** (`CGO_ENABLED=0` friendly) | Yes |
| **Build-Time C Toolchain** | **None** | MSVC / GCC / Clang required |
| **CUDA Toolkit at Build Time** | **None** (for library consumers) | Required (`cuda.h`, import libs) |
| **Cross-Compilation** | Seamless from any OS/architecture | Difficult (requires cross-toolchain) |
| **Driver Dependency** | Runtime `nvcuda.dll` | Runtime + link-time driver libraries |
| **Platform Support** | Windows (`amd64`, v1) | Windows, Linux |
| **API Coverage** | Focused driver subset ([Coverage](docs/driver-api-coverage.md)) | Broad driver API coverage |
| **Maturity** | New / Experimental | Mature |

## API Coverage

See [docs/driver-api-coverage.md](docs/driver-api-coverage.md) for the current status of bound CUDA Driver API functions.

## Installation

```bash
go get github.com/cugo/cugo
```

Requires Go 1.22+ and an NVIDIA GPU with drivers installed.

## Quickstart: Device Enumeration

```go
package main

import (
	"fmt"
	"log"

	"github.com/cugo/cugo/driver"
)

func main() {
	if err := driver.Init(); err != nil {
		log.Fatalf("driver.Init failed: %v", err)
	}

	version, err := driver.DriverVersion()
	if err != nil {
		log.Fatalf("driver.DriverVersion: %v", err)
	}
	fmt.Printf("CUDA Driver Version: %d.%d\n", version/1000, (version%100)/10)

	devices, err := driver.Devices()
	if err != nil {
		log.Fatalf("driver.Devices: %v", err)
	}

	for i, dev := range devices {
		name, _ := dev.Name()
		major, minor, _ := dev.ComputeCapability()
		totalMem, _ := dev.TotalMemory()
		fmt.Printf("[%d] %s (Compute %d.%d, Memory: %.2f GiB)\n",
			i, name, major, minor, float64(totalMem)/(1024*1024*1024))
	}
}
```

Run the included example:

```bash
go run ./examples/device-info
```

## Architecture & Design

- [docs/design.md](docs/design.md): System architecture, calling conventions, error mapping.
- [docs/decisions.md](docs/decisions.md): Architecture Decision Records (ADRs).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
