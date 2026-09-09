# cugo Benchmarks

Measured on physical hardware using Windows host.

## Environment

```
$ go env GOVERSION GOOS GOARCH
go1.26.0
windows
amd64

$ nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 4060 Laptop GPU, 616.64, 8188 MiB
```

## Raw Benchmark Output

Command:
```
go test -bench Benchmark ./bench/...
```

Output:
```
goos: windows
goarch: amd64
pkg: github.com/cugo/cugo/bench
cpu: AMD Ryzen 7 7435HS                             
BenchmarkDriverCallOverhead-16      	19850984	        56.39 ns/op	       8 B/op	       1 allocs/op
BenchmarkVecAddRoundTrip_100K-16    	    4797	    235236 ns/op	5101.26 MB/s	     352 B/op	      17 allocs/op
PASS
ok  	github.com/cugo/cugo/bench	3.049s
```
