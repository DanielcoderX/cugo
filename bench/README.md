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
go test -bench Benchmark -benchtime 500ms ./bench/...
```

Output:
```
goos: windows
goarch: amd64
pkg: github.com/cugo/cugo/bench
cpu: AMD Ryzen 7 7435HS                             
BenchmarkDriverCallOverhead-16       	10020541	        55.86 ns/op	       8 B/op	       1 allocs/op
BenchmarkKernelLaunchLatency-16      	   56131	     10222 ns/op	     256 B/op	      11 allocs/op
BenchmarkCUDAGraphLaunch-16          	   53188	     10081 ns/op	      16 B/op	       1 allocs/op
BenchmarkMemcpyHtoD_1MB-16           	    5166	    114349 ns/op	9169.98 MB/s	      32 B/op	       2 allocs/op
BenchmarkMemcpyHtoD_16MB-16          	     406	   1451610 ns/op	11557.66 MB/s	      32 B/op	       2 allocs/op
BenchmarkMemcpyDtoH_1MB-16           	    3000	    168264 ns/op	6231.73 MB/s	      32 B/op	       2 allocs/op
BenchmarkMemcpyDtoH_16MB-16          	     374	   1569601 ns/op	10688.84 MB/s	      32 B/op	       2 allocs/op
BenchmarkPinnedMemcpyHtoD_16MB-16    	     441	   1314774 ns/op	12760.53 MB/s	      48 B/op	       3 allocs/op
BenchmarkPinnedMemcpyDtoH_16MB-16    	     434	   1345018 ns/op	12473.60 MB/s	      48 B/op	       3 allocs/op
BenchmarkVecAddRoundTrip_100K-16     	    2152	    240290 ns/op	4993.97 MB/s	     352 B/op	      17 allocs/op
PASS
ok  	github.com/cugo/cugo/bench	10.837s
```
