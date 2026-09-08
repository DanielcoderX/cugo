//go:build windows || linux

package nvapi

import "unsafe"

// Core handle types matching cuda.h exactly.
type CUdevice int32
type CUdeviceptr uintptr
type CUcontext uintptr
type CUmodule uintptr
type CUfunction uintptr
type CUstream uintptr
type CUevent uintptr
type CUgraph uintptr
type CUgraphExec uintptr
type CUmemoryPool uintptr
type CUlinkState uintptr
type CUarray uintptr
type CUtexObject uint64
type CUsurfObject uint64

// CUjitInputType specifies device-code input types to the JIT linker.
type CUjitInputType int32

const (
	CU_JIT_INPUT_CUBIN     CUjitInputType = 0
	CU_JIT_INPUT_PTX       CUjitInputType = 1
	CU_JIT_INPUT_FATBINARY CUjitInputType = 2
	CU_JIT_INPUT_OBJECT    CUjitInputType = 3
	CU_JIT_INPUT_LIBRARY   CUjitInputType = 4
)

// CUjit_option specifies online compiler/linker options.
type CUjit_option int32

const (
	CU_JIT_MAX_REGISTERS               CUjit_option = 0
	CU_JIT_THREADS_PER_BLOCK           CUjit_option = 1
	CU_JIT_WALL_TIME                   CUjit_option = 2
	CU_JIT_INFO_LOG_BUFFER             CUjit_option = 3
	CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES  CUjit_option = 4
	CU_JIT_ERROR_LOG_BUFFER            CUjit_option = 5
	CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES CUjit_option = 6
	CU_JIT_OPTIMIZATION_LEVEL          CUjit_option = 7
	CU_JIT_TARGET_FROM_CUCONTEXT       CUjit_option = 8
	CU_JIT_TARGET                      CUjit_option = 9
	CU_JIT_FALLBACK_STRATEGY           CUjit_option = 10
	CU_JIT_GENERATE_DEBUG_INFO         CUjit_option = 11
	CU_JIT_LOG_VERBOSE                 CUjit_option = 12
	CU_JIT_GENERATE_LINE_INFO          CUjit_option = 13
	CU_JIT_CACHE_MODE                  CUjit_option = 14
)

// CUmemorytype represents memory types for 2D copies.
type CUmemorytype int32

const (
	CU_MEMORYTYPE_HOST    CUmemorytype = 0x01
	CU_MEMORYTYPE_DEVICE  CUmemorytype = 0x02
	CU_MEMORYTYPE_ARRAY   CUmemorytype = 0x03
	CU_MEMORYTYPE_UNIFIED CUmemorytype = 0x04
)

// CUarray_format represents array element format types.
type CUarray_format int32

const (
	CU_AD_FORMAT_UNSIGNED_INT8  CUarray_format = 0x01
	CU_AD_FORMAT_UNSIGNED_INT16 CUarray_format = 0x02
	CU_AD_FORMAT_UNSIGNED_INT32 CUarray_format = 0x03
	CU_AD_FORMAT_SIGNED_INT8    CUarray_format = 0x08
	CU_AD_FORMAT_SIGNED_INT16   CUarray_format = 0x09
	CU_AD_FORMAT_SIGNED_INT32   CUarray_format = 0x0a
	CU_AD_FORMAT_HALF           CUarray_format = 0x10
	CU_AD_FORMAT_FLOAT          CUarray_format = 0x20
)

// CUDA_MEMCPY2D contains parameters for 2D memory copies matching CUDA C ABI exactly.
type CUDA_MEMCPY2D struct {
	SrcXInBytes   uint64
	SrcY          uint64
	SrcMemoryType CUmemorytype
	_             uint32 // explicit padding for 8-byte pointer alignment
	SrcHost       unsafe.Pointer
	SrcDevice     CUdeviceptr
	SrcArray      CUarray
	SrcPitch      uint64

	DstXInBytes   uint64
	DstY          uint64
	DstMemoryType CUmemorytype
	_             uint32 // explicit padding for 8-byte pointer alignment
	DstHost       unsafe.Pointer
	DstDevice     CUdeviceptr
	DstArray      CUarray
	DstPitch      uint64

	WidthInBytes uint64
	Height       uint64
}

// CUDA_ARRAY_DESCRIPTOR contains parameters for allocating CUDA 2D arrays.
type CUDA_ARRAY_DESCRIPTOR struct {
	Width       uint64
	Height      uint64
	Format      CUarray_format
	NumChannels uint32
}

// CUresourcetype specifies resource type for texture/surface descriptors.
type CUresourcetype int32

const (
	CU_RESOURCE_TYPE_ARRAY           CUresourcetype = 0x00
	CU_RESOURCE_TYPE_MIPMAPPED_ARRAY CUresourcetype = 0x01
	CU_RESOURCE_TYPE_LINEAR          CUresourcetype = 0x02
	CU_RESOURCE_TYPE_PITCH2D         CUresourcetype = 0x03
)

// CUaddress_mode specifies texture addressing modes.
type CUaddress_mode int32

const (
	CU_TR_ADDRESS_MODE_WRAP   CUaddress_mode = 0
	CU_TR_ADDRESS_MODE_CLAMP  CUaddress_mode = 1
	CU_TR_ADDRESS_MODE_MIRROR CUaddress_mode = 2
	CU_TR_ADDRESS_MODE_BORDER CUaddress_mode = 3
)

// CUfilter_mode specifies texture filtering modes.
type CUfilter_mode int32

const (
	CU_TR_FILTER_MODE_POINT  CUfilter_mode = 0
	CU_TR_FILTER_MODE_LINEAR CUfilter_mode = 1
)

// CUDA_RESOURCE_DESC (144 bytes) contains resource binding parameters for texture and surface objects.
type CUDA_RESOURCE_DESC struct {
	ResType CUresourcetype
	_       uint32 // padding
	ResData [16]uint64
	Flags   uint32
	_       uint32 // padding
}

// CUDA_TEXTURE_DESC (104 bytes) contains sampling configuration for texture objects.
type CUDA_TEXTURE_DESC struct {
	AddressMode         [3]CUaddress_mode
	FilterMode          CUfilter_mode
	Flags               uint32
	MaxAnisotropy       uint32
	MipmapFilterMode    CUfilter_mode
	MipmapLevelBias     float32
	MinMipmapLevelClamp float32
	MaxMipmapLevelClamp float32
	BorderColor         [4]float32
	Reserved            [12]int32
}

// CU_DEVICE_CPU indicates CPU device target for prefetching.
const CU_DEVICE_CPU CUdevice = -1

// Unified Memory Attach flags.
const (
	CU_MEM_ATTACH_GLOBAL uint32 = 0x1
	CU_MEM_ATTACH_HOST   uint32 = 0x2
	CU_MEM_ATTACH_SINGLE uint32 = 0x4
)

// CUmem_advise represents memory advising hints for Unified Memory.
type CUmem_advise int32

const (
	CU_MEM_ADVISE_SET_READ_MOSTLY          CUmem_advise = 1
	CU_MEM_ADVISE_UNSET_READ_MOSTLY        CUmem_advise = 2
	CU_MEM_ADVISE_SET_PREFERRED_LOCATION   CUmem_advise = 3
	CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION CUmem_advise = 4
	CU_MEM_ADVISE_SET_ACCESSED_BY          CUmem_advise = 5
	CU_MEM_ADVISE_UNSET_ACCESSED_BY        CUmem_advise = 6
)

// CUdevice_P2PAttribute represents peer-to-peer link attributes between two devices.
type CUdevice_P2PAttribute int32

const (
	CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK                     CUdevice_P2PAttribute = 0x01
	CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED                     CUdevice_P2PAttribute = 0x02
	CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED              CUdevice_P2PAttribute = 0x03
	CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED        CUdevice_P2PAttribute = 0x04
	CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED          CUdevice_P2PAttribute = 0x04
)

// CUstreamCaptureMode specifies how work dispatched during stream capture is tracked.
type CUstreamCaptureMode int32

const (
	CU_STREAM_CAPTURE_MODE_GLOBAL       CUstreamCaptureMode = 0
	CU_STREAM_CAPTURE_MODE_THREAD_LOCAL CUstreamCaptureMode = 1
	CU_STREAM_CAPTURE_MODE_RELAXED      CUstreamCaptureMode = 2
)

// CUstreamCaptureStatus represents the capture status of a stream.
type CUstreamCaptureStatus int32

const (
	CU_STREAM_CAPTURE_STATUS_NONE        CUstreamCaptureStatus = 0
	CU_STREAM_CAPTURE_STATUS_ACTIVE      CUstreamCaptureStatus = 1
	CU_STREAM_CAPTURE_STATUS_INVALIDATED CUstreamCaptureStatus = 2
)

// CUmemPool_attribute specifies attributes for a memory pool.
type CUmemPool_attribute int32

const (
	CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES   CUmemPool_attribute = 1
	CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC         CUmemPool_attribute = 2
	CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES CUmemPool_attribute = 3
	CU_MEMPOOL_ATTR_RELEASE_THRESHOLD                 CUmemPool_attribute = 4
	CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT              CUmemPool_attribute = 5
	CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH                 CUmemPool_attribute = 6
	CU_MEMPOOL_ATTR_USED_MEM_CURRENT                  CUmemPool_attribute = 7
	CU_MEMPOOL_ATTR_USED_MEM_HIGH                     CUmemPool_attribute = 8
)

// CUdevice_attribute represents hardware properties queryable on CUdevice.
type CUdevice_attribute int32

const (
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK CUdevice_attribute = 1
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X CUdevice_attribute = 2
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y CUdevice_attribute = 3
	CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z CUdevice_attribute = 4
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X CUdevice_attribute = 5
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y CUdevice_attribute = 6
	CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z CUdevice_attribute = 7
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK CUdevice_attribute = 8
	CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK CUdevice_attribute = 8
	CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY CUdevice_attribute = 9
	CU_DEVICE_ATTRIBUTE_WARP_SIZE CUdevice_attribute = 10
	CU_DEVICE_ATTRIBUTE_MAX_PITCH CUdevice_attribute = 11
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK CUdevice_attribute = 12
	CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK CUdevice_attribute = 12
	CU_DEVICE_ATTRIBUTE_CLOCK_RATE CUdevice_attribute = 13
	CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT CUdevice_attribute = 14
	CU_DEVICE_ATTRIBUTE_GPU_OVERLAP CUdevice_attribute = 15
	CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT CUdevice_attribute = 16
	CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT CUdevice_attribute = 17
	CU_DEVICE_ATTRIBUTE_INTEGRATED CUdevice_attribute = 18
	CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY CUdevice_attribute = 19
	CU_DEVICE_ATTRIBUTE_COMPUTE_MODE CUdevice_attribute = 20
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH CUdevice_attribute = 21
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH CUdevice_attribute = 22
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT CUdevice_attribute = 23
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH CUdevice_attribute = 24
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT CUdevice_attribute = 25
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH CUdevice_attribute = 26
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH CUdevice_attribute = 27
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT CUdevice_attribute = 28
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS CUdevice_attribute = 29
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH CUdevice_attribute = 27
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT CUdevice_attribute = 28
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES CUdevice_attribute = 29
	CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT CUdevice_attribute = 30
	CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS CUdevice_attribute = 31
	CU_DEVICE_ATTRIBUTE_ECC_ENABLED CUdevice_attribute = 32
	CU_DEVICE_ATTRIBUTE_PCI_BUS_ID CUdevice_attribute = 33
	CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID CUdevice_attribute = 34
	CU_DEVICE_ATTRIBUTE_TCC_DRIVER CUdevice_attribute = 35
	CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE CUdevice_attribute = 36
	CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH CUdevice_attribute = 37
	CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE CUdevice_attribute = 38
	CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR CUdevice_attribute = 39
	CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT CUdevice_attribute = 40
	CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING CUdevice_attribute = 41
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH CUdevice_attribute = 42
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS CUdevice_attribute = 43
	CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER CUdevice_attribute = 44
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH CUdevice_attribute = 45
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT CUdevice_attribute = 46
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE CUdevice_attribute = 47
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE CUdevice_attribute = 48
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE CUdevice_attribute = 49
	CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID CUdevice_attribute = 50
	CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT CUdevice_attribute = 51
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH CUdevice_attribute = 52
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH CUdevice_attribute = 53
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS CUdevice_attribute = 54
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH CUdevice_attribute = 55
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH CUdevice_attribute = 56
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT CUdevice_attribute = 57
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH CUdevice_attribute = 58
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT CUdevice_attribute = 59
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH CUdevice_attribute = 60
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH CUdevice_attribute = 61
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS CUdevice_attribute = 62
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH CUdevice_attribute = 63
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT CUdevice_attribute = 64
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS CUdevice_attribute = 65
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH CUdevice_attribute = 66
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH CUdevice_attribute = 67
	CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS CUdevice_attribute = 68
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH CUdevice_attribute = 69
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH CUdevice_attribute = 70
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT CUdevice_attribute = 71
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH CUdevice_attribute = 72
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH CUdevice_attribute = 73
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT CUdevice_attribute = 74
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR CUdevice_attribute = 75
	CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR CUdevice_attribute = 76
	CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH CUdevice_attribute = 77
	CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED CUdevice_attribute = 78
	CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED CUdevice_attribute = 79
	CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED CUdevice_attribute = 80
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR CUdevice_attribute = 81
	CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR CUdevice_attribute = 82
	CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY CUdevice_attribute = 83
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD CUdevice_attribute = 84
	CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID CUdevice_attribute = 85
	CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED CUdevice_attribute = 86
	CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO CUdevice_attribute = 87
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS CUdevice_attribute = 88
	CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS CUdevice_attribute = 89
	CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED CUdevice_attribute = 90
	CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM CUdevice_attribute = 91
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 CUdevice_attribute = 92
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 CUdevice_attribute = 93
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 CUdevice_attribute = 94
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH CUdevice_attribute = 95
	CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH CUdevice_attribute = 96
	CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN CUdevice_attribute = 97
	CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES CUdevice_attribute = 98
	CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED CUdevice_attribute = 99
	CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES CUdevice_attribute = 100
	CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST CUdevice_attribute = 101
	CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED CUdevice_attribute = 102
	CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CUdevice_attribute = 102
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED CUdevice_attribute = 103
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED CUdevice_attribute = 104
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED CUdevice_attribute = 105
	CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR CUdevice_attribute = 106
	CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED CUdevice_attribute = 107
	CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE CUdevice_attribute = 108
	CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE CUdevice_attribute = 109
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED CUdevice_attribute = 110
	CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK CUdevice_attribute = 111
	CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED CUdevice_attribute = 112
	CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED CUdevice_attribute = 113
	CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED CUdevice_attribute = 114
	CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED CUdevice_attribute = 115
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED CUdevice_attribute = 116
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS CUdevice_attribute = 117
	CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING CUdevice_attribute = 118
	CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES CUdevice_attribute = 119
	CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH CUdevice_attribute = 120
	CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED CUdevice_attribute = 121
	CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS CUdevice_attribute = 122
	CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR CUdevice_attribute = 123
	CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED CUdevice_attribute = 124
	CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED CUdevice_attribute = 125
	CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT CUdevice_attribute = 126
	CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED CUdevice_attribute = 127
	CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED CUdevice_attribute = 128
	CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS CUdevice_attribute = 129
	CU_DEVICE_ATTRIBUTE_NUMA_CONFIG CUdevice_attribute = 130
	CU_DEVICE_ATTRIBUTE_NUMA_ID CUdevice_attribute = 131
	CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED CUdevice_attribute = 132
	CU_DEVICE_ATTRIBUTE_MPS_ENABLED CUdevice_attribute = 133
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID CUdevice_attribute = 134
	CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED CUdevice_attribute = 135
	CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK CUdevice_attribute = 136
	CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH CUdevice_attribute = 137
	CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED CUdevice_attribute = 138
	CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID CUdevice_attribute = 139
	CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID CUdevice_attribute = 140
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CUdevice_attribute = 141
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED CUdevice_attribute = 142
	CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED CUdevice_attribute = 143
	CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED CUdevice_attribute = 144
	CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CUdevice_attribute = 145
	CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED CUdevice_attribute = 146
	CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED CUdevice_attribute = 147
	CU_DEVICE_ATTRIBUTE_MAX CUdevice_attribute = 148
)
