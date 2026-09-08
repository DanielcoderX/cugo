import re

with open('scripts/attrs.txt', 'r') as f:
    lines = f.readlines()

attrs = []
for line in lines:
    line = line.strip()
    if not line: continue
    name, val = line.split(' = ')
    attrs.append((name, int(val)))

with open('internal/nvapi/types.go', 'w', encoding='utf-8') as f:
    f.write('//go:build windows\n\n')
    f.write('package nvapi\n\n')
    f.write('// Core handle types matching cuda.h exactly.\n')
    f.write('type CUdevice int32\n')
    f.write('type CUdeviceptr uintptr\n')
    f.write('type CUcontext uintptr\n')
    f.write('type CUmodule uintptr\n')
    f.write('type CUfunction uintptr\n')
    f.write('type CUstream uintptr\n')
    f.write('type CUevent uintptr\n\n')
    
    f.write('// CUdevice_attribute represents hardware properties queryable on CUdevice.\n')
    f.write('type CUdevice_attribute int32\n\n')
    f.write('const (\n')
    for n, v in attrs:
        f.write(f'\t{n} CUdevice_attribute = {v}\n')
    f.write(')\n')

print('Generated internal/nvapi/types.go successfully')
