import re
import os

with open('cuda_errors_raw.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

errors = []
current_comment = ''
val = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    cm = re.search(r'/\*\*\s*(.*?)\s*\*/', line)
    if cm:
        current_comment = cm.group(1).replace('/**', '').replace('*/', '').strip()
    
    m = re.search(r'^(CUDA_\w+)\s*(?:=\s*(\d+))?,?', line)
    if m:
        name = m.group(1)
        if m.group(2) is not None:
            val = int(m.group(2))
        errors.append((name, val, current_comment))
        val += 1
        current_comment = ''

os.makedirs('internal/nvapi', exist_ok=True)

with open('internal/nvapi/errors.go', 'w', encoding='utf-8') as f:
    f.write('//go:build windows\n\n')
    f.write('package nvapi\n\n')
    f.write('import "fmt"\n\n')
    f.write('// CUresult represents CUDA Driver API return status codes.\n')
    f.write('type CUresult uint32\n\n')
    f.write('const (\n')
    for name, v, comment in errors:
        c_str = f' // {comment}' if comment else ''
        f.write(f'\t{name} CUresult = {v}{c_str}\n')
    f.write(')\n\n')
    
    f.write('// Error implements the error interface for CUresult.\n')
    f.write('func (r CUresult) Error() string {\n')
    f.write('\tswitch r {\n')
    for name, v, comment in errors:
        desc = comment if comment else name
        desc_escaped = desc.replace('"', '\\"')
        f.write(f'\tcase {name}:\n\t\treturn "{name}: {desc_escaped}"\n')
    f.write('\tdefault:\n\t\treturn fmt.Sprintf("CUDA_ERROR_UNKNOWN (%d)", uint32(r))\n')
    f.write('\t}\n')
    f.write('}\n\n')
    
    f.write('// ResultToError converts CUresult to Go error. Returns nil on CUDA_SUCCESS.\n')
    f.write('func ResultToError(res CUresult) error {\n')
    f.write('\tif res == CUDA_SUCCESS {\n\t\treturn nil\n\t}\n')
    f.write('\treturn res\n')
    f.write('}\n')

print(f'Wrote internal/nvapi/errors.go with {len(errors)} error codes')
