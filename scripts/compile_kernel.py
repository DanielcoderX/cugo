import sys
import subprocess

vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cu_file = sys.argv[1] if len(sys.argv) > 1 else "kernels/reduction/reduction.cu"
ptx_file = sys.argv[2] if len(sys.argv) > 2 else "kernels/reduction/reduction.ptx"

cmd = f'call "{vcvars}" -vcvars_ver=14.44.35207 && nvcc -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ptx -o {ptx_file} {cu_file}'
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("RC:", res.returncode)
if res.returncode != 0:
    sys.exit(res.returncode)
