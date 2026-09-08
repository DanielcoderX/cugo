import subprocess

vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd = f'call "{vcvars}" -vcvars_ver=14.44.35207 && nvcc -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ptx -o kernels/gemm/gemm.ptx kernels/gemm/gemm.cu'
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
print("RC:", res.returncode)
