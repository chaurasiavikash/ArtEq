import re

# Read the setup.py file
with open("setup.py", 'r') as f:
    content = f.read()

# Replace any hardcoded CUDA paths
content = re.sub(r'/usr/local/cuda', '/usr', content)
content = re.sub(r'cuda_home = .*', 'cuda_home = "/usr"', content)

# If there's a specific nvcc path detection, fix it
if 'nvcc' in content:
    content = re.sub(r'/usr/local/cuda/bin/nvcc', '/usr/bin/nvcc', content)

with open("setup.py", 'w') as f:
    f.write(content)

print("Fixed CUDA paths in setup.py")
