import re

# Fix the main compatibility issue
file_path = "vgtk/cuda/gathering_cuda.cpp"
with open(file_path, 'r') as f:
    content = f.read()

# Replace the deprecated tensor.type() calls
content = re.sub(r'x\.type\(\)\.is_cuda\(\)', 'x.is_cuda()', content)
content = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\.type\(\)\.is_cuda\(\)', r'\1.is_cuda()', content)

with open(file_path, 'w') as f:
    f.write(content)

print("Applied critical patch")
