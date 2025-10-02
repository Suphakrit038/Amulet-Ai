import torch
import torch.nn as nn

# Test head with 2D input (what ResNet gives)
x_2d = torch.randn(1, 2048)
print(f"Input 2D: {x_2d.shape}")

head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(2048, 3)
)

try:
    out = head(x_2d)
    print(f"Output: {out.shape}")
except Exception as e:
    print(f"ERROR with 2D input: {e}")
    
# Now test with 4D
x_4d = x_2d.unsqueeze(-1).unsqueeze(-1)
print(f"\nInput 4D: {x_4d.shape}")

try:
    out = head(x_4d)
    print(f"Output: {out.shape}")
except Exception as e:
    print(f"ERROR with 4D input: {e}")
