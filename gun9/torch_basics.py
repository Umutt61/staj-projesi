import torch

# Cihaz seçimi (GPU varsa onu kullanmalı)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")

# 2B rastgele tensör oluşturma
a = torch.rand(2, 3, device=device)
b = torch.rand(2, 3, device=device)
print("Tensör A:\n", a)
print("Tensör B:\n", b)

# Element-wise toplama
toplam = a + b
print("Toplam:\n", toplam)

# Matris çarpımı
a_reshaped = torch.rand(2, 3, device=device)
b_reshaped = torch.rand(3, 2, device=device)
matmul = torch.matmul(a_reshaped, b_reshaped)
print("Matris Çarpımı:\n", matmul)

# Element-wise fonksiyon: exp
exp_tensor = torch.exp(a)
print("Exp(A):\n", exp_tensor)

exp_tensor_b = torch.exp(b)
print("Exp(B):\n", exp_tensor_b)

# Otomatik türev
x = torch.tensor([[2.0, 3.0]], requires_grad=True, device=device)
y = x.pow(2).sum()
y.backward()
print("Türev (∂y/∂x):\n", x.grad)