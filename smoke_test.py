import torch
from quantum_sam import QuantumSAMSegmenter

torch.manual_seed(7)
model=QuantumSAMSegmenter(num_classes=8,sam_model=None,qubits=4,channels=32)
logits=model(torch.rand(2,32,16,16)); loss=torch.nn.functional.cross_entropy(logits,torch.randint(0,8,(2,16,16)))
loss.backward()
assert logits.shape == (2,8,16,16) and model.quantum.weights.grad is not None
print('PASS',tuple(logits.shape),'quantum_grad_norm=',model.quantum.weights.grad.norm().item())
