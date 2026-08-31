"""Small differentiable variational circuit used only on pooled SAM embeddings."""
from __future__ import annotations
import pennylane as qml
import torch
from torch import nn

class QuantumBottleneck(nn.Module):
    def __init__(self, channels: int = 256, qubits: int = 8, layers: int = 2):
        super().__init__()
        if not 1 <= qubits <= 16: raise ValueError("qubits must be in [1, 16]")
        self.to_angles = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, qubits), nn.Tanh())
        self.weights = nn.Parameter(0.01 * torch.randn(layers, qubits, 3))
        device = qml.device("default.qubit", wires=qubits)
        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            qml.AngleEmbedding(x * torch.pi, wires=range(qubits), rotation="Y")
            for layer in range(layers):
                for wire in range(qubits): qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(qubits): qml.CNOT(wires=[wire, (wire + 1) % qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]
        self.circuit = circuit
        self.from_quantum = nn.Sequential(nn.Linear(qubits, channels), nn.GELU(), nn.Linear(channels, channels))
        self.norm = nn.GroupNorm(8, channels)
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        angles = self.to_angles(embedding)
        # ``default.qubit`` is a CPU simulator. Move both circuit inputs and
        # circuit parameters to its device, then return its tiny descriptor to
        # CUDA. ``Tensor.to`` remains differentiable, so gradients flow back to
        # the original GPU parameters through the copies.
        cpu_angles = angles.to(device="cpu")
        cpu_weights = self.weights.to(device="cpu")
        features = torch.stack([self.circuit(x, cpu_weights) for x in cpu_angles])
        features = features.to(device=embedding.device, dtype=embedding.dtype)
        return self.norm(embedding + self.from_quantum(features).unsqueeze(-1).unsqueeze(-1))