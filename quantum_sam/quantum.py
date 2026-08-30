"""Small differentiable variational circuit used only on pooled SAM embeddings."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn


class QuantumBottleneck(nn.Module):
    """Projects pooled features through an angle-encoded <=16-qubit circuit.

    A residual broadcast puts the learned global quantum descriptor back into the
    spatial SAM embedding, avoiding a per-pixel circuit evaluation.
    """

    def __init__(self, channels: int = 256, qubits: int = 8, layers: int = 2):
        super().__init__()
        if not 1 <= qubits <= 16:
            raise ValueError("qubits must be in [1, 16]")
        self.qubits = qubits
        self.to_angles = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, qubits), nn.Tanh())
        self.weights = nn.Parameter(0.01 * torch.randn(layers, qubits, 3))
        self.device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            qml.AngleEmbedding(x * torch.pi, wires=range(qubits), rotation="Y")
            for layer in range(layers):
                for wire in range(qubits):
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(qubits):
                    qml.CNOT(wires=[wire, (wire + 1) % qubits])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(qubits)]

        self.circuit = circuit
        self.from_quantum = nn.Sequential(nn.Linear(qubits, channels), nn.GELU(), nn.Linear(channels, channels))
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        angles = self.to_angles(embedding)
        # PennyLane's qnode consumes one feature vector; the circuit is tiny and
        # batch iteration keeps its behavior stable across supported versions.
        quantum_features = torch.stack([self.circuit(x, self.weights) for x in angles]).to(embedding.dtype)
        residual = self.from_quantum(quantum_features).unsqueeze(-1).unsqueeze(-1)
        return self.norm(embedding + residual)
