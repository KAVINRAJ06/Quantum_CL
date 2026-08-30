"""SAM encoder/decoder wrapper with a quantum embedding bottleneck."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .quantum import QuantumBottleneck


class ConvSemanticDecoder(nn.Module):
    """Fallback decoder for smoke tests; production uses SAM's mask decoder."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, num_classes, 1))
    def forward(self, x, output_size):
        return F.interpolate(self.net(x), size=output_size, mode="bilinear", align_corners=False)


class QuantumSAMSegmenter(nn.Module):
    """Semantic model using SAM's image encoder *and* mask decoder.

    SAM emits its native three mask hypotheses. A lightweight 1x1 semantic head
    maps those hypotheses to land-cover classes. The quantum module transforms
    image embeddings between the SAM encoder and decoder.
    """
    def __init__(self, num_classes: int, sam_model: str | None = "facebook/sam-vit-base", qubits: int = 8, freeze_sam: bool = True, channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.sam = None
        if sam_model:
            from transformers import SamModel
            self.sam = SamModel.from_pretrained(sam_model)
            channels = self.sam.config.vision_config.output_channels
            if freeze_sam:
                for parameter in self.sam.parameters():
                    parameter.requires_grad = False
        self.quantum = QuantumBottleneck(channels=channels, qubits=qubits)
        self.semantic_head = nn.Conv2d(3, num_classes, kernel_size=1)
        self.fallback_decoder = ConvSemanticDecoder(channels, num_classes)

    def _sam_masks(self, pixel_values: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        # The prompt encoder's learned no-mask embedding represents an empty prompt.
        batch = pixel_values.shape[0]
        sparse = image_embeddings.new_zeros((batch, 1, 0, image_embeddings.shape[1]))
        dense = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        dense = dense.expand(batch, -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        outputs = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=self.sam.get_image_wide_positional_embeddings(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            output_attentions=False,
        )
        masks = outputs.pred_masks.squeeze(1)  # [B, 3, 256, 256]
        return F.interpolate(masks, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.sam is None:
            return self.fallback_decoder(self.quantum(pixel_values), pixel_values.shape[-2:])
        image_embeddings = self.sam.get_image_embeddings(pixel_values)
        image_embeddings = self.quantum(image_embeddings)
        native_masks = self._sam_masks(pixel_values, image_embeddings)
        return self.semantic_head(native_masks)
