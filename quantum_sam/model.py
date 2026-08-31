"""SAM encoder/decoder wrapper with a quantum embedding bottleneck."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .quantum import QuantumBottleneck

class ConvSemanticDecoder(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(), nn.Conv2d(channels, num_classes, 1))
    def forward(self, x, output_size):
        return F.interpolate(self.net(x), size=output_size, mode="bilinear", align_corners=False)

class QuantumSAMSegmenter(nn.Module):
    def __init__(self, num_classes: int, sam_model: str | None = "facebook/sam-vit-base", qubits: int = 8, freeze_sam: bool = True, channels: int = 256):
        super().__init__()
        self.num_classes, self.sam = num_classes, None
        if sam_model:
            from transformers import SamModel
            self.sam = SamModel.from_pretrained(sam_model)
            channels = self.sam.config.vision_config.output_channels
            if freeze_sam:
                for parameter in self.sam.parameters(): parameter.requires_grad = False
        self.quantum = QuantumBottleneck(channels=channels, qubits=qubits)
        self.semantic_head = nn.Conv2d(3, num_classes, 1)
        self.fallback_decoder = ConvSemanticDecoder(channels, num_classes)

    def _sam_masks(self, pixel_values: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        batch = pixel_values.shape[0]
        sparse = image_embeddings.new_zeros((batch, 1, 0, image_embeddings.shape[1]))
        dense = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).to(image_embeddings)
        dense = dense.expand(batch, -1, image_embeddings.shape[-2], image_embeddings.shape[-1])
        outputs = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=self.sam.get_image_wide_positional_embeddings().to(image_embeddings),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True,
        )
        return F.interpolate(outputs.pred_masks.squeeze(1), size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.sam is None:
            return self.fallback_decoder(self.quantum(pixel_values), pixel_values.shape[-2:])
        output_size = pixel_values.shape[-2:]
        encoder_size = self.sam.config.vision_config.image_size
        if pixel_values.shape[-2:] != (encoder_size, encoder_size):
            pixel_values = F.interpolate(pixel_values, size=(encoder_size, encoder_size), mode="bilinear", align_corners=False)
        embeddings = self.quantum(self.sam.get_image_embeddings(pixel_values))
        return F.interpolate(self.semantic_head(self._sam_masks(pixel_values, embeddings)), size=output_size, mode="bilinear", align_corners=False)