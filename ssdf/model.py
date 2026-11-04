"""CLIP facial encoding and source-supervised decision fusion."""

import math

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel


class TemporalAttention(nn.Module):
    def __init__(self, width=768, heads=8, head_width=64):
        super().__init__()
        self.heads = heads
        self.head_width = head_width
        self.to_qkv = nn.Linear(width, 3 * heads * head_width, bias=False)
        self.to_out = nn.Linear(heads * head_width, width)

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        q, k, v = self.to_qkv(tokens).chunk(3, dim=-1)
        reshape = lambda x: x.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        weights = (q @ k.transpose(-1, -2) / math.sqrt(self.head_width)).softmax(dim=-1)
        return self.to_out((weights @ v).transpose(1, 2).reshape(batch, length, -1))


class TemporalEncoder(nn.Module):
    def __init__(self, width=768, frames=16):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.pos_embedding = nn.Parameter(torch.randn(1, frames + 1, width))
        self.attention_norm = nn.LayerNorm(width)
        self.attention = TemporalAttention(width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 1024), nn.GELU(approximate="tanh"), nn.Linear(1024, width)
        )

    def forward(self, frames):
        batch = frames.shape[0]
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), frames), dim=1)
        tokens = tokens + self.pos_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return (tokens + self.mlp(self.mlp_norm(tokens)))[:, 0]


class GaussianBottleneck(nn.Module):
    def __init__(self, width, latent_width, scale_floor):
        super().__init__()
        self.scale_floor = scale_floor
        self.layers = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, latent_width), nn.GELU(),
            nn.Linear(latent_width, 2 * latent_width),
        )

    def forward(self, inputs):
        mean, raw_scale = self.layers(inputs).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.scale_floor
        sample = mean + scale * torch.randn_like(mean) if self.training else mean
        kl = 0.5 * (mean.square() + scale.square() - 1 - 2 * scale.log()).sum(dim=-1)
        return sample, kl


class SourceSupervisedHead(nn.Module):
    def __init__(self, classes, latent_width, dropout, scale_floor):
        super().__init__()
        self.visual = GaussianBottleneck(768, latent_width, scale_floor)
        self.context = GaussianBottleneck(256, latent_width, scale_floor)
        self.joint = GaussianBottleneck(2 * latent_width, latent_width, scale_floor)
        self.visual_class = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_width, classes))
        self.context_class = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_width, classes))
        self.joint_class = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_width, classes))

    def forward(self, visual, context):
        with torch.autocast(device_type=visual.device.type, enabled=False):
            visual_latent, visual_kl = self.visual(visual.float())
            context_latent, context_kl = self.context(context.float())
            joint_latent, joint_kl = self.joint(torch.cat((visual_latent, context_latent), dim=-1))
            return {
                "logits": self.joint_class(joint_latent),
                "visual_logits": self.visual_class(visual_latent),
                "context_logits": self.context_class(context_latent),
                "kl": (visual_kl, context_kl, joint_kl),
            }


class SSDF(nn.Module):
    def __init__(self, clip_path, classes, *, lora_rank, lora_alpha, lora_dropout,
                 reader_dropout, classifier_dropout, query_init_std, scale_floor):
        super().__init__()
        clip = CLIPModel.from_pretrained(clip_path)
        clip.requires_grad_(False)
        self.vision = get_peft_model(
            clip.vision_model,
            LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            ),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(2048), nn.Linear(2048, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.queries = nn.Parameter(torch.randn(4, 256) * query_init_std)
        self.query_norm = nn.LayerNorm(256)
        self.patch_norm = nn.LayerNorm(768)
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(768, 256)
        self.v_proj = nn.Linear(768, 256)
        self.reader_out = nn.Linear(1024, 768)
        self.reader_dropout = nn.Dropout(reader_dropout)
        self.frame_norm = nn.LayerNorm(768)
        self.temporal = TemporalEncoder()
        self.classifier = SourceSupervisedHead(classes, 256, classifier_dropout, scale_floor)

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        for name, module in self.vision.named_modules():
            if "lora_dropout" in name:
                module.train(mode)
        return self

    def forward(self, faces, cached_context):
        batch, frames, channels, height, width = faces.shape
        if (frames, channels, height, width) != (16, 3, 224, 224):
            raise ValueError("Expected 16 RGB face frames of size 224 x 224")
        if cached_context.shape != (batch, 2048):
            raise ValueError("Expected one 2048-dimensional context vector per video")

        encoded = self.vision(pixel_values=faces.reshape(-1, 3, 224, 224), return_dict=True)
        patches = encoded.last_hidden_state[:, 1:].reshape(batch * frames, 196, 768)
        cls = encoded.pooler_output.reshape(batch, frames, 768)
        context = self.context(F.normalize(cached_context.float(), dim=-1))

        queries = self.q_proj(self.query_norm(self.queries))
        queries = queries.reshape(4, 4, 64).permute(1, 0, 2)
        queries = queries.unsqueeze(0).expand(batch * frames, -1, -1, -1)
        patches = self.patch_norm(patches)
        keys = self.k_proj(patches).reshape(batch * frames, 196, 4, 64).transpose(1, 2)
        values = self.v_proj(patches).reshape(batch * frames, 196, 4, 64).transpose(1, 2)
        weights = (queries.float() @ keys.float().transpose(-1, -2) / 8).softmax(dim=-1)
        weights = F.dropout(weights, p=self.reader_dropout.p, training=self.training)
        readout = (weights.to(values.dtype) @ values).transpose(1, 2)
        readout = readout.reshape(batch, frames, 1024)
        frame_features = self.frame_norm(cls + self.reader_dropout(self.reader_out(readout)))
        return self.classifier(self.temporal(frame_features), context)
