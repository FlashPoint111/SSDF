"""Frozen Qwen3-VL-Embedding context extraction for prepared observations."""

import numpy as np
import torch
from torch.nn import functional as F


class FrozenQwenContext:
    """Use an externally supplied Qwen3-VL-Embedding-2B instance."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.model = embedder.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, ordered_observations, reference_face, metadata, instruction, task_text):
        """Build one multimodal input and return its normalized context vector."""
        count = len(ordered_observations)
        if count < 2 or count % 2:
            raise ValueError("An even, nonempty observation sequence is required")
        timestamps = np.asarray(metadata["relative_pts_seconds"], dtype=np.float64)
        frame_indices = list(metadata["frame_indices"])
        if timestamps.shape != (count,) or len(frame_indices) != count:
            raise ValueError("Observation timestamps and source indices must align")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError("Observation timestamps must be finite and ordered")

        message = self.embedder.format_model_input(
            video=ordered_observations,
            image=reference_face,
            text=task_text,
            instruction=instruction,
        )
        processor = self.embedder.processor
        prompt = processor.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
        video_pixels = torch.from_numpy(
            np.stack([np.asarray(frame.convert("RGB")) for frame in ordered_observations])
        ).permute(0, 3, 1, 2)

        original_timestamps = processor._calculate_timestamps

        def source_timestamps(indices, fps, merge_size):
            if list(indices) != frame_indices or merge_size != 2:
                raise ValueError("The processor changed the source frame order")
            return [float((timestamps[i] + timestamps[i + 1]) / 2) for i in range(0, count, 2)]

        processor._calculate_timestamps = source_timestamps
        try:
            model_inputs = processor(
                text=prompt,
                images=[reference_face],
                videos=[video_pixels],
                video_metadata=[{
                    "fps": metadata["fps"],
                    "total_num_frames": metadata["total_num_frames"],
                    "frames_indices": frame_indices,
                    "duration": metadata["duration_seconds"],
                }],
                do_resize=False,
                do_sample_frames=False,
                truncation=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            processor._calculate_timestamps = original_timestamps

        device = next(self.model.parameters()).device
        model_inputs = {name: value.to(device) for name, value in model_inputs.items()}
        output = self.model(
            **model_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        hidden = output.last_hidden_state.float()
        mask = model_inputs["attention_mask"].bool()
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last_valid = positions.masked_fill(~mask, -1).max(dim=1).values
        if (last_valid < 0).any():
            raise ValueError("The multimodal input has no valid token")
        vectors = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_valid]
        if vectors.shape[-1] != 2048:
            raise ValueError("Unexpected context vector width")
        return F.normalize(vectors, dim=-1)
