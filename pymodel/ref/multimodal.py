"""OP-13: replace complete placeholder rows in logical batch/sequence order."""

import torch


def visual_feature_scatter(text_embedding: torch.Tensor, visual_features: torch.Tensor,
                            placeholder_rows: torch.Tensor) -> torch.Tensor:
    """Return a new embedding; IF-04 associates each feature with a flat row ID."""
    if text_embedding.ndim != 3 or visual_features.ndim != 2:
        raise ValueError("Expected text [B,S,D] and visual features [N,D]")
    if visual_features.shape[1] != text_embedding.shape[-1]:
        raise ValueError("Text and visual feature widths must match")
    if placeholder_rows.ndim != 1 or len(placeholder_rows) != len(visual_features):
        raise ValueError("One placeholder row is required per visual feature")
    if placeholder_rows.dtype not in (torch.int32, torch.int64):
        raise TypeError("Placeholder rows must be int32 or int64")
    # OP-13：按 row=b*S+s 将文本 [B,S,D] 展为 [B*S,D]，先保留全部文本行。
    flat = text_embedding.reshape(-1, text_embedding.shape[-1]).clone()
    rows = placeholder_rows.to(device=flat.device, dtype=torch.int64)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= flat.shape[0]):
        raise ValueError("Placeholder row outside the text sequence")
    if rows.unique().numel() != rows.numel():
        raise ValueError("Placeholder rows must be unique")
    # out[placeholder_rows[i], :] = visual_features[i, :]；索引唯一，执行覆盖写入。
    # 视觉特征转为文本 dtype，其余行沿用原文本 embedding。
    flat.index_copy_(0, rows, visual_features.to(device=flat.device, dtype=flat.dtype))
    return flat.reshape(text_embedding.shape)
