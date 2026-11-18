import torch


def recall_at_k(scores, labels, k):
    _, topk_indices = torch.topk(scores, k, dim=1)
    labels_expanded = labels.unsqueeze(1).expand_as(topk_indices)
    is_hit = topk_indices == labels_expanded
    hits = is_hit.sum(dim=1) > 0
    return hits.float().mean().item()


def hit_rate_at_k(scores, labels, k):
    return recall_at_k(scores, labels, k)


def ndcg_at_k(scores, labels, k):
    _, topk_indices = torch.topk(scores, k, dim=1)
    is_hit = topk_indices == labels.unsqueeze(1).expand_as(topk_indices)
    rank = torch.arange(1, k + 1, dtype=torch.float, device=scores.device)
    denominator = torch.log2(rank + 1)
    dcg_weights = (1.0 / denominator).unsqueeze(0).expand_as(is_hit)
    dcg = (is_hit.float() * dcg_weights).sum(dim=1)
    return dcg.mean().item()


def mrr_at_k(scores, labels, k):
    _, topk_indices = torch.topk(scores, k, dim=1)
    is_hit = topk_indices == labels.unsqueeze(1).expand_as(topk_indices)
    ranks = torch.arange(1, k + 1, dtype=torch.float, device=scores.device)
    ranks_expanded = ranks.unsqueeze(0).expand_as(is_hit)
    hit_ranks = is_hit.float() * ranks_expanded
    first_rank = hit_ranks.max(dim=1)[0]
    reciprocal_rank = torch.where(first_rank > 0, 1.0 / first_rank, torch.zeros_like(first_rank))
    return reciprocal_rank.mean().item()
