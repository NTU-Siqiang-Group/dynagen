import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def weight_bias_concat(weight, bias, scaling=False, head_dim=1.0):
    """Concatenates the weight matrix and bias.

    On the warmup phase, concatenates the weight matrix and bias for skewing.
    This manipulation does not hurt the correctness.

    Args:
        weight: Weight matrix (D, D)
        bias: Bias vector (D)
        scaling: If ture, scales the concatenated weight and bias to skip
            the scaling after projection.
        head_dim: Hidden dimension of each head which we refer to as d

    Returns:
        concatenated weight and bias (D, D+1)
    """

    weight, bias = weight.data, bias.data

    if scaling:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1) * (head_dim**-0.5)
    else:
        return torch.cat((weight, bias.unsqueeze(1).to(weight.device)), dim=1)


def reform_hidden_states(hidden_states):
    """Concatenates the weight matrix and bias.

    Concatenates the hidden states with a column of 1.
    This reformation with the concatenated weight and bias  makes the linear
    projection into a one matrix multiplication without bias addition.

    Args:
        hidden: Hidden states (b, n, D)

    Returns:
        reformed hidden states (b, n, D+1)
    """

    return torch.cat((hidden_states, torch.ones_like(hidden_states)[:, :, 1].unsqueeze(2)), dim=-1)


def skew(query, key, wq, wk, n_head, head_dim, n_kv_groups=None):
    """Manipulates the query/key weight matrix for skewing the query and key matrix.

    On the warmup phase, manipulates the query/key weight matrix for
    skewing the query and key matrix. By doing so, a few columns of
    the query and key matrix have become much more important. We use
    the columns for attention speculation.

    Args:
        query: Query matrix (b, n, h, d)
        key: Key matrix (b, n, h / n_kv_groups, d)
        w_q: query weight (D, D)
        w_k: key weight (D / n_kv_groups, D)
        n_head: Number of heads which we refer to as h
        head_dim: Hidden dimension of each head which we refer to as d
        n_kv_groups: Number of key/value groups

    Returns:
        w_q: Manipulated w_q (D, D)
        w_k: Manipulated w_k (D, D)

    """

    if n_kv_groups is not None:
        k = repeat_kv(key, n_kv_groups)
        # wk = wk.repeat(n_kv_groups, 1)

    A = torch.zeros(n_head, head_dim, head_dim).to("cuda").to(torch.float16)
    for h_idx in range(n_head):
        start = h_idx * head_dim
        end = (h_idx + 1) * head_dim
        _, sq, vq = torch.svd(query[0, h_idx].to(torch.float))
        _, sk, _ = torch.svd(k[0, h_idx].to(torch.float))
        sq = sq.to(torch.float16)
        vq = vq.to(torch.float16)
        sk = sk.to(torch.float16)
        s = sq * sk
        a = torch.zeros(head_dim, head_dim, dtype=torch.float16, device="cuda")
        _, ind = s.sort()
        r, c = a.shape
        A[h_idx] = a.scatter(-1, ind.unsqueeze(0).repeat(r, 1), vq)
        # wq[start:end, :] = A[h_idx].t() @ wq[start:end]
        # wk[start:end, :] = A.t() @ wk[start:end]
    return A
