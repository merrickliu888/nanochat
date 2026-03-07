import torch

from nanochat.gpt import GPT, GPTConfig


def _tiny_config():
    return GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
        mlp_type="relu2",
        tie_embeddings=True,
    )


def _tiny_config_untied():
    return GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
        mlp_type="relu2",
        tie_embeddings=False,
    )


def test_embedding_and_lm_head_weights_are_tied():
    model = GPT(_tiny_config())
    model.init_weights()

    assert model.lm_head.weight is model.transformer.wte.weight
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_optimizer_does_not_duplicate_tied_matrix():
    model = GPT(_tiny_config())
    model.init_weights()
    optimizer = model.setup_optimizer()

    tied_param_id = id(model.transformer.wte.weight)
    param_groups_with_tied = [
        group
        for group in optimizer.param_groups
        if any(id(param) == tied_param_id for param in group["params"])
    ]

    assert len(param_groups_with_tied) == 1
    flat_param_ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(flat_param_ids) == len(set(flat_param_ids))

    embedding_scale = (model.config.n_embd / 768) ** -0.5
    expected_embedding_lr = 0.2 * embedding_scale
    assert abs(param_groups_with_tied[0]["lr"] - expected_embedding_lr) < 1e-12


def test_num_scaling_params_dedupes_tied_output_projection():
    model = GPT(_tiny_config())
    model.init_weights()
    counts = model.num_scaling_params()

    expected = model.transformer.wte.weight.numel() + sum(
        ve.weight.numel() for ve in model.value_embeds.values()
    ) + sum(p.numel() for p in model.transformer.h.parameters()) + model.resid_lambdas.numel() + model.x0_lambdas.numel()

    assert counts["lm_head"] == 0
    assert counts["wte"] == model.transformer.wte.weight.numel()
    assert counts["total"] == expected


def test_untied_embeddings_have_separate_weights():
    model = GPT(_tiny_config_untied())
    model.init_weights()

    assert model.lm_head.weight is not model.transformer.wte.weight
    assert model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr()


def test_optimizer_uses_unembedding_lr_for_untied_lm_head():
    model = GPT(_tiny_config_untied())
    model.init_weights()
    optimizer = model.setup_optimizer()

    tied_param_id = id(model.transformer.wte.weight)
    lm_head_param_id = id(model.lm_head.weight)
    tied_group = next(
        group for group in optimizer.param_groups
        if any(id(param) == tied_param_id for param in group["params"])
    )
    lm_head_group = next(
        group for group in optimizer.param_groups
        if any(id(param) == lm_head_param_id for param in group["params"])
    )

    embedding_scale = (model.config.n_embd / 768) ** -0.5
    expected_embedding_lr = 0.2 * embedding_scale
    expected_unembedding_lr = 0.004 * embedding_scale
    assert abs(tied_group["lr"] - expected_embedding_lr) < 1e-12
    assert abs(lm_head_group["lr"] - expected_unembedding_lr) < 1e-12
    assert tied_group is not lm_head_group


def test_strict_load_state_dict_accepts_previously_untied_checkpoints():
    base = GPT(_tiny_config())
    base.init_weights()

    checkpoint = base.state_dict()
    untied_wte = torch.randn_like(checkpoint["transformer.wte.weight"])
    untied_lm_head = torch.randn_like(checkpoint["lm_head.weight"])
    checkpoint["transformer.wte.weight"] = untied_wte
    checkpoint["lm_head.weight"] = untied_lm_head

    model = GPT(_tiny_config())
    model.init_weights()
    incompat = model.load_state_dict(checkpoint, strict=True)
    assert len(incompat.missing_keys) == 0
    assert len(incompat.unexpected_keys) == 0
    assert model.lm_head.weight is model.transformer.wte.weight
    # Both keys should resolve to the same tensor after tying.
    assert torch.equal(model.lm_head.weight, model.transformer.wte.weight)


def test_forward_and_backward_work_with_tied_weights():
    model = GPT(_tiny_config())
    model.init_weights()

    idx = torch.randint(0, model.config.vocab_size, size=(2, 8))
    targets = torch.randint(0, model.config.vocab_size, size=(2, 8))

    logits = model(idx)
    assert logits.shape == (2, 8, model.config.vocab_size)

    loss = model(idx, targets=targets)
    assert torch.isfinite(loss).item()
    loss.backward()

    assert model.transformer.wte.weight.grad is not None
    assert not torch.isnan(model.transformer.wte.weight.grad).any()
