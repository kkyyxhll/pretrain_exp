# Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Unmodified function from Megatron-LM b92c3ed70029005fd5769d2cf4c65bc3d061a74b
# Source: megatron/training/training.py
# The caller supplies the three upstream classification helpers.

def num_floating_point_operations(
    args,
    batch_size,
    seqlen_squared_sum_in_batch=None,
    total_real_tokens_in_batch=None,
):
    """Compute the number of floating-point operations for one global batch.

    Args:
        args: Megatron args (provides hidden size, num layers, seq length, ...).
        batch_size: Global batch size (samples per iteration). Only used to
            compute the default token counts for the BSHD path -- with THD
            both ``total_real_tokens_in_batch`` and ``seqlen_squared_sum_in_batch``
            are passed in explicitly and ``batch_size`` is irrelevant.
        seqlen_squared_sum_in_batch: ``sum_i(L_i ** 2)`` across all REAL
            (unpadded) sub-sequences in the global batch. Drives the
            core-attention L^2 FLOPs. For BSHD this equals
            ``batch_size * args.seq_length ** 2`` (the default when ``None``).
            For THD it is the actual ragged sum and is strictly less than the
            BSHD value, reflecting per-chunk causal masking AND the fact that
            padding tokens do not contribute to attention scores.
        total_real_tokens_in_batch: ``sum_i(L_i)``, the TOTAL REAL (unpadded)
            token count across the global batch. Drives all token-linear FLOPs
            (QKV+output projections, MLP, MoE, MTP norms/projs, logits). For
            BSHD this equals ``batch_size * args.seq_length`` (the default when
            ``None``). For THD it equals the real token count -- strictly less
            than ``batch_size * args.seq_length`` whenever the dataloader added
            CP-alignment padding or end-of-sequence padding, so neither kind of
            padding shows up in the reported FLOPs.
    """
    # Defaults: BSHD layout assumption (full causal mask, every sample length =
    # seq_length, no padding). For BSHD ``total_real_tokens = batch * s`` and
    # ``seqlen_squared_sum = batch * s^2`` recover the original closed-form.
    if seqlen_squared_sum_in_batch is None:
        seqlen_squared_sum_in_batch = batch_size * args.seq_length * args.seq_length
    if total_real_tokens_in_batch is None:
        total_real_tokens_in_batch = batch_size * args.seq_length

    def mlp_layer_flops(total_tokens, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * total_tokens * hidden_size**2

    def moe_layer_flops(total_tokens, hidden_size, moe_ffn_hidden_size,
                        shared_expert_ffn_hidden_size, num_experts_routed_to,
                        moe_latent_size=None, swiglu=False):
        """Calculate FLOPs for an MoE layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        if moe_latent_size is None:
            routed_flops = (4 * total_tokens * hidden_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
        else:
            # Routed experts run on moe_latent_size.
            routed_flops = (4 * total_tokens * moe_latent_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
            # Up proj and down proj.
            routed_flops += (4 * total_tokens * hidden_size * moe_latent_size)
        shared_flops = 4 * total_tokens * hidden_size * shared_expert_ffn_hidden_size * scale_factor
        return routed_flops + shared_flops

    def attn_layer_flops(
        total_tokens, seqlen_squared_sum, hidden_size, num_heads, gqa=True,
        gqa_groups=8, kv_channels=None,
    ):
        """Calculate FLOPs for an attention layer.

        Two parts:
          - Token-linear (QKV + output projections): scales with ``total_tokens``
            (= ``sum_i(L_i)``). Padding tokens are excluded by passing the real
            (unpadded) token count.
          - Core attention (``QK^T`` and ``softmax(QK^T) V``): scales with
            ``seqlen_squared_sum`` (= ``sum_i(L_i ** 2)``).
        """
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        # 4 * total_tokens * h * p * (h + h*(g/n)): QKV + output projections (fwd*3, with FMA*2).
        # 2 * sum(L^2) * h * p: core attention (causal mask -> /2 cancels with FMA *2).
        return (
            4 * total_tokens * hidden_size * p
            * (hidden_size + hidden_size * (g / num_heads))
            + 2 * seqlen_squared_sum * hidden_size * p
        )

    def mamba_layer_flops(total_tokens, hidden_size, state_dim=16,
                          head_dim=64, num_groups=1, num_heads=128):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (
                2
                * total_tokens
                * hidden_size
                * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * total_tokens * d_in * state_dim)  # scan
            + (2 * total_tokens * d_in * hidden_size)  # out_proj
        )

    def gated_delta_product_layer_flops(
        total_tokens,
        hidden_size,
        num_householder,
        state_dim=128,
        head_dim=64,
        num_groups=8,
        num_heads=None,
        conv_kernel_dim=4,
    ):
        """Calculate FLOPs for a Gated Delta Product (GDP) layer."""
        if num_heads is None:
            d_inner = 2 * hidden_size
            num_heads = d_inner // head_dim
        else:
            d_inner = num_heads * head_dim
        in_proj_dim = (
            d_inner * (1 + num_householder)
            + num_groups * state_dim * (1 + num_householder)
            + num_heads * (1 + num_householder)
        )
        conv_dim = d_inner * num_householder + num_groups * state_dim * (1 + num_householder)
        non_core_flops = 2 * total_tokens * (
            hidden_size * in_proj_dim
            + conv_kernel_dim * conv_dim
            + d_inner * hidden_size
        )
        # Best-case recurrent GDP core estimate. The FLA chunk kernel may do additional
        # score/solve/WY work, but this keeps the implementation-agnostic lower bound explicit.
        core_flops = (4 * num_householder + 3) * total_tokens * d_inner * state_dim
        return non_core_flops + core_flops

    def gdn_layer_flops(total_tokens, hidden_size,
                        qk_head_dim=128, v_head_dim=128,
                        num_qk_heads=16, num_v_heads=32,
                        conv_kernel_dim=4, use_gdn2=False):
        """Calculate FLOPs for a Gated Delta Net (GDN) layer."""
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads
        if use_gdn2:
            # GDN2 in_proj: hidden_size -> (4*qk_dim + 3*v_dim) for q, k, v, z, f, b, w
            in_proj_dim = 4 * qk_dim + 3 * v_dim
        else:
            # GDN in_proj: hidden_size -> (2*qk_dim + 2*v_dim + 2*num_v_heads)
            in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads
        return (
            2 * total_tokens * (
                # in_proj
                hidden_size * in_proj_dim
                # conv1d
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                # gated delta rule: KK^T, VK^T, S(a(I-bKK^T)), and SQ
                + num_v_heads * (v_head_dim ** 2) * 4
                # out_proj: v_dim -> hidden_size
                + hidden_size * v_dim
            )
        )

    def hybrid_flops(total_tokens, seqlen_squared_sum, hidden_size,
                     num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers,
                     gdp_num_householder,
                     num_gdn_layers=0,
                     mamba_state_dim=128, mamba_head_dim=64,
                     mamba_num_groups=8, mamba_num_heads=128,
                     num_attn_heads=32, gqa=True,
                     gqa_groups=8, kv_channels=None,
                     mlp_expansion=4.0, swiglu=False,
                     moe_latent_size=None,
                     moe_ffn_hidden_size=2048, shared_expert_ffn_hidden_size=2048, num_experts_routed_to=1,
                     use_gated_delta_product=False,
                     gdn_qk_head_dim=128, gdn_v_head_dim=128,
                     gdn_num_qk_heads=16, gdn_num_v_heads=32,
                     gdn_conv_kernel_dim=4, gdn_use_gdn2=False,
                     vocab_size=256000, mtp_num_layers=0):
        """Calculate total FLOPs for the hybrid model."""
        mamba_flops = (
            gated_delta_product_layer_flops(total_tokens, hidden_size,
                                            gdp_num_householder,
                                            mamba_state_dim, mamba_head_dim,
                                            mamba_num_groups, mamba_num_heads)
            if use_gated_delta_product
            else mamba_layer_flops(total_tokens, hidden_size,
                                   mamba_state_dim, mamba_head_dim,
                                   mamba_num_groups, mamba_num_heads)
        )
        flops_fwd = (
                num_attn_layers * attn_layer_flops(total_tokens, seqlen_squared_sum,
                                                   hidden_size, num_attn_heads, gqa,
                                                   gqa_groups, kv_channels) +
                num_mlp_layers * mlp_layer_flops(total_tokens, hidden_size,
                                                 mlp_expansion, swiglu) +
                num_mamba_layers * mamba_flops +
                num_moe_layers * moe_layer_flops(total_tokens, hidden_size, moe_ffn_hidden_size,
                                                 shared_expert_ffn_hidden_size, num_experts_routed_to,
                                                 moe_latent_size, swiglu) +
                num_gdn_layers * gdn_layer_flops(total_tokens, hidden_size,
                                                  gdn_qk_head_dim, gdn_v_head_dim,
                                                  gdn_num_qk_heads, gdn_num_v_heads,
                                                  gdn_conv_kernel_dim, gdn_use_gdn2) +
                (2 * total_tokens * hidden_size * vocab_size * (1 + mtp_num_layers))  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]

        if args.mtp_num_layers is not None:
            mtp_num_layers = args.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = args.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = args.num_layers

        moe_ffn_hidden_size = (
            args.moe_ffn_hidden_size
            if args.moe_ffn_hidden_size is not None
            else args.ffn_hidden_size
        )
        moe_latent_size = args.moe_latent_size
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )

        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        forward_backward_expansion_factor = 3
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        fma_expansion_factor = 2
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if args.swiglu else 2

        # self_attn is split into a token-linear part (projections, multiplied by
        # ``batch_size * args.seq_length`` like all other token-linear work) and a
        # core-attention part (``QK^T`` and ``softmax(QK^T) V``) whose compute scales
        # with ``sum_i(L_i ** 2)`` instead of ``batch * seq^2``. With unpacked BSHD the
        # two are equal, so the BSHD result is unchanged. With THD/packed sequences
        # ``seqlen_squared_sum_in_batch`` is strictly smaller, reflecting the per-chunk
        # causal mask of packed attention.
        if args.multi_latent_attention:
            assert not args.group_query_attention
            '''
            Basic arithmetic
            let B is batch size, s is seq_len, h is embedding dim,
            for one self_attnetion block (prenorm is not included)
            qkv projection:  6Bsh^2
            attn:            2Bs^2h
            attn over value: 2Bs^2h
            oproj:           2Bsh^2

            references
            https://arxiv.org/abs/2305.10403
            https://arxiv.org/abs/2205.05198
            '''
            ## MLA
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            # Token-linear part of MLA self-attention (lora projs, kv proj, RoPE, output proj).
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## q lora + rope + q norm
                    q_term
                    ## kv lora + rope + kv norm
                    + args.kv_lora_rank
                    * (
                        args.hidden_size
                        + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                        + 1
                    )
                    + args.hidden_size * args.qk_pos_emb_head_dim
                    ## o proj
                    + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
                )
            )
            # Core-attention (L^2) part: ``QK^T`` and ``(softmax(QK^T)) V``. The
            # ``/2`` accounts for the causal mask and the ``*2`` cancels it via FMA.
            standard_self_attn_core_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    / 2
                    + args.num_attention_heads * args.v_head_dim / 2
                )
            )

        else:
            ## MHA or GQA
            query_projection_size = args.kv_channels * args.num_attention_heads
            key_projection_size = args.kv_channels * args.num_query_groups
            value_projection_size = args.kv_channels * args.num_query_groups
            gate_projection_size = query_projection_size if args.attention_output_gate else 0
            # Token-linear part of self-attention (QKV projection + output projection).
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## qkv proj
                    args.hidden_size
                    * (
                        query_projection_size
                        + key_projection_size
                        + value_projection_size
                        + gate_projection_size
                    )
                    ## out proj
                    + query_projection_size
                    * args.hidden_size
                )
            )
            # Core-attention (L^2) part: ``QK^T`` and ``(softmax(QK^T)) V``.
            standard_self_attn_core_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * query_projection_size
                / 2  # causal mask (only half of the mask is non-zero)
                * 2  # QK^T and (QK^T)V
            )

        if is_linear_attention_variant(args.experimental_attention_variant):
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.linear_attention_freq, int):
                linear_attention_pattern = [
                    # [1,1,...,1,0,1,1,...,1,0,...]
                    0 if ((i + 1) % args.linear_attention_freq == 0)
                    else 1 for i in range(num_layers)
                ]
            elif isinstance(args.linear_attention_freq, list):
                linear_attention_pattern = args.linear_attention_freq
                assert len(linear_attention_pattern) == num_layers, (
                    f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                    f"expected {num_layers}, "
                    f"current linear attention pattern: {args.linear_attention_freq}"
                )
            elif args.linear_attention_freq is None:
                # This should be caught by config validation, but raise here as a safety check
                raise ValueError(
                    f"Linear attention type {args.experimental_attention_variant} is specified "
                    "but linear_attention_freq is None. "
                    "Please set linear_attention_freq to specify the LA/SDPA layer pattern."
                )
            else:
                raise ValueError(
                    f"Invalid linear_attention_freq: {type(args.linear_attention_freq)},"
                    f" {args.linear_attention_freq}"
                )
            num_linear_attention_layers = sum(linear_attention_pattern)
            num_standard_attention_layers = num_layers - num_linear_attention_layers

            if is_gated_delta_net_variant(args.experimental_attention_variant):
                # Calculate the FLOPs for the gated delta net attention.
                qk_head_dim = args.linear_key_head_dim
                v_head_dim = args.linear_value_head_dim
                num_qk_heads = args.linear_num_key_heads
                num_v_heads = args.linear_num_value_heads
                qk_dim = qk_head_dim * num_qk_heads
                v_dim = v_head_dim * num_v_heads
                if args.experimental_attention_variant == "gdn2":
                    # GDN2 in_proj: q, k, v, z, f, b, w
                    in_proj_dim = 4 * qk_dim + 3 * v_dim
                else:
                    in_proj_dim = 2 * qk_dim + 2 * v_dim + 2 * num_v_heads
                linear_self_attn_term = (
                    forward_backward_expansion_factor
                    * fma_expansion_factor
                    * (
                        ## in proj
                        args.hidden_size
                        * in_proj_dim
                        ## conv1d
                        + args.linear_conv_kernel_dim
                        * (2 * qk_dim + v_dim)
                        ## gated delta rule
                        + num_v_heads
                        * (v_head_dim ** 2)
                        * 4  # KK^T, VK^T, S(a(I-bKK^T)), and SQ
                        ## out proj
                        + args.hidden_size
                        * v_dim
                    )
                )
            else:
                raise ValueError(
                    "Invalid experimental_attention_variant: "
                    f"{args.experimental_attention_variant}"
                )
        else:
            num_linear_attention_layers = 0
            linear_self_attn_term = 0
            num_standard_attention_layers = num_layers

        # Token-linear self-attention work (projections + linear-attention variants).
        # Linear attention has no L^2 term, so it stays entirely token-linear.
        self_attn_term = (
            linear_self_attn_term * num_linear_attention_layers
            + standard_self_attn_term * num_standard_attention_layers
        )
        # Core attention (L^2) FLOPs per standard-attention layer.
        self_attn_core_term = standard_self_attn_core_term * num_standard_attention_layers

        # Token-linear FLOPs scale with the real (unpadded) token count.
        # For BSHD this falls back to ``batch_size * seq_length`` (no padding).
        total_floating_point_operations = (
            total_real_tokens_in_batch
            * (
                # MLP
                forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                    (args.ffn_hidden_size * ffn_expansion_factor)
                    * num_dense_layers
                    # routed experts
                    + (
                        (moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor)
                        if moe_latent_size is None
                        else (
                            (
                                moe_ffn_hidden_size
                                * num_experts_routed_to
                                * ffn_expansion_factor
                                * moe_latent_size
                                / args.hidden_size
                            )  # Routed experts run on moe_latent_size.
                            + 2 * moe_latent_size  # Up proj and down proj.
                        )
                    )
                    * num_moe_layers
                    # Shared Experts.
                    + (shared_expert_ffn_hidden_size * ffn_expansion_factor)
                    * num_moe_layers
                )
                # Self Attention (token-linear part).
                + self_attn_term
                # MTP norms and proj
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * mtp_num_layers
                * (
                    # MTP eh norm + final nrom
                    3 * args.hidden_size
                    # MTH eh proj
                    + 2 * args.hidden_size * args.hidden_size
                )
                # Logit.
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * args.padded_vocab_size
                * (mtp_num_layers + 1)  # MTP + final logit
            )
            # Self Attention (core L^2 part). For BSHD the default
            # ``seqlen_squared_sum_in_batch = batch_size * seq_length^2`` recovers the
            # original ``batch * seq^2 * Q_proj`` term; for THD this uses the actual
            # ragged ``sum(L_i^2)`` and so undercounts the BSHD figure correctly.
            + seqlen_squared_sum_in_batch * self_attn_core_term
        )
        return total_floating_point_operations

    def _uses_gated_delta_product_spec(args):
        """Return True when the selected hybrid stack spec swaps Mamba layers to GDP."""
        def _split_spec_part(part):
            return str(part).replace('[', ' ').replace(']', ' ').replace(',', ' ').split()

        spec = getattr(args, 'spec', None)
        if spec is None:
            return False
        if isinstance(spec, str):
            spec_parts = _split_spec_part(spec)
        else:
            spec_parts = []
            for part in spec:
                spec_parts.extend(_split_spec_part(part))
        if not spec_parts:
            return False
        return spec_parts[-1] in {'gdp_stack_spec', 'gated_delta_product_stack_spec'}

    # Main entrypoint for FLOPs calculation.
    if is_hybrid_model(args):
        # Calculate the number of each type of layer.
        from operator import itemgetter

        from megatron.core.models.hybrid.hybrid_layer_allocation import (
            Symbols,
            get_hybrid_layer_counts,
        )
        num_mamba_layers, num_gdn_layers, num_attn_layers, num_mlp_layers, num_moe_layers = (
            itemgetter(Symbols.MAMBA, Symbols.GDN, Symbols.ATTENTION, Symbols.MLP, Symbols.MOE)(
                get_hybrid_layer_counts(args.hybrid_layer_pattern)
            )
        )

        mtp_num_layers = args.mtp_num_layers
        if mtp_num_layers is None:
            mtp_num_layers = 0
        # Compute hybrid model FLOPs.
        return hybrid_flops(
            total_tokens=total_real_tokens_in_batch,
            seqlen_squared_sum=seqlen_squared_sum_in_batch,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
            num_gdn_layers=num_gdn_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_num_heads=args.mamba_num_heads,
            gdp_num_householder=args.gdp_num_householder,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            use_gated_delta_product=_uses_gated_delta_product_spec(args),
            moe_latent_size=args.moe_latent_size,
            moe_ffn_hidden_size=(args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None
                                 else args.ffn_hidden_size),
            shared_expert_ffn_hidden_size=(0 if args.moe_shared_expert_intermediate_size is None
                                           else args.moe_shared_expert_intermediate_size),
            num_experts_routed_to=args.moe_router_topk,
            gdn_qk_head_dim=args.linear_key_head_dim or 128,
            gdn_v_head_dim=args.linear_value_head_dim or 128,
            gdn_num_qk_heads=args.linear_num_key_heads or 16,
            gdn_num_v_heads=args.linear_num_value_heads or 32,
            gdn_conv_kernel_dim=args.linear_conv_kernel_dim or 4,
            gdn_use_gdn2=(args.experimental_attention_variant == "gdn2"),
            vocab_size=args.padded_vocab_size,
            mtp_num_layers=mtp_num_layers,
        )
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
