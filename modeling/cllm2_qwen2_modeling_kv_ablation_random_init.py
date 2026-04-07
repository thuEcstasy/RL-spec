from einops import rearrange
from torch import nn
import torch.nn.functional as F
import torch
import random
import math
from typing import Dict, Optional, Sequence, List, Tuple

# logits processors
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]
            
DynamicCache.delete_false_key_value = delete_false_key_value

@torch.inference_mode()
def jacobi_forward_ablation_random_init(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len = 64,
    temperature = 1.0,
    top_p = 0.9, 
    top_k = None,
    repetition_penalty = None, 
    lenience = 1.,
    accept_threshold = 0.99,
    tokenizer = None,
    accepted_history_ids: Optional[torch.LongTensor] = None,
    fixed_window: Optional[bool] = False,
    eos_token_id: Optional[int] = None,
    profile_sample_index: Optional[int] = None,
    profile_call_index: Optional[int] = None,
    profile_global_call_index: Optional[int] = None,
    capture_noisy_block: bool = False,
    capture_len: Optional[int] = None,
    ):

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    # Resolve EOS id
    eos_id = eos_token_id

    eos_enabled = eos_id is not None
    if not eos_enabled:
        print("!!! WARNING: EOS handling disabled since eos_token_id is None !!!")

    # ---- LogitsProcessor: greedy only
    from transformers.generation.logits_process import LogitsProcessorList
    #logits_processors = LogitsProcessorList()

    if prefill_phase: # prefill phase, just compute the keys & values of prompt, return first_correct_token

        # Reset sliding-call counter at the beginning of each sample.
        # Caller typically runs prefill once per sample.
        self._jacobi_sliding_call_idx = 0
        # Reset deferred profile backfill state per sample.
        self._jacobi_profile_pending = []
        
        if (attention_mask is None) or (input_ids.shape[1] > attention_mask.shape[1]):
            attention_mask = torch.ones_like(input_ids)
        
        inputs_embeds = self.model.embed_tokens(input_ids)
    
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
    
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        
        position_ids = cache_position.unsqueeze(0)
    
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                # "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    
        hidden_states = inputs_embeds
    
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
    
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.model.norm(hidden_states)
        
        logits = self.lm_head(hidden_states).float()
        
        #scores = logits_processors(input_ids, logits.squeeze(0)).unsqueeze(0) 
        #probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # ---- build prefill_drafted_n_gram from ARGMAX of model outputs over the draft block ----
        # take the last n_token_seq_len positions from the sequence and use their next-token predictions
        prefill_drafted_n_gram = torch.argmax(                                  
            logits[:, -n_token_seq_len-1:-1, :], dim=-1                                       
        )                   
        # shape: [1, n_token_seq_len] when input includes a full draft block
        # first_correct_token is mapped from the last token in the prompt
        first_correct_token = prefill_drafted_n_gram[0]

        # [PROFILING]: dump one prefill snapshot (out + greedy) for debugging.
        # This mirrors generation-phase profiling but runs only once per sample.
        # out_ids = input_ids[0].detach().cpu().tolist()
        # greedy_ids_full = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()
        # draft_input_ids = input_ids[0, -n_token_seq_len:].detach().cpu().tolist() if n_token_seq_len > 0 else []
        # draft_greedy_ids = prefill_drafted_n_gram[0].detach().cpu().tolist() if n_token_seq_len > 0 else []

        # topk = 10
        # topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
        # topk_logits = topk_logits[0].detach().cpu().tolist()
        # topk_indices = topk_indices[0].detach().cpu().tolist()

        # per_token = []
        # for pos in range(len(out_ids)):
        #     input_id = out_ids[pos]
        #     target_next_id = out_ids[pos + 1] if (pos + 1) < len(out_ids) else None
        #     greedy_next_id = greedy_ids_full[pos]

        #     if tokenizer is not None:
        #         input_text = tokenizer.decode([input_id], skip_special_tokens=False)
        #         target_next_text = tokenizer.decode([target_next_id], skip_special_tokens=False) if target_next_id is not None else None
        #         greedy_next_text = tokenizer.decode([greedy_next_id], skip_special_tokens=False)
        #     else:
        #         input_text = None
        #         target_next_text = None
        #         greedy_next_text = None

        #     topk_list = []
        #     for rank, (tok_id, tok_logit) in enumerate(zip(topk_indices[pos], topk_logits[pos]), start=1):
        #         topk_list.append(
        #             {
        #                 "rank": rank,
        #                 "token_id": tok_id,
        #                 "token": tokenizer.decode([tok_id], skip_special_tokens=False) if tokenizer is not None else None,
        #                 "logit": tok_logit,
        #                 "is_target_next": (tok_id == target_next_id) if target_next_id is not None else None,
        #                 "is_greedy_next": (tok_id == greedy_next_id),
        #             }
        #         )

        #     per_token.append(
        #         {
        #             "position": pos,
        #             "input_token": input_text,
        #             "target_next_token": target_next_text,
        #             "greedy_next_token": greedy_next_text,
        #             "next_token_matches": (target_next_id == greedy_next_id) if target_next_id is not None else None,
        #             "topk": topk_list,
        #         }
        #     )

        # # Global alignment summary with next-token shift:
        # # target sequence uses out_ids[1:], greedy sequence uses greedy_ids_full[:-1].
        # target_next_ids_aligned = out_ids[1:]
        # greedy_next_ids_aligned = greedy_ids_full[: len(target_next_ids_aligned)]
        # match_mask = [tid == gid for tid, gid in zip(target_next_ids_aligned, greedy_next_ids_aligned)]
        # mismatch_rel = next((i for i, ok in enumerate(match_mask) if not ok), None)
        # aligned_match_prefix_len = mismatch_rel if mismatch_rel is not None else len(match_mask)
        # first_mismatch_target_abs_position = (mismatch_rel + 1) if mismatch_rel is not None else None

        # if mismatch_rel is not None:
        #     target_suffix_ids = target_next_ids_aligned[mismatch_rel:]
        #     greedy_suffix_ids = greedy_next_ids_aligned[mismatch_rel:]
        # else:
        #     target_suffix_ids = []
        #     greedy_suffix_ids = []

        # if tokenizer is not None:
        #     target_next_tokens_aligned = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in target_next_ids_aligned]
        #     greedy_next_tokens_aligned = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in greedy_next_ids_aligned]
        #     target_mismatch_suffix_text = "".join(
        #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in target_suffix_ids
        #     )
        #     greedy_mismatch_suffix_text = "".join(
        #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in greedy_suffix_ids
        #     )
        #     first_mismatch_target_token = (
        #         tokenizer.decode([target_next_ids_aligned[mismatch_rel]], skip_special_tokens=False)
        #         if mismatch_rel is not None
        #         else None
        #     )
        #     first_mismatch_greedy_token = (
        #         tokenizer.decode([greedy_next_ids_aligned[mismatch_rel]], skip_special_tokens=False)
        #         if mismatch_rel is not None
        #         else None
        #     )
        # else:
        #     target_next_tokens_aligned = None
        #     greedy_next_tokens_aligned = None
        #     target_mismatch_suffix_text = None
        #     greedy_mismatch_suffix_text = None
        #     first_mismatch_target_token = None
        #     first_mismatch_greedy_token = None

        # prefill_file_tags = []
        # if profile_sample_index is not None:
        #     prefill_file_tags.append(f"s{int(profile_sample_index)}")
        # if profile_call_index is not None:
        #     prefill_file_tags.append(f"c{int(profile_call_index)}")
        # if profile_global_call_index is not None:
        #     prefill_file_tags.append(f"g{int(profile_global_call_index)}")
        # prefill_file_tags.append("prefill")
        # prefill_output_file = f"../../profiling/jacobi_debug_logits_{'_'.join(prefill_file_tags)}.json"

        # if tokenizer is not None:
        #     out_tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in out_ids]
        #     greedy_tokens_full = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in greedy_ids_full]
        #     draft_input_tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in draft_input_ids]
        #     draft_greedy_tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in draft_greedy_ids]
        # else:
        #     out_tokens = None
        #     greedy_tokens_full = None
        #     draft_input_tokens = None
        #     draft_greedy_tokens = None

        # import os
        # import json
        # os.makedirs(os.path.dirname(prefill_output_file), exist_ok=True)
        # with open(prefill_output_file, "w") as f:
        #     json.dump(
        #         {
        #             "phase": "prefill",
        #             "profile_sample_index": profile_sample_index,
        #             "profile_call_index": profile_call_index,
        #             "profile_global_call_index": profile_global_call_index,
        #             "sequence_length": int(input_ids.shape[1]),
        #             "n_token_seq_len": int(n_token_seq_len),
        #             "out_ids": out_ids,
        #             "out_tokens": out_tokens,
        #             "greedy_ids_full": greedy_ids_full,
        #             "greedy_tokens_full": greedy_tokens_full,
        #             "draft_input_ids_last_n": draft_input_ids,
        #             "draft_input_tokens_last_n": draft_input_tokens,
        #             "draft_greedy_ids": draft_greedy_ids,
        #             "draft_greedy_tokens": draft_greedy_tokens,
        #             "target_greedy_alignment": {
        #                 "alignment_rule": "target_next=out_ids[1:], greedy_next=greedy_ids_full[:-1]",
        #                 "aligned_length": len(target_next_ids_aligned),
        #                 "aligned_match_prefix_len": aligned_match_prefix_len,
        #                 "first_mismatch_relative_index": mismatch_rel,
        #                 "first_mismatch_target_abs_position": first_mismatch_target_abs_position,
        #                 "first_mismatch_target_token": first_mismatch_target_token,
        #                 "first_mismatch_greedy_token": first_mismatch_greedy_token,
        #                 "target_next_tokens_aligned": target_next_tokens_aligned,
        #                 "greedy_next_tokens_aligned": greedy_next_tokens_aligned,
        #                 "target_mismatch_suffix_text": target_mismatch_suffix_text,
        #                 "greedy_mismatch_suffix_text": greedy_mismatch_suffix_text,
        #             },
        #             "topk_k": topk,
        #             "per_token": per_token,
        #         },
        #         f,
        #         indent=2,
        #     )
        
        # crop KV back to prompt(remove appended draft)
        if (past_key_values is not None) and (n_token_seq_len > 0):
            past_key_values.delete_false_key_value(n_token_seq_len)    
        
        return past_key_values, first_correct_token, prefill_drafted_n_gram, 0, None

    else: # generation phase, input as random_initilized point ([first_corrected_token, tokens_from_prompt]) and output as fixed point

        assert past_key_values is not None

        import os
        import json

        vocab_size = self.config.vocab_size

        # --- helper: run a forward pass, return logits ---
        def _run_forward(tokens, kv):
            _embeds = self.model.embed_tokens(tokens)
            _amask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device)
            _seen = kv.get_seq_length()
            _cpos = torch.arange(_seen, _seen + tokens.shape[1], device=tokens.device)
            _pids = _cpos.unsqueeze(0)

            if not isinstance(_cm := _amask, dict):
                _mk = {
                    "config": self.config,
                    "input_embeds": _embeds,
                    "attention_mask": _amask,
                    "cache_position": _cpos,
                    "past_key_values": kv,
                }
                _cm = {"full_attention": create_causal_mask(**_mk)}
                if self.model.has_sliding_layers:
                    _cm["sliding_attention"] = create_sliding_window_causal_mask(**_mk)

            _hs = _embeds
            _pe = self.model.rotary_emb(_hs, _pids)
            for _dl in self.model.layers[: self.model.config.num_hidden_layers]:
                _hs = _dl(
                    _hs,
                    attention_mask=_cm[_dl.attention_type],
                    position_ids=_pids,
                    past_key_value=kv,
                    use_cache=True,
                    cache_position=_cpos,
                    position_embeddings=_pe,
                )[0]
            _hs = self.model.norm(_hs)
            return self.lm_head(_hs).float()

        batch, out, device = input_ids.shape[0], input_ids, input_ids.device

        # Pool for random sampling: prefer accepted_history_ids, fallback to out
        def _rand_from_pool(n):
            """Sample n tokens from history pool."""
            if accepted_history_ids is not None and accepted_history_ids.numel() > 0:
                pool = accepted_history_ids.view(-1)
            else:
                pool = out.view(-1)
            idx = torch.randint(0, pool.shape[0], (n,), device=device)
            return pool[idx].unsqueeze(0)  # [1, n]

        # Build parallel random draft (same first token, random tail from history)
        if out.shape[1] > 1:
            # out_rand = torch.cat((out[:, :1], _rand_from_pool(out.shape[1] - 1)), dim=-1)
            # using randint instead
            out_rand = torch.cat((out[:, :1], torch.randint(0, vocab_size, (1, out.shape[1] - 1), device=device)), dim=-1)
        else:
            out_rand = out.clone()

        rand_accepted_per_itr = []   # collect random acceptance counts
        clean_accepted_per_itr = []  # collect clean acceptance counts
        # In fixed-window mode, accepted output is dynamic-length; otherwise keep original fixed-size buffer.
        accepted_n_gram = out[:, :0].clone() if fixed_window else out

        total_accepted = 0
        itr = 0
        
        noisy_block_record = None   # unused, kept for compat
        noisy_target = None
        if fixed_window:
            self._jacobi_sliding_call_idx = getattr(self, "_jacobi_sliding_call_idx", 0) + 1
        sliding_call_idx = int(profile_call_index) if profile_call_index is not None else getattr(self, "_jacobi_sliding_call_idx", 0)

        if accepted_history_ids is not None:
            history_tensor = accepted_history_ids[0] if accepted_history_ids.dim() == 2 else accepted_history_ids
            history_prefix_ids_call = history_tensor.detach().cpu().tolist()
        else:
            history_prefix_ids_call = []

        if not hasattr(self, "_jacobi_profile_pending"):
            self._jacobi_profile_pending = []

        def _decode_ids(ids):
            if tokenizer is None:
                return None
            return [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in ids]

        def _flush_profile_pending(clean_prefix_ids_now):
            pending = getattr(self, "_jacobi_profile_pending", [])
            if not pending:
                return

            keep_pending = []
            for meta in pending:
                file_path = meta.get("file_path")
                start_offset = int(meta.get("start_offset", 0))
                span_len = int(meta.get("span_len", 0))
                pending_sample = meta.get("profile_sample_index")

                if pending_sample is not None and profile_sample_index is not None and int(pending_sample) != int(profile_sample_index):
                    keep_pending.append(meta)
                    continue

                if (not file_path) or span_len <= 0 or start_offset < 0:
                    continue

                if len(clean_prefix_ids_now) < (start_offset + span_len):
                    keep_pending.append(meta)
                    continue

                clean_slice_ids = clean_prefix_ids_now[start_offset : start_offset + span_len]
                clean_slice_tokens = _decode_ids(clean_slice_ids)

                try:
                    with open(file_path, "r") as rf:
                        payload = json.load(rf)
                except Exception:
                    continue

                noisy_vs_clean = payload.get("noisy_vs_clean_context")
                if not isinstance(noisy_vs_clean, dict):
                    noisy_vs_clean = {}

                noisy_vs_clean["resolved_clean_context_available"] = True
                noisy_vs_clean["resolved_clean_context_ids"] = clean_slice_ids
                noisy_vs_clean["resolved_clean_context_tokens"] = clean_slice_tokens
                noisy_vs_clean["resolved_at_profile_call_index"] = (
                    int(profile_call_index) if profile_call_index is not None else int(sliding_call_idx)
                )
                noisy_vs_clean["resolved_at_profile_global_call_index"] = profile_global_call_index
                payload["noisy_vs_clean_context"] = noisy_vs_clean

                try:
                    with open(file_path, "w") as wf:
                        json.dump(payload, wf, indent=2)
                except Exception:
                    continue

            self._jacobi_profile_pending = keep_pending

        def _current_clean_prefix_ids():
            accepted_now = accepted_n_gram[0, :total_accepted].detach().cpu().tolist() if total_accepted > 0 else []
            return history_prefix_ids_call + accepted_now

        # Resolve older noisy forwards as soon as future clean history is long enough.
        _flush_profile_pending(history_prefix_ids_call)

        while total_accepted < n_token_seq_len:
            itr += 1

            # ============================================================
            # Shadow forward: random draft → measure acceptance (no KV side-effects)
            # ============================================================
            rand_logits = _run_forward(out_rand, past_key_values)
            L_rand = out_rand.shape[1]
            rand_greedy = torch.argmax(rand_logits[:, :-1, :], dim=-1)
            rand_mismatch = (out[:, 1:] != rand_greedy)
            rand_acc = int((rand_mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0]) + 1
            # Undo the KV entries the shadow forward appended
            past_key_values.delete_false_key_value(L_rand)

            # ============================================================
            # Real forward: greedy draft (identical to original greedy)
            # ============================================================
            inputs_embeds = self.model.embed_tokens(out)
            attention_mask = torch.ones_like(out, device=input_ids.device)

            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + out.shape[1], device=inputs_embeds.device
            )
            position_ids = cache_position.unsqueeze(0)

            if not isinstance(causal_mask_mapping := attention_mask, dict):
                mask_kwargs = {
                    "config": self.config,
                    "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                }
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                if self.model.has_sliding_layers:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        
            hidden_states = inputs_embeds
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            for layer_idx, decoder_layer in enumerate(self.model.layers[: self.model.config.num_hidden_layers]):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]
                # # decode in every layer to get intermediate logits for profiling, which is useful for understanding the model's behavior and debugging.
                # layer_hidden_states = self.model.norm(hidden_states)
                # layer_logits = self.lm_head(layer_hidden_states).float()
                # topk = 10
                # topk_logits, topk_indices = torch.topk(layer_logits, k=topk, dim=-1)
                # topk_logits = topk_logits[0].detach().cpu().tolist()
                # topk_indices = topk_indices[0].detach().cpu().tolist()
                # # [PROFILING]: dump intermediate logits for this layer
                # layer_file_tags = []
                # if profile_sample_index is not None:
                #     layer_file_tags.append(f"s{int(profile_sample_index)}")
                # layer_call_idx = int(profile_call_index) if profile_call_index is not None else int(sliding_call_idx)
                # layer_file_tags.append(f"c{layer_call_idx}")
                # if profile_global_call_index is not None:
                #     layer_file_tags.append(f"g{int(profile_global_call_index)}")
                # layer_file_tags.append(f"itr{int(itr)}")
                # layer_id = getattr(decoder_layer, "layer_id", getattr(decoder_layer, "layer_idx", layer_idx))
                # layer_file_tags.append(f"layer{int(layer_id)}")
                # layer_output_file = f"../../profiling/jacobi_debug_logits_{'_'.join(layer_file_tags)}.json"
                # out_ids = out[0].detach().cpu().tolist()
                # if tokenizer is not None:
                #     out_tokens = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in out_ids]
                # else:                    
                #     out_tokens = None
                # per_token = []
                # for pos in range(len(out_ids)):
                #     input_id = out_ids[pos]
                #     if tokenizer is not None:
                #         input_text = tokenizer.decode([input_id], skip_special_tokens=False)
                #     else:
                #         input_text = None

                #     topk_list = []
                #     for rank, (tok_id, tok_logit) in enumerate(zip(topk_indices[pos], topk_logits[pos]), start=1):
                #         topk_list.append(
                #             {
                #                 "rank": rank,
                #                 "token_id": tok_id,
                #                 "token": tokenizer.decode([tok_id], skip_special_tokens=False) if tokenizer is not None else None,
                #                 "logit": tok_logit,
                #             }
                #         )

                #     per_token.append(
                #         {
                #             "position": pos,
                #             "input_token": input_text,
                #             "topk": topk_list,
                #         }
                #     )

                # import os
                # import json
                # os.makedirs(os.path.dirname(layer_output_file), exist_ok=True)
                # with open(layer_output_file, "w") as f:
                #     json.dump(
                #         {
                #             "phase": "layer",
                #             "iteration": itr,
                #             "profile_sample_index": profile_sample_index,
                #             "profile_call_index": layer_call_idx,
                #             "profile_global_call_index": profile_global_call_index,
                #             "layer_index": int(layer_id),
                #             "sequence_length": len(out_ids),
                #             "input_tokens": out_tokens,
                #             "per_token": per_token,
                #         },
                #         f,
                #         indent=2,
                #     )
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states).float()

                
            # Apply logits processor, then softmax
            #p_scores = logits_processors(out, logits.squeeze(0)).unsqueeze(0) 
            #p_prob = torch.nn.functional.softmax(p_scores, dim=-1)
    
            # Greedy tokens for each draft position (exclude the last slot which is prob_next)
            greedy_tokens = torch.argmax(logits[:, :-1, :], dim=-1)      # [1, L-1]
            # Compare draft vs greedy: accept the longest exact-match prefix
            mismatch = (out[:, 1:] != greedy_tokens)
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)+1
            L = out.shape[1]
            num_accepted_raw = int(accepted[0])
            clean_accepted_per_itr.append(num_accepted_raw)
            rand_accepted_per_itr.append(min(rand_acc, num_accepted_raw))

            # Build accepted prefixes for profiling before writing this step into accepted_n_gram.
            current_prefix_raw_ids = torch.cat(
                (accepted_n_gram[0, :total_accepted], out[0, :num_accepted_raw]), dim=0
            ).detach().cpu().tolist()
            
            # [PROFILING]: for plotting logits
            # print("out_rand:", out_rand)
            # print("out:", out)
            # print("rand_greedy:", rand_greedy)
            # print("greedy_tokens:", greedy_tokens)
            
            # history_prefix_ids = []
            # if accepted_history_ids is not None:
            #     history_tensor = accepted_history_ids[0] if accepted_history_ids.dim() == 2 else accepted_history_ids
            #     history_prefix_ids = history_tensor.detach().cpu().tolist()
            # global_prefix_raw_ids = history_prefix_ids + current_prefix_raw_ids
            # itr_effective = sliding_call_idx if fixed_window else itr
            # file_tags = []
            # if profile_sample_index is not None:
            #     file_tags.append(f"s{int(profile_sample_index)}")
            # if sliding_call_idx > 0:
            #     file_tags.append(f"c{int(sliding_call_idx)}")
            # if profile_global_call_index is not None:
            #     file_tags.append(f"g{int(profile_global_call_index)}")
            # file_tags.append(f"itr{int(itr)}")
            # logits_output_file = f"../../profiling/jacobi_debug_logits_{'_'.join(file_tags)}.json"
            # # create filepath if not exists
            # os.makedirs(os.path.dirname(logits_output_file), exist_ok=True)
            # # select topk logits for each position
            # topk = 10
            # topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)
            # topk_logits = topk_logits[0].cpu().tolist()    # [L, topk]
            # topk_indices = topk_indices[0].cpu().tolist()  # [L, topk]

            # out_ids = out[0].cpu().tolist()                # [L]
            # greedy_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()  # [L]
            # noisy_context_start_offset = len(history_prefix_ids_call) + total_accepted
            # per_token = []
            # for pos in range(L):
            #     input_id = out_ids[pos]
            #     if tokenizer is not None:
            #         input_text = tokenizer.decode([input_id], skip_special_tokens=False)
            #     else:
            #         input_text = None

            #     target_next_id = out_ids[pos + 1] if (pos + 1) < L else None
            #     greedy_next_id = greedy_ids[pos]
            #     if tokenizer is not None:
            #         target_next_text = tokenizer.decode([target_next_id], skip_special_tokens=False) if target_next_id is not None else None
            #         greedy_next_text = tokenizer.decode([greedy_next_id], skip_special_tokens=False)
            #     else:
            #         target_next_text = None
            #         greedy_next_text = None

            #     topk_list = []
            #     for rank, (tok_id, tok_logit) in enumerate(zip(topk_indices[pos], topk_logits[pos]), start=1):
            #         topk_list.append({
            #             "rank": rank,
            #             "token_id": tok_id,
            #             "token": tokenizer.decode([tok_id], skip_special_tokens=False) if tokenizer is not None else None,
            #             "logit": tok_logit,
            #             "is_target_next": (tok_id == target_next_id) if target_next_id is not None else None,
            #             "is_greedy_next": (tok_id == greedy_next_id),
            #         })

            #     per_token.append({
            #         "position": pos,
            #         "input_token": input_text,
            #         "target_next_token": target_next_text,
            #         "greedy_next_token": greedy_next_text,
            #         "next_token_matches": (target_next_id == greedy_next_id) if target_next_id is not None else None,
            #         "topk": topk_list,
            #     })

            # # Global alignment summary with next-token shift:
            # # target sequence uses out_ids[1:], greedy sequence uses greedy_ids[:-1].
            # target_next_ids_aligned = out_ids[1:]
            # greedy_next_ids_aligned = greedy_ids[: len(target_next_ids_aligned)]
            # match_mask = [tid == gid for tid, gid in zip(target_next_ids_aligned, greedy_next_ids_aligned)]
            # mismatch_rel = next((i for i, ok in enumerate(match_mask) if not ok), None)
            # aligned_match_prefix_len = mismatch_rel if mismatch_rel is not None else len(match_mask)
            # first_mismatch_target_abs_position = (mismatch_rel + 1) if mismatch_rel is not None else None

            # if mismatch_rel is not None:
            #     target_suffix_ids = target_next_ids_aligned[mismatch_rel:]
            #     greedy_suffix_ids = greedy_next_ids_aligned[mismatch_rel:]
            # else:
            #     target_suffix_ids = []
            #     greedy_suffix_ids = []

            # if tokenizer is not None:
            #     target_next_tokens_aligned = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in target_next_ids_aligned]
            #     greedy_next_tokens_aligned = [tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in greedy_next_ids_aligned]
            #     target_mismatch_suffix_text = "".join(
            #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in target_suffix_ids
            #     )
            #     greedy_mismatch_suffix_text = "".join(
            #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in greedy_suffix_ids
            #     )
            #     first_mismatch_target_token = (
            #         tokenizer.decode([target_next_ids_aligned[mismatch_rel]], skip_special_tokens=False)
            #         if mismatch_rel is not None
            #         else None
            #     )
            #     first_mismatch_greedy_token = (
            #         tokenizer.decode([greedy_next_ids_aligned[mismatch_rel]], skip_special_tokens=False)
            #         if mismatch_rel is not None
            #         else None
            #     )
            # else:
            #     target_next_tokens_aligned = None
            #     greedy_next_tokens_aligned = None
            #     target_mismatch_suffix_text = None
            #     greedy_mismatch_suffix_text = None
            #     first_mismatch_target_token = None
            #     first_mismatch_greedy_token = None
            # if tokenizer is not None:
            #     current_prefix_raw_tokens = [
            #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in current_prefix_raw_ids
            #     ]
            #     global_prefix_raw_tokens = [
            #         tokenizer.decode([tok_id], skip_special_tokens=False) for tok_id in global_prefix_raw_ids
            #     ]
            # else:
            #     current_prefix_raw_tokens = None
            #     global_prefix_raw_tokens = None

            # noisy_context_tokens = _decode_ids(out_ids)
            # with open(logits_output_file, "w") as f:
            #     json.dump({
            #         "iteration": itr,
            #         "iteration_effective": itr_effective,
            #         "profile_sample_index": profile_sample_index,
            #         "profile_call_index": profile_call_index,
            #         "profile_global_call_index": profile_global_call_index,
            #         "sliding_call_index_internal": sliding_call_idx,
            #         "sequence_length": L,
            #         "num_accepted_raw": num_accepted_raw,
            #         "accepted_prefix_current_raw_tokens": current_prefix_raw_tokens,
            #         "accepted_prefix_global_raw_tokens": global_prefix_raw_tokens,
            #         "target_greedy_alignment": {
            #             "alignment_rule": "target_next=out_ids[1:], greedy_next=greedy_ids[:-1]",
            #             "aligned_length": len(target_next_ids_aligned),
            #             "aligned_match_prefix_len": aligned_match_prefix_len,
            #             "first_mismatch_relative_index": mismatch_rel,
            #             "first_mismatch_target_abs_position": first_mismatch_target_abs_position,
            #             "first_mismatch_target_token": first_mismatch_target_token,
            #             "first_mismatch_greedy_token": first_mismatch_greedy_token,
            #             "target_next_tokens_aligned": target_next_tokens_aligned,
            #             "greedy_next_tokens_aligned": greedy_next_tokens_aligned,
            #             "target_mismatch_suffix_text": target_mismatch_suffix_text,
            #             "greedy_mismatch_suffix_text": greedy_mismatch_suffix_text,
            #         },
            #         "noisy_vs_clean_context": {
            #             "noisy_start_offset": noisy_context_start_offset,
            #             "noisy_length": len(out_ids),
            #             "noisy_context_ids": out_ids,
            #             "noisy_context_tokens": noisy_context_tokens,
            #             "resolved_clean_context_available": False,
            #             "resolved_clean_context_ids": None,
            #             "resolved_clean_context_tokens": None,
            #             "resolved_at_profile_call_index": None,
            #             "resolved_at_profile_global_call_index": None,
            #         },
            #         "per_token": per_token,
            #     }, f, indent=2)

            # self._jacobi_profile_pending.append(
            #     {
            #         "file_path": os.path.abspath(logits_output_file),
            #         "start_offset": noisy_context_start_offset,
            #         "span_len": len(out_ids),
            #         "profile_sample_index": profile_sample_index,
            #     }
            # )

            # if tokenizer is not None:
            #     input_draft_str = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
            #     output_draft_str = tokenizer.decode(greedy_tokens[0].tolist(), skip_special_tokens=False)
            #     accepted_str = tokenizer.decode(out[0, :num_accepted_raw].tolist(), skip_special_tokens=False)
            #     print(f"[Jacobi itr={itr}] input  draft({L} tokens):    {repr(input_draft_str)}")
            #     print(f"[Jacobi itr={itr}] output draft({L-1} tokens):  {repr(output_draft_str)}")
            #     print(f"[Jacobi itr={itr}] accepted({num_accepted_raw} tokens):       {repr(accepted_str)}")



            # --- EOS handling within accepted prefix
            num_accepted = num_accepted_raw
            if eos_enabled:
                # if EOS appears in the accepted prefix, cap acceptance at first EOS
                eos_in_prefix = (out[0, :num_accepted_raw] == eos_id)
                if eos_in_prefix.any():
                    first_eos_idx = torch.nonzero(eos_in_prefix, as_tuple=False)[0].item()
                    num_accepted = first_eos_idx + 1

            if not fixed_window:
                # Guard the fixed-size write path near block tail.
                remaining_slots = n_token_seq_len - total_accepted
                if num_accepted > remaining_slots:
                    num_accepted = remaining_slots

            # Write accepted portion (possibly capped at EOS)
            prev_total_accepted = total_accepted

            if num_accepted > 0:
                accepted_chunk = out[:, :num_accepted].clone()
                if fixed_window:
                    accepted_n_gram = torch.cat((accepted_n_gram, accepted_chunk), dim=-1)
                else:
                    accepted_n_gram[:, total_accepted:total_accepted + num_accepted] = accepted_chunk

            total_accepted += num_accepted

            # capture noisy block at the first moment when accepted tokens reach noisy_target
            if capture_noisy_block and noisy_block_record is None and noisy_target is not None:
                if prev_total_accepted < noisy_target <= total_accepted:
                    noisy_full = torch.cat(
                        [accepted_n_gram[:, :prev_total_accepted].clone(), out.clone()],
                        dim=-1,
                    )[:, :n_token_seq_len]

                    noisy_block_record = {
                        "noisy_block": noisy_full,
                        "accepted_prefix": accepted_n_gram[:, :noisy_target].clone(),
                        "accepted_len": int(noisy_target),
                        "itr": int(itr),
                    }

            # If EOS was inside the accepted prefix, finalize immediately
            if eos_enabled and (out[0, :num_accepted] == eos_id).any():
                # Trim KV cache to exactly the accepted sequence length
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    past_key_values.delete_false_key_value(to_delete)
                # Return truncated outputs up to EOS
                _flush_profile_pending(_current_clean_prefix_ids())
                return past_key_values, torch.full((1,1), eos_id, device=device, dtype=out.dtype), accepted_n_gram[:, :total_accepted], itr, {"rand_accepted_per_itr": rand_accepted_per_itr, "clean_accepted_per_itr": clean_accepted_per_itr}

            has_rejected = (num_accepted_raw < L)  # note: use raw to preserve original mismatch logic
            # BRANCH: WITH REJECTED TOKENS IN THE DRAFT
            if has_rejected:
                # Delete false keys&values for the rejected tail
                past_key_values.delete_false_key_value(out.shape[1]-num_accepted_raw)
                # Next token is the greedy token at the first mismatch position
                next_token = torch.argmax(logits[:, num_accepted_raw-1, :], dim=-1, keepdim=True)

                # --- EOS on the next sampled token, return
                if eos_enabled and next_token.item() == eos_id:
                    # accept EOS and stop
                    if fixed_window:
                        accepted_n_gram = torch.cat((accepted_n_gram, next_token), dim=-1)
                    else:
                        accepted_n_gram[:, total_accepted:total_accepted+1] = next_token
                    total_accepted += 1
                    current_len = past_key_values.get_seq_length()
                    desired_len = total_accepted
                    to_delete = max(0, current_len - desired_len)
                    if to_delete > 0:
                        past_key_values.delete_false_key_value(to_delete)
                    _flush_profile_pending(_current_clean_prefix_ids())
                    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, {"rand_accepted_per_itr": rand_accepted_per_itr, "clean_accepted_per_itr": clean_accepted_per_itr}

                # keep drafting from the mismatch token
                out = next_token
                # Rebuild draft tail greedily from the remaining positions in this pass (after the mismatch slot)
                q_probs_rem = logits[:, num_accepted_raw:-1, :]
                if q_probs_rem.shape[1] > 0:
                    q_sampled = torch.argmax(q_probs_rem, dim=-1)  # [1, L']
                    out = torch.cat((out, q_sampled), dim=-1)

                # Build parallel random draft (same first token, random tail from history)
                rand_tail_len = out.shape[1] - 1
                if rand_tail_len > 0:
                    # out_rand = torch.cat((next_token, _rand_from_pool(rand_tail_len)), dim=-1)
                    out_rand = torch.cat((out[:, :1], torch.randint(0, vocab_size, (next_token.shape[0], rand_tail_len), device=device)), dim=-1)
                else:
                    out_rand = next_token.clone()

                # Optional sliding-window mode: keep a constant draft size for every iteration.
                if fixed_window:
                    if out.shape[1] < n_token_seq_len:
                        pad_len = n_token_seq_len - out.shape[1]
                        if (accepted_history_ids is not None) and (accepted_history_ids.numel() > 0):
                            pool = accepted_history_ids
                        else:
                            pool = out
                        rand_idx = torch.randint(0, pool.shape[1], (pad_len,), device=out.device)
                        pad_tokens = pool[:, rand_idx]
                        out = torch.cat((out, pad_tokens), dim=-1)
                        # pad out_rand to same length
                        # out_rand = torch.cat((out_rand, _rand_from_pool(pad_len)), dim=-1)
                        out_rand = torch.cat((out_rand, torch.randint(0, vocab_size, (out_rand.shape[0], pad_len), device=device)), dim=-1)
                    elif out.shape[1] > n_token_seq_len:
                        out = out[:, :n_token_seq_len]
                        out_rand = out_rand[:, :n_token_seq_len]
            
            # BRANCH: WITHOUT REJECTED TOKENS IN THE DRAFT
            else:
                # If we didn't reject anything, append the next greedy token and finish this block
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # --- write the appended token to accepted_n_gram
                if fixed_window:
                    accepted_n_gram = torch.cat((accepted_n_gram, next_token), dim=-1)
                else:
                    accepted_n_gram[:, total_accepted:total_accepted+1] = next_token
                total_accepted += 1

                # --- EOS handling on the appended next token
                if eos_enabled and next_token.item() == eos_id:
                    current_len = past_key_values.get_seq_length()
                    desired_len = total_accepted
                    to_delete = max(0, current_len - desired_len)
                    if to_delete > 0:
                        past_key_values.delete_false_key_value(to_delete)
                    _flush_profile_pending(_current_clean_prefix_ids())
                    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, {"rand_accepted_per_itr": rand_accepted_per_itr, "clean_accepted_per_itr": clean_accepted_per_itr}

                # All accepted → next iteration is a single token for both
                out_rand = next_token.clone()

            # Sliding-window mode: one Jacobi update per outer call.
            # Stash the current `out` so the caller can reuse it as a warm draft
            # for the next call — this is what makes Jacobi iteration converge.
            if fixed_window:
                self._jacobi_draft = out
                break

        # Hit length limit without EOS
        _flush_profile_pending(_current_clean_prefix_ids())
        return past_key_values, next_token, accepted_n_gram[:, :total_accepted], itr, {"rand_accepted_per_itr": rand_accepted_per_itr, "clean_accepted_per_itr": clean_accepted_per_itr}