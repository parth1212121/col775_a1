from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from .model import DecoderState, EncoderState, Seq2SeqModel


# ------------------------------
# Beam-search data structures
# ------------------------------

@dataclass
class BeamHypothesis:
    tokens: list[int]
    score: float
    state: DecoderState
    finished: bool


def copy_decoder_state(state: DecoderState) -> DecoderState:
    return DecoderState(
        hidden=state.hidden.clone(),
        cell=state.cell.clone(),
        prev_context=state.prev_context.clone(),
    )


# ------------------------------
# Text cleanup helpers
# ------------------------------

def trim_special_tokens(token_ids: list[int], tokenizer: PreTrainedTokenizerFast) -> list[int]:
    trimmed: list[int] = []
    for token_id in token_ids:
        if token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id}:
            continue
        if token_id == tokenizer.eos_token_id:
            break
        trimmed.append(token_id)
    return trimmed


def cleanup_decoded_text(text: str) -> str:
    normalized_text = " ".join(text.strip().split())
    normalized_text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", normalized_text)
    normalized_text = re.sub(r"([(\[\{])\s+", r"\1", normalized_text)
    normalized_text = re.sub(r"\s+([।॥])", r"\1", normalized_text)
    normalized_text = re.sub(r"([(\[\{])\s+", r"\1", normalized_text)
    normalized_text = re.sub(r"\s+'", "'", normalized_text)
    normalized_text = re.sub(r'"\s+', '"', normalized_text)
    return normalized_text.strip()


def decode_token_ids(token_ids: list[int], tokenizer: PreTrainedTokenizerFast) -> str:
    cleaned_ids = trim_special_tokens(token_ids, tokenizer)
    text = tokenizer.decode(
        cleaned_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return cleanup_decoded_text(text)


# ------------------------------
# Decoding routines
# ------------------------------

def greedy_decode_batch(
    model: Seq2SeqModel,
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    max_decode_steps: int,
) -> list[list[int]]:
    model.eval()
    with torch.inference_mode():
        encoder_state = model.encode(
            batch["source_ids"],
            batch["source_attention_mask"],
            batch["source_lengths"],
        )
        decoder_state = model.init_decoder_state(encoder_state)

        batch_size = batch["source_ids"].size(0)
        current_input_tokens = torch.full(
            (batch_size,),
            fill_value=tokenizer.bos_token_id,
            dtype=torch.long,
            device=batch["source_ids"].device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=batch["source_ids"].device)
        decoded_token_sequences = [[] for _ in range(batch_size)]

        for _ in range(max_decode_steps):
            step_logits, decoder_state, _ = model.decode_step(current_input_tokens, decoder_state, encoder_state)
            next_tokens = step_logits.argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, tokenizer.eos_token_id),
                next_tokens,
            )
            for index, token_id in enumerate(next_tokens.tolist()):
                decoded_token_sequences[index].append(token_id)
            finished = finished | next_tokens.eq(tokenizer.eos_token_id)
            current_input_tokens = next_tokens
            if finished.all():
                break

        return decoded_token_sequences


def length_penalized_beam_score(score: float, length: int, alpha: float) -> float:
    return score / ((5.0 + length) / 6.0) ** alpha


def slice_single_example_encoder_state(encoder_state: EncoderState, index: int) -> EncoderState:
    return EncoderState(
        outputs=encoder_state.outputs[index : index + 1],
        mask=encoder_state.mask[index : index + 1],
        summary_hidden=encoder_state.summary_hidden[index : index + 1],
        summary_cell=encoder_state.summary_cell[index : index + 1],
        fixed_context=encoder_state.fixed_context[index : index + 1],
    )


def rank_beam_hypothesis(hypothesis: BeamHypothesis, length_penalty_alpha: float) -> float:
    return length_penalized_beam_score(
        hypothesis.score,
        len(hypothesis.tokens),
        length_penalty_alpha,
    )


def expand_single_beam_hypothesis(
    *,
    model: Seq2SeqModel,
    hypothesis: BeamHypothesis,
    encoder_state: EncoderState,
    beam_size: int,
    eos_token_id: int,
) -> list[BeamHypothesis]:
    if hypothesis.finished:
        return [hypothesis]

    current_input_token = torch.tensor([hypothesis.tokens[-1]], device=encoder_state.outputs.device)
    step_logits, next_state, _ = model.decode_step(current_input_token, hypothesis.state, encoder_state)
    log_probabilities = torch.log_softmax(step_logits.squeeze(0), dim=-1)
    top_scores, top_tokens = torch.topk(log_probabilities, k=beam_size)

    expanded_hypotheses: list[BeamHypothesis] = []
    for token_score, token_id in zip(top_scores.tolist(), top_tokens.tolist()):
        expanded_hypotheses.append(
            BeamHypothesis(
                tokens=hypothesis.tokens + [token_id],
                score=hypothesis.score + token_score,
                state=copy_decoder_state(next_state),
                finished=(token_id == eos_token_id),
            )
        )
    return expanded_hypotheses


def select_top_beams(
    candidates: list[BeamHypothesis],
    *,
    beam_size: int,
    length_penalty_alpha: float,
) -> list[BeamHypothesis]:
    return sorted(
        candidates,
        key=lambda item: rank_beam_hypothesis(item, length_penalty_alpha),
        reverse=True,
    )[:beam_size]


def beam_search_decode_batch(
    model: Seq2SeqModel,
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    beam_size: int,
    max_decode_steps: int,
    length_penalty_alpha: float,
) -> tuple[list[list[int]], float]:
    model.eval()
    predicted_token_sequences: list[list[int]] = []
    total_decode_time = 0.0

    with torch.inference_mode():
        batch_encoder_state = model.encode(
            batch["source_ids"],
            batch["source_attention_mask"],
            batch["source_lengths"],
        )

        for sample_index in range(batch["source_ids"].size(0)):
            encoder_state = slice_single_example_encoder_state(batch_encoder_state, sample_index)
            initial_state = model.init_decoder_state(encoder_state)

            active_hypotheses = [
                BeamHypothesis(
                    tokens=[tokenizer.bos_token_id],
                    score=0.0,
                    state=initial_state,
                    finished=False,
                )
            ]

            start_time = time.perf_counter()
            for _ in range(max_decode_steps):
                expanded_hypotheses: list[BeamHypothesis] = []
                for hypothesis in active_hypotheses:
                    expanded_hypotheses.extend(
                        expand_single_beam_hypothesis(
                            model=model,
                            hypothesis=hypothesis,
                            encoder_state=encoder_state,
                            beam_size=beam_size,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    )

                active_hypotheses = select_top_beams(
                    expanded_hypotheses,
                    beam_size=beam_size,
                    length_penalty_alpha=length_penalty_alpha,
                )

                if all(hypothesis.finished for hypothesis in active_hypotheses):
                    break

            total_decode_time += time.perf_counter() - start_time
            best_hypothesis = max(
                active_hypotheses,
                key=lambda item: rank_beam_hypothesis(item, length_penalty_alpha),
            )
            predicted_token_sequences.append(best_hypothesis.tokens[1:])

    average_time = total_decode_time / max(1, len(predicted_token_sequences))
    return predicted_token_sequences, average_time


def generate_text_predictions(
    model: Seq2SeqModel,
    batch: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    beam_size: int,
    max_decode_steps: int,
    length_penalty_alpha: float,
) -> tuple[list[str], float]:
    if beam_size <= 1:
        start_time = time.perf_counter()
        token_predictions = greedy_decode_batch(model, batch, tokenizer, max_decode_steps)
        average_time = (time.perf_counter() - start_time) / max(1, len(token_predictions))
    else:
        token_predictions, average_time = beam_search_decode_batch(
            model,
            batch,
            tokenizer,
            beam_size=beam_size,
            max_decode_steps=max_decode_steps,
            length_penalty_alpha=length_penalty_alpha,
        )
    return [decode_token_ids(tokens, tokenizer) for tokens in token_predictions], average_time
