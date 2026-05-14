from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformers import AutoModel

from .data import Vocabulary


@dataclass
class EncoderState:
    outputs: torch.Tensor
    mask: torch.Tensor
    summary_hidden: torch.Tensor
    summary_cell: torch.Tensor
    fixed_context: torch.Tensor


@dataclass
class DecoderState:
    hidden: torch.Tensor
    cell: torch.Tensor
    prev_context: torch.Tensor


class GloveBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=pad_id)
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size * 2

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> EncoderState:
        token_embeddings = self.dropout(self.embedding(source_ids))
        packed_embeddings = pack_padded_sequence(
            token_embeddings,
            lengths=source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_encoder_outputs, (hidden_state, cell_state) = self.lstm(packed_embeddings)
        encoder_outputs, _ = pad_packed_sequence(packed_encoder_outputs, batch_first=True)
        encoder_outputs = self.dropout(encoder_outputs)

        summary_hidden = torch.cat([hidden_state[-2], hidden_state[-1]], dim=-1)
        summary_cell = torch.cat([cell_state[-2], cell_state[-1]], dim=-1)
        fixed_context = summary_hidden
        mask = source_attention_mask.bool()
        return EncoderState(
            outputs=encoder_outputs,
            mask=mask,
            summary_hidden=summary_hidden,
            summary_cell=summary_cell,
            fixed_context=fixed_context,
        )


class BertEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.output_size = self.model.config.hidden_size

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> EncoderState:
        del source_lengths
        encoder_outputs = self.model(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
        ).last_hidden_state
        mask = source_attention_mask.bool()
        masked_outputs = encoder_outputs * source_attention_mask.unsqueeze(-1)
        mean_pooled_context = masked_outputs.sum(dim=1) / source_attention_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return EncoderState(
            outputs=encoder_outputs,
            mask=mask,
            summary_hidden=mean_pooled_context,
            summary_cell=mean_pooled_context,
            fixed_context=mean_pooled_context,
        )


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.key_projection = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.query_projection = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.energy = nn.Linear(decoder_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_projection(decoder_hidden).unsqueeze(1)
        keys = self.key_projection(encoder_outputs)
        energies = self.energy(torch.tanh(query + keys)).squeeze(-1)
        energies = energies.masked_fill(~encoder_mask, -1e9)
        attention_weights = torch.softmax(energies, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        context_dim: int,
        use_attention: bool,
        pad_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.context_dim = context_dim
        # Keep the registered module name as "cell" so older checkpoints with
        # decoder.cell.* keys continue to load cleanly.
        self.cell = nn.LSTMCell(embedding_dim + context_dim, hidden_size)
        self.attention = AdditiveAttention(context_dim, hidden_size) if use_attention else None
        self.output_projection = nn.Linear(hidden_size + context_dim, vocab_size)

    @property
    def recurrent_cell(self) -> nn.LSTMCell:
        return self.cell

    def init_state(self, hidden: torch.Tensor, cell: torch.Tensor) -> DecoderState:
        prev_context = torch.zeros(hidden.size(0), self.context_dim, device=hidden.device)
        return DecoderState(hidden=hidden, cell=cell, prev_context=prev_context)

    def step(
        self,
        input_tokens: torch.Tensor,
        state: DecoderState,
        encoder_state: EncoderState,
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        token_embeddings = self.dropout(self.embedding(input_tokens))
        if self.use_attention:
            decoder_input = torch.cat([token_embeddings, state.prev_context], dim=-1)
        else:
            decoder_input = torch.cat([token_embeddings, encoder_state.fixed_context], dim=-1)

        next_hidden, next_cell = self.cell(decoder_input, (state.hidden, state.cell))

        if self.use_attention:
            attention_context, attention_weights = self.attention(
                next_hidden,
                encoder_state.outputs,
                encoder_state.mask,
            )
        else:
            attention_context = encoder_state.fixed_context
            attention_weights = None

        decoder_logits = self.output_projection(
            self.dropout(torch.cat([next_hidden, attention_context], dim=-1))
        )
        next_state = DecoderState(
            hidden=next_hidden,
            cell=next_cell,
            prev_context=attention_context,
        )
        return decoder_logits, next_state, attention_weights


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: LSTMDecoder,
        encoder_state_dim: int,
        decoder_hidden_size: int,
        target_pad_id: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bridge_hidden = nn.Linear(encoder_state_dim, decoder_hidden_size)
        self.bridge_cell = nn.Linear(encoder_state_dim, decoder_hidden_size)
        self.target_pad_id = target_pad_id

    def encode(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> EncoderState:
        return self.encoder(source_ids, source_attention_mask, source_lengths)

    def init_decoder_state(self, encoder_state: EncoderState) -> DecoderState:
        hidden = torch.tanh(self.bridge_hidden(encoder_state.summary_hidden))
        cell = torch.tanh(self.bridge_cell(encoder_state.summary_cell))
        return self.decoder.init_state(hidden, cell)

    def decode_step(
        self,
        input_tokens: torch.Tensor,
        decoder_state: DecoderState,
        encoder_state: EncoderState,
    ) -> tuple[torch.Tensor, DecoderState, torch.Tensor | None]:
        return self.decoder.step(input_tokens, decoder_state, encoder_state)

    def forward(
        self,
        batch: dict[str, Any],
        teacher_forcing_ratio: float,
    ) -> torch.Tensor:
        target_ids = batch["target_ids"]
        encoder_state = self.encode(
            batch["source_ids"],
            batch["source_attention_mask"],
            batch["source_lengths"],
        )
        decoder_state = self.init_decoder_state(encoder_state)

        batch_size, target_length = target_ids.shape
        decoder_step_logits = []
        current_input_tokens = target_ids[:, 0]

        for timestep in range(1, target_length):
            step_logits, decoder_state, _ = self.decode_step(
                current_input_tokens,
                decoder_state,
                encoder_state,
            )
            decoder_step_logits.append(step_logits.unsqueeze(1))

            predicted_next_tokens = step_logits.argmax(dim=-1)
            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                current_input_tokens = target_ids[:, timestep]
            else:
                current_input_tokens = predicted_next_tokens

        if not decoder_step_logits:
            vocab_size = self.decoder.output_projection.out_features
            return torch.empty(batch_size, 0, vocab_size, device=target_ids.device)
        return torch.cat(decoder_step_logits, dim=1)


def build_model(
    config: dict[str, Any],
    source_processor: Vocabulary | Any,
    target_vocab_size: int,
    source_pad_id: int,
    target_pad_id: int,
    embedding_matrix: torch.Tensor | None = None,
) -> Seq2SeqModel:
    model_config = config["model"]
    source_representation = config["source"]["type"]

    if source_representation == "word":
        if embedding_matrix is None:
            raise ValueError("embedding_matrix is required for source.type='word'.")
        encoder = GloveBiLSTMEncoder(
            embedding_matrix=embedding_matrix,
            hidden_size=model_config["encoder_hidden_size"],
            num_layers=model_config["encoder_layers"],
            dropout=model_config["dropout"],
            pad_id=source_pad_id,
        )
        encoder_state_dim = encoder.output_size
        context_dim = encoder.output_size
    elif source_representation == "bert":
        del source_processor
        encoder = BertEncoder(
            model_name=model_config["bert_model_name"],
            freeze=model_config["freeze_encoder"],
        )
        encoder_state_dim = encoder.output_size
        context_dim = encoder.output_size
    else:
        raise ValueError(f"Unsupported source.type: {source_representation}")

    decoder = LSTMDecoder(
        vocab_size=target_vocab_size,
        embedding_dim=model_config["target_embedding_dim"],
        hidden_size=model_config["decoder_hidden_size"],
        context_dim=context_dim,
        use_attention=model_config["use_attention"],
        pad_id=target_pad_id,
        dropout=model_config["dropout"],
    )
    return Seq2SeqModel(
        encoder=encoder,
        decoder=decoder,
        encoder_state_dim=encoder_state_dim,
        decoder_hidden_size=model_config["decoder_hidden_size"],
        target_pad_id=target_pad_id,
    )
