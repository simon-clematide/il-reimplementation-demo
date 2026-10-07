"""
Inference and action trace presentation layer.
Provides clean conversion from model outputs and action histories to human-readable
records (COPY, DELETE, INSERT, SUBSTITUTE, END) without exposing raw tensor IDs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch

from trans import actions
from trans import transducer
from trans import vocabulary


@dataclass
class ActionRecord:
    step: int
    operation: str
    source_pos: int
    source_char: Optional[str]
    emitted_char: Optional[str]
    description: str


@dataclass
class PredictionResult:
    lemma: str
    features: str
    predicted_form: str
    decoder_score: float  # Average action log-probability (not calibrated confidence)
    action_trace: List[ActionRecord]
    unknown_features: List[str]


def explain_action(
    action: Union[int, actions.Edit],
    vocab: vocabulary.FeatureVocabularies,
    source_tokens: List[str],
    current_pos: int,
    step: int,
) -> ActionRecord:
    """Formats an action (integer ID or Edit object) into a human-readable record."""
    if isinstance(action, int):
        action = vocab.decode_action(action)

    source_char = source_tokens[current_pos] if current_pos < len(source_tokens) else None

    if isinstance(action, (actions.Copy, actions.ConditionalCopy)):
        op = "COPY"
        emitted = source_char
        desc = f"Copy '{source_char}'"
    elif isinstance(action, (actions.Del, actions.ConditionalDel)):
        op = "DELETE"
        emitted = None
        desc = f"Delete '{source_char}'"
    elif isinstance(action, (actions.Ins, actions.ConditionalIns)):
        op = "INSERT"
        emitted = action.new
        desc = f"Insert '{emitted}'"
    elif isinstance(action, (actions.Sub, actions.ConditionalSub)):
        op = "SUBSTITUTE"
        emitted = action.new
        desc = f"Substitute '{source_char}' with '{emitted}'"
    elif isinstance(action, actions.EndOfSequence):
        op = "END"
        emitted = None
        desc = "End of sequence"
    elif isinstance(action, actions.BeginOfSequence):
        op = "BEGIN"
        emitted = None
        desc = "Begin of sequence"
    else:
        op = action.__class__.__name__.upper()
        emitted = getattr(action, "new", None)
        desc = f"Action {op}"

    return ActionRecord(
        step=step,
        operation=op,
        source_pos=current_pos,
        source_char=source_char,
        emitted_char=emitted,
        description=desc,
    )


def predict_single(
    model: transducer.Transducer,
    vocab: vocabulary.FeatureVocabularies,
    lemma: str,
    features_str: str,
    beam_width: int = 1,
) -> PredictionResult:
    """Performs inference for a single (lemma, features) pair and extracts action traces."""
    model.eval()
    device = model.device

    # Tokenize input
    input_tokens = model.source_tokenizer.tokenize(lemma)
    encoded_input = torch.tensor(
        vocab.encode_unseen_input(input_tokens), dtype=torch.long, device=device
    ).unsqueeze(dim=0)

    # Encode features
    feature_tokens = [f.strip() for f in features_str.split(";") if f.strip()]
    encoded_features = torch.tensor(
        vocab.encode_unseen_features(features_str), dtype=torch.long, device=device
    ).unsqueeze(dim=0)

    # Check unknown features
    unknown_features = [f for f in feature_tokens if f not in vocab.features.w2i]

    with torch.no_grad():
        if beam_width <= 1:
            out = model.transduce(
                [input_tokens],
                encoded_input,
                encoded_features,
            )
            predicted_form = out.output[0]
            action_history = out.action_history[0]
            decoder_score = float(out.log_p)
        else:
            hypotheses = model.beam_search_decode(
                input_tokens,
                encoded_input,
                encoded_features,
                beam_width,
            )
            top_h = hypotheses[0]
            predicted_form = top_h.output if isinstance(top_h.output, str) else model.target_tokenizer.untokenize(top_h.output)
            action_history = top_h.action_history
            decoder_score = float(top_h.log_p)

    # Trace action alignment progression
    trace: List[ActionRecord] = []
    pos = 0
    for step, act in enumerate(action_history, 1):
        rec = explain_action(act, vocab, input_tokens, pos, step)
        trace.append(rec)
        decoded_act = vocab.decode_action(act) if isinstance(act, int) else act
        if isinstance(decoded_act, (actions.Copy, actions.ConditionalCopy, actions.Del, actions.ConditionalDel, actions.Sub, actions.ConditionalSub)):
            pos += 1

    return PredictionResult(
        lemma=lemma,
        features=features_str,
        predicted_form=predicted_form,
        decoder_score=decoder_score,
        action_trace=trace,
        unknown_features=unknown_features,
    )
