"""
Gradio Web Application for Neural Transducer Morphological Inflection.
Features:
- Language selection (English, German, French, Italian)
- Side-by-side comparison (100 examples vs 1,000 examples)
- Random sampling from official CoNLL-SIGMORPHON 2017 Dev and Test splits with gold target display
- Training exposure indicators (Lemma seen? Feature combination seen?) for both regimes
- Detailed action alignment traces (COPY, DELETE, INSERT, SUBSTITUTE)
- Curated linguistic examples
"""

import json
import os
import gradio as gr
from inference import TransducerSpaceEngine

# Initialize inference engine
engine = TransducerSpaceEngine()

# Load curated examples
EXAMPLES_FILE = os.path.join(os.path.dirname(__file__), "examples.json")
with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
    CURATED_EXAMPLES = json.load(f)

LANGUAGE_MAP = {
    "English": "eng",
    "German": "deu",
    "French": "fra",
    "Italian": "ita",
}


def format_action_trace(trace):
    if not trace:
        return "No action trace available."
    rows = []
    for r in trace:
        src = f"'{r.source_char}' (pos {r.source_pos})" if r.source_char else "-"
        emitted = f"'{r.emitted_char}'" if r.emitted_char else "-"
        rows.append(f"Step {r.step:02d}: {r.operation:10s} | In: {src:16s} | Out: {emitted:8s} | {r.description}")
    return "\n".join(rows)


def build_coverage_markdown(lang_iso, lemma, features_str):
    if not lemma or not features_str:
        return ""
    cov = engine.check_training_coverage(lang_iso, lemma, features_str)
    low_lemma = "✅ Seen in training" if cov["100"]["lemma_seen"] else "❌ Unseen (zero-shot lemma)"
    med_lemma = "✅ Seen in training" if cov["1000"]["lemma_seen"] else "❌ Unseen (zero-shot lemma)"

    low_feat = "✅ Seen in training" if cov["100"]["features_seen"] else "❌ Unseen combination"
    med_feat = "✅ Seen in training" if cov["1000"]["features_seen"] else "❌ Unseen combination"

    md = f"""
| Training Exposure | 100 Examples (Low) | 1,000 Examples (Medium) |
| :--- | :--- | :--- |
| **Lemma** (`{lemma}`) | {low_lemma} | {med_lemma} |
| **Features** (`{features_str}`) | {low_feat} | {med_feat} |
"""
    return md.strip()


def inflect(language_label, mode, lemma, features_str, gold_target=""):
    if not lemma or not features_str:
        return (
            "Please provide both a lemma and UniMorph features.",
            "-",
            "-",
            "Waiting for valid input...",
            "",
            "",
        )

    iso = LANGUAGE_MAP.get(language_label, "eng")
    lemma = lemma.strip()
    features_str = features_str.strip()
    gold_target = (gold_target or "").strip()

    cov_md = build_coverage_markdown(iso, lemma, features_str)

    try:
        if mode == "Compare Both (100 vs 1000)":
            res_100 = engine.predict(iso, "100", lemma, features_str, beam_width=1)
            res_1000 = engine.predict(iso, "1000", lemma, features_str, beam_width=1)

            pred_100 = res_100.predicted_form
            if gold_target:
                status_100 = " ✅ Match" if pred_100 == gold_target else " ❌ Mismatch"
                pred_100 = f"{pred_100}{status_100}"
            score_100 = f"{res_100.decoder_score:.4f}"

            pred_1000 = res_1000.predicted_form
            if gold_target:
                status_1000 = " ✅ Match" if pred_1000 == gold_target else " ❌ Mismatch"
                pred_1000 = f"{pred_1000}{status_1000}"
            score_1000 = f"{res_1000.decoder_score:.4f}"

            trace_text = (
                f"=== Actions (100 Examples) ===\n{format_action_trace(res_100.action_trace)}\n\n"
                f"=== Actions (1,000 Examples) ===\n{format_action_trace(res_1000.action_trace)}"
            )

            warn = ""
            if res_1000.unknown_features:
                warn = f"⚠️ Unseen individual tag tokens (treated as UNK): {', '.join(res_1000.unknown_features)}"

            return pred_100, score_100, pred_1000, score_1000, trace_text, warn, cov_md

        elif mode == "100 Examples (Low-Resource)":
            res = engine.predict(iso, "100", lemma, features_str, beam_width=1)
            pred = res.predicted_form
            if gold_target:
                pred += " ✅ Match" if pred == gold_target else " ❌ Mismatch"
            warn = f"⚠️ Unseen individual tag tokens: {', '.join(res.unknown_features)}" if res.unknown_features else ""
            return pred, f"{res.decoder_score:.4f}", "-", "-", format_action_trace(res.action_trace), warn, cov_md

        else:  # 1000 Examples
            res = engine.predict(iso, "1000", lemma, features_str, beam_width=1)
            pred = res.predicted_form
            if gold_target:
                pred += " ✅ Match" if pred == gold_target else " ❌ Mismatch"
            warn = f"⚠️ Unseen individual tag tokens: {', '.join(res.unknown_features)}" if res.unknown_features else ""
            return "-", "-", pred, f"{res.decoder_score:.4f}", format_action_trace(res.action_trace), warn, cov_md

    except Exception as e:
        return f"Error: {e}", "-", "-", "-", str(e), "", cov_md


def sample_random_item(language_label, split_choice):
    iso = LANGUAGE_MAP.get(language_label, "eng")
    sample = engine.get_random_sample(iso, split=split_choice)
    if not sample:
        return "sing", "V;PST", "", "No samples available."
    return sample["lemma"], sample["features"], sample["target"]


# Gradio UI definition
with gr.Blocks(title="Neural Transducer: Morphological Inflection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔤 Neural Transducer: Morphological Inflection Generation
        Demonstrating imitation-learning string transduction (Makarov & Clematide, 2020) on the **CoNLL-SIGMORPHON 2017** benchmark.
        Compare inflection generalization under **low-resource (100 examples)** vs. **medium-resource (1,000 examples)** supervision.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            lang_dropdown = gr.Dropdown(
                label="Language",
                choices=list(LANGUAGE_MAP.keys()),
                value="English",
            )
            mode_radio = gr.Radio(
                label="Regime / Comparison Mode",
                choices=["Compare Both (100 vs 1000)", "100 Examples (Low-Resource)", "1000 Examples (Medium-Resource)"],
                value="Compare Both (100 vs 1000)",
            )

            with gr.Group():
                gr.Markdown("#### 🎲 Benchmark Evaluation Sampler")
                with gr.Row():
                    split_selector = gr.Radio(
                        label="Benchmark Split",
                        choices=["Dev Set", "Test Set"],
                        value="Dev Set",
                    )
                    sample_btn = gr.Button("🎲 Sample Random Item", variant="secondary")

            lemma_input = gr.Textbox(
                label="Lemma",
                placeholder="e.g. sing",
                value="sing",
            )
            feat_input = gr.Textbox(
                label="UniMorph Morphological Features (semicolon-separated)",
                placeholder="e.g. V;PST",
                value="V;PST",
            )
            gold_target_input = gr.Textbox(
                label="Gold Target Form (Optional / Benchmark Reference)",
                placeholder="Populated automatically when sampling from Dev/Test",
                value="",
            )

            run_btn = gr.Button("Generate Inflection", variant="primary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎯 Generated Predictions (Greedy Decoding)")
                with gr.Row():
                    out_100 = gr.Textbox(label="Prediction (100 Examples)", interactive=False)
                    score_100 = gr.Textbox(label="Decoder Score (100)", interactive=False)
                with gr.Row():
                    out_1000 = gr.Textbox(label="Prediction (1,000 Examples)", interactive=False)
                    score_1000 = gr.Textbox(label="Decoder Score (1,000)", interactive=False)

                warning_box = gr.Markdown("")

            with gr.Group():
                gr.Markdown("### 📊 Training Data Exposure")
                coverage_box = gr.Markdown("Enter an input or sample from the benchmark to view coverage.")

            with gr.Accordion("🔍 Transducer Edit Action Trace (Alignment Steps)", open=False):
                trace_box = gr.Code(label="Action Sequence", language="markdown", interactive=False)

    gr.Markdown("### 💡 Click an Example to Load")
    example_rows = []
    for ex in CURATED_EXAMPLES:
        example_rows.append([ex["language"], "Compare Both (100 vs 1000)", ex["lemma"], ex["features"], ""])

    gr.Examples(
        examples=example_rows,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box],
        fn=inflect,
        cache_examples=False,
    )

    sample_btn.click(
        fn=sample_random_item,
        inputs=[lang_dropdown, split_selector],
        outputs=[lemma_input, feat_input, gold_target_input],
    ).then(
        fn=inflect,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box],
    )

    run_btn.click(
        fn=inflect,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box],
    )

if __name__ == "__main__":
    demo.launch()
