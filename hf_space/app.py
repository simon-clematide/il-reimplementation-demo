"""
Gradio Web Application for Neural Transducer Morphological Inflection.
Features:
- Language selection (English, German, French, Italian)
- Side-by-side comparison (100 examples vs 1,000 examples)
- Random sampling from official CoNLL-SIGMORPHON 2017 Dev and Test splits with gold target display
- Training exposure indicators (Lemma seen? Feature combination seen?) for both regimes
- Detailed action alignment traces (COPY, DELETE, INSERT, SUBSTITUTE)
- Interactive morphological feature explainer and full UniMorph tag glossary
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

TAG_DEFINITIONS = {
    # Part of Speech
    "V": ("Part of Speech", "Verb", "Base action or state word (e.g. sing, gehen, chanter)"),
    "N": ("Part of Speech", "Noun", "Person, place, thing, or concept (e.g. apple, Apfel)"),
    # Person
    "1": ("Person", "1st Person", "Speaker / speaker group (I, we)"),
    "2": ("Person", "2nd Person", "Addressee (you)"),
    "3": ("Person", "3rd Person", "Entity spoken about (he, she, it, they)"),
    # Number
    "SG": ("Number", "Singular", "Single entity (e.g. book)"),
    "PL": ("Number", "Plural", "Multiple entities (e.g. books)"),
    # Tense & Aspect
    "PRS": ("Tense", "Present", "Action happening currently (e.g. walks)"),
    "PST": ("Tense", "Past", "Action that occurred previously (e.g. walked)"),
    "FUT": ("Tense", "Future", "Action that will occur (e.g. ira in French)"),
    "IPFV": ("Aspect", "Imperfect / Imperfective", "Ongoing, habitual, or continuous past action (e.g. cantava in Italian)"),
    "PFV": ("Aspect", "Perfective", "Completed action / past historic / passé simple (e.g. chanta in French)"),
    # Mood & Finiteness
    "IND": ("Mood", "Indicative", "Objective statement of fact"),
    "SBJV": ("Mood", "Subjunctive", "Hypothetical, doubt, wish, or dependent clause"),
    "COND": ("Mood", "Conditional", "Action dependent on condition (would/should)"),
    "IMP": ("Mood", "Imperative", "Direct command or request (e.g. Listen!)"),
    "NFIN": ("Finiteness", "Infinitive / Non-finite", "Uninflected base dictionary form (e.g. to walk)"),
    "V.PTCP": ("Finiteness", "Participle", "Verbal adjective (e.g. broken, singing)"),
    "V.CVB": ("Finiteness", "Converb / Gerund", "Verbal adverb (e.g. cantando in Italian)"),
    # Case (German)
    "NOM": ("Case", "Nominative", "Subject of the sentence"),
    "ACC": ("Case", "Accusative", "Direct object of transitive verb"),
    "DAT": ("Case", "Dative", "Indirect object"),
    "GEN": ("Case", "Genitive", "Possessive / relationship of source"),
    # Degree (German)
    "POS": ("Degree", "Positive", "Base degree of adjective/adverb (e.g. schön)"),
}


def explain_features(features_str):
    if not features_str or not features_str.strip():
        return "*Enter feature tags above to see their breakdown.*"
    tags = [t.strip() for t in features_str.strip().split(";") if t.strip()]
    if not tags:
        return "*No valid tags provided.*"

    items = []
    for t in tags:
        if t in TAG_DEFINITIONS:
            category, name, desc = TAG_DEFINITIONS[t]
            items.append(f"- **`{t}`** ({category}: *{name}*) – {desc}")
        else:
            items.append(f"- **`{t}`** – *Unknown / custom tag*")
    return "\n".join(items)


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
    feat_explanation = explain_features(features_str)

    if not lemma or not features_str:
        return (
            "Please provide both a lemma and UniMorph features.",
            "-",
            "-",
            "Waiting for valid input...",
            "",
            "",
            feat_explanation,
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

            return pred_100, score_100, pred_1000, score_1000, trace_text, warn, cov_md, feat_explanation

        elif mode == "100 Examples (Low-Resource)":
            res = engine.predict(iso, "100", lemma, features_str, beam_width=1)
            pred = res.predicted_form
            if gold_target:
                pred += " ✅ Match" if pred == gold_target else " ❌ Mismatch"
            warn = f"⚠️ Unseen individual tag tokens: {', '.join(res.unknown_features)}" if res.unknown_features else ""
            return pred, f"{res.decoder_score:.4f}", "-", "-", format_action_trace(res.action_trace), warn, cov_md, feat_explanation

        else:  # 1000 Examples
            res = engine.predict(iso, "1000", lemma, features_str, beam_width=1)
            pred = res.predicted_form
            if gold_target:
                pred += " ✅ Match" if pred == gold_target else " ❌ Mismatch"
            warn = f"⚠️ Unseen individual tag tokens: {', '.join(res.unknown_features)}" if res.unknown_features else ""
            return "-", "-", pred, f"{res.decoder_score:.4f}", format_action_trace(res.action_trace), warn, cov_md, feat_explanation

    except Exception as e:
        return f"Error: {e}", "-", "-", "-", str(e), "", cov_md, feat_explanation


def sample_random_item(language_label, split_choice):
    iso = LANGUAGE_MAP.get(language_label, "eng")
    sample = engine.get_random_sample(iso, split=split_choice)
    if not sample:
        return "sing", "V;PST", "", explain_features("V;PST")
    return sample["lemma"], sample["features"], sample["target"], explain_features(sample["features"])


# Build markdown glossary table
GLOSSARY_MD = """
### 📖 UniMorph Tag Reference Guide

UniMorph tags represent morphological properties as standardized, semicolon-delimited dimensional feature bundles.

| Dimension | Tag | Name | Description & Example |
| :--- | :--- | :--- | :--- |
| **Part of Speech** | `V` | Verb | Action/state: *sing*, *jump*, *be* |
| | `N` | Noun | Entity: *apple*, *dog*, *water* |
| **Person** | `1` | 1st Person | Speaker (*I, we*) |
| | `2` | 2nd Person | Addressee (*you*) |
| | `3` | 3rd Person | Spoken about (*he, she, it, they*) |
| **Number** | `SG` | Singular | Single item (*child*) |
| | `PL` | Plural | Multiple items (*children*) |
| **Tense & Aspect** | `PRS` | Present | Present time (*walks*) |
| | `PST` | Past | Past time (*walked*, *sang*) |
| | `FUT` | Future | Future time (*ira* in French) |
| | `IPFV` | Imperfective | Ongoing past (*cantava* in Italian, *chantait* in French) |
| | `PFV` | Perfective | Completed past / Passé simple / Past historic (*cantò*, *chanta*) |
| **Mood & Finiteness** | `IND` | Indicative | Real fact or statement (*he goes*) |
| | `SBJV` | Subjunctive | Hypothetical, doubt, necessity (*chante*, *singe*) |
| | `COND` | Conditional | Conditional mood (*would sing*, *chanterait*) |
| | `IMP` | Imperative | Direct command (*Go!*) |
| | `NFIN` | Non-finite / Infinitive | Base dictionary infinitive form (*to go*) |
| | `V.PTCP` | Participle | Verbal adjective (*eaten*, *walking*) |
| | `V.CVB` | Converb / Gerund | Verbal adverb (*cantando* in Italian) |
| **Case** *(German)* | `NOM` | Nominative | Subject of clause (*der Hund*) |
| | `ACC` | Accusative | Direct object (*den Hund*) |
| | `DAT` | Dative | Indirect object (*dem Hund*) |
| | `GEN` | Genitive | Possessive case (*des Hundes*) |
| **Degree** *(German)* | `POS` | Positive | Base adjective degree (*schön*) |
"""

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

            with gr.Group():
                gr.Markdown("#### ℹ️ Feature Breakdown")
                feat_explanation_box = gr.Markdown(explain_features("V;PST"))

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

            with gr.Accordion("📚 UniMorph Feature Tag Glossary & Help", open=False):
                gr.Markdown(GLOSSARY_MD)

    gr.Markdown("### 💡 Click an Example to Load")
    example_rows = []
    for ex in CURATED_EXAMPLES:
        example_rows.append([ex["language"], "Compare Both (100 vs 1000)", ex["lemma"], ex["features"], ""])

    gr.Examples(
        examples=example_rows,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box, feat_explanation_box],
        fn=inflect,
        cache_examples=False,
    )

    feat_input.change(
        fn=explain_features,
        inputs=[feat_input],
        outputs=[feat_explanation_box],
    )

    sample_btn.click(
        fn=sample_random_item,
        inputs=[lang_dropdown, split_selector],
        outputs=[lemma_input, feat_input, gold_target_input, feat_explanation_box],
    ).then(
        fn=inflect,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box, feat_explanation_box],
    )

    run_btn.click(
        fn=inflect,
        inputs=[lang_dropdown, mode_radio, lemma_input, feat_input, gold_target_input],
        outputs=[out_100, score_100, out_1000, score_1000, trace_box, warning_box, coverage_box, feat_explanation_box],
    )

if __name__ == "__main__":
    demo.launch()
