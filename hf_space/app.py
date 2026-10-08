"""
Gradio Web Application for Neural Transducer Morphological Inflection.
Calm, pedagogical interface organized into:
- Overview (Purpose, research questions, getting started)
- Experiment (Configure -> Experiment -> Results with progressive disclosure)
- Documentation (UniMorph tag guide, model architecture, benchmark datasets)
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

DATASET_LINKS = {
    "English": {
        "train_100": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-low",
        "train_1000": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-medium",
        "test": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-uncovered-test",
    },
    "German": {
        "train_100": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-low",
        "train_1000": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-medium",
        "test": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-uncovered-test",
    },
    "French": {
        "train_100": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-low",
        "train_1000": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-medium",
        "test": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-uncovered-test",
    },
    "Italian": {
        "train_100": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-low",
        "train_1000": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-medium",
        "test": "https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-uncovered-test",
    },
}

TAG_DEFINITIONS = {
    # Part of Speech
    "V": "Verb",
    "N": "Noun",
    # Person
    "1": "1st Person",
    "2": "2nd Person",
    "3": "3rd Person",
    # Number
    "SG": "Singular",
    "PL": "Plural",
    # Tense & Aspect
    "PRS": "Present",
    "PST": "Past",
    "FUT": "Future",
    "IPFV": "Imperfective",
    "PFV": "Perfective",
    # Mood & Finiteness
    "IND": "Indicative",
    "SBJV": "Subjunctive",
    "COND": "Conditional",
    "IMP": "Imperative",
    "NFIN": "Infinitive",
    "V.PTCP": "Participle",
    "V.CVB": "Converb / Gerund",
    # Case (German)
    "NOM": "Nominative",
    "ACC": "Accusative",
    "DAT": "Dative",
    "GEN": "Genitive",
    # Degree (German)
    "POS": "Positive",
}


def explain_features_inline(features_str: str) -> str:
    """Concise inline tag decoder, e.g.: Verb · Indicative · Past · 3rd Person · Singular"""
    if not features_str or not features_str.strip():
        return ""
    tags = [t.strip() for t in features_str.strip().split(";") if t.strip()]
    if not tags:
        return ""
    parts = [TAG_DEFINITIONS.get(t, t) for t in tags]
    return " · ".join(parts)


def format_action_trace(trace) -> str:
    if not trace:
        return "No action trace available."
    rows = []
    for r in trace:
        src = f"'{r.source_char}' (pos {r.source_pos})" if r.source_char else "-"
        emitted = f"'{r.emitted_char}'" if r.emitted_char else "-"
        rows.append(f"Step {r.step:02d}: {r.operation:10s} | In: {src:16s} | Out: {emitted:8s} | {r.description}")
    return "\n".join(rows)


def build_coverage_markdown(lang_iso: str, lemma: str, features_str: str) -> str:
    if not lemma or not features_str:
        return ""
    cov = engine.check_training_coverage(lang_iso, lemma, features_str)
    low_lemma = "Observed" if cov["100"]["lemma_seen"] else "Unseen in training"
    med_lemma = "Observed" if cov["1000"]["lemma_seen"] else "Unseen in training"

    low_feat = "Observed" if cov["100"]["features_seen"] else "Unseen combination"
    med_feat = "Observed" if cov["1000"]["features_seen"] else "Unseen combination"

    md = f"""
| Input Element | 100 Examples (Low-Resource) | 1,000 Examples (Medium-Resource) |
| :--- | :--- | :--- |
| **Lemma** (`{lemma}`) | {low_lemma} | {med_lemma} |
| **Features** (`{features_str}`) | {low_feat} | {med_feat} |
"""
    return md.strip()


def run_experiment(language_label: str, lemma: str, features_str: str, gold_target: str = ""):
    """Runs prediction for both 100 and 1,000 regimes, comparing against gold target if provided."""
    inline_desc = explain_features_inline(features_str)

    if not lemma or not features_str:
        return (
            "Please provide a lemma and features.",
            "",
            "Please provide a lemma and features.",
            "",
            "",
            "No action trace available.",
            inline_desc,
        )

    iso = LANGUAGE_MAP.get(language_label, "eng")
    lemma = lemma.strip()
    features_str = features_str.strip()
    gold_target = (gold_target or "").strip()

    cov_md = build_coverage_markdown(iso, lemma, features_str)

    try:
        res_100 = engine.predict(iso, "100", lemma, features_str, beam_width=1)
        res_1000 = engine.predict(iso, "1000", lemma, features_str, beam_width=1)

        pred_100 = res_100.predicted_form
        pred_1000 = res_1000.predicted_form

        status_100 = ""
        status_1000 = ""
        if gold_target:
            status_100 = "Match" if pred_100 == gold_target else "Mismatch"
            status_1000 = "Match" if pred_1000 == gold_target else "Mismatch"

        trace_text = (
            f"=== 100 Examples ===\n{format_action_trace(res_100.action_trace)}\n\n"
            f"=== 1,000 Examples ===\n{format_action_trace(res_1000.action_trace)}"
        )

        return (
            pred_100,
            status_100,
            pred_1000,
            status_1000,
            cov_md,
            trace_text,
            inline_desc,
        )
    except Exception as e:
        return (
            f"Error: {e}",
            "",
            f"Error: {e}",
            "",
            "",
            str(e),
            inline_desc,
        )


def sample_test_and_run(language_label: str):
    """Samples a random item from the test set and immediately runs the models."""
    iso = LANGUAGE_MAP.get(language_label, "eng")
    sample = engine.get_random_sample(iso, split="test")
    if not sample:
        return "sing", "V;PST", "", "", "", "", "", "", "No test samples available.", "Verb · Past"

    lemma = sample["lemma"]
    features_str = sample["features"]
    gold_target = sample["target"]

    pred_100, stat_100, pred_1000, stat_1000, cov_md, trace_text, inline_desc = run_experiment(
        language_label, lemma, features_str, gold_target
    )

    return (
        lemma,
        features_str,
        gold_target,
        pred_100,
        stat_100,
        pred_1000,
        stat_1000,
        cov_md,
        trace_text,
        inline_desc,
    )


# Documentation markdown constants
GLOSSARY_MD = """
### UniMorph Morphological Features

Morphological features in UniMorph are represented as standardized, semicolon-delimited dimensional tags.

| Dimension | Tag | Category | Description & Example |
| :--- | :--- | :--- | :--- |
| **Part of Speech** | `V` | Verb | Action or state (*sing*, *walk*, *be*) |
| | `N` | Noun | Entity or concept (*child*, *apple*) |
| **Person** | `1` | 1st Person | Speaker (*I, we*) |
| | `2` | 2nd Person | Addressee (*you*) |
| | `3` | 3rd Person | Spoken about (*he, she, it, they*) |
| **Number** | `SG` | Singular | Single item (*book*) |
| | `PL` | Plural | Multiple items (*books*) |
| **Tense & Aspect** | `PRS` | Present | Present time (*walks*) |
| | `PST` | Past | Past time (*walked*, *sang*) |
| | `FUT` | Future | Future time (*ira* in French) |
| | `IPFV` | Imperfective | Continuous or habitual past (*cantava*, *chantait*) |
| | `PFV` | Perfective | Completed past / Passé simple (*chanta*) |
| **Mood & Finiteness** | `IND` | Indicative | Objective fact (*he goes*) |
| | `SBJV` | Subjunctive | Hypothetical, doubt, or wish (*chante*) |
| | `COND` | Conditional | Conditional action (*would walk*, *chanterait*) |
| | `IMP` | Imperative | Direct command (*Go!*) |
| | `NFIN` | Infinitive | Base dictionary non-finite form (*to walk*) |
| | `V.PTCP` | Participle | Verbal adjective (*eaten*, *walking*) |
| | `V.CVB` | Converb / Gerund | Verbal adverb (*cantando* in Italian) |
| **Case** *(German)* | `NOM` | Nominative | Subject of clause (*der Hund*) |
| | `ACC` | Accusative | Direct object (*den Hund*) |
| | `DAT` | Dative | Indirect object (*dem Hund*) |
| | `GEN` | Genitive | Possessive case (*des Hundes*) |
| **Degree** *(German)* | `POS` | Positive | Base adjective degree (*schön*) |

---

### Model Architecture & Training

The models implemented here are **transition-based neural string transducers** following Makarov and Clematide (2020):
- **Encoder:** Bidirectional LSTM representing character sequences and feature embeddings.
- **Decoder:** Monotonic transition-based decoder that emits character edit actions: `COPY`, `DELETE`, `INSERT(c)`, and `SUBSTITUTE(c)`.
- **Training Strategy:** Imitation learning using expert policies derived from Stochastic Edit Distance (SED) alignments.
- **Decoding Mode:** Greedy decoding (`beam_width = 1`) for clean, deterministic evaluation.

---

### Benchmark Data: CoNLL-SIGMORPHON 2017 Task 1

The models are trained and evaluated on official splits from the CoNLL-SIGMORPHON 2017 Shared Task 1:

| Language | Low-Resource (100) | Medium-Resource (1,000) | Test Set (1,000) |
| :--- | :--- | :--- | :--- |
| **English** | [english-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-low) | [english-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-medium) | [english-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-uncovered-test) |
| **German** | [german-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-low) | [german-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-medium) | [german-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-uncovered-test) |
| **French** | [french-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-low) | [french-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-medium) | [french-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-uncovered-test) |
| **Italian** | [italian-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-low) | [italian-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-medium) | [italian-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-uncovered-test) |
"""


CUSTOM_CSS = """
/* Calm academic styling */
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
}
.overview-card {
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    padding: 16px;
}
.benchmark-badge {
    font-size: 0.9em;
    font-weight: 500;
}
"""

with gr.Blocks(title="Morphological Inflection", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    with gr.Tabs() as tabs:

        # ----------------------------------------------------
        # Tab 1: Overview
        # ----------------------------------------------------
        with gr.Tab("Overview", id="tab_overview"):
            gr.Markdown(
                """
                # Morphological Inflection
                What happens when a neural transducer learns morphology from **100 examples** rather than **1,000**?

                This demonstrator investigates how transition-based neural string transducers (Makarov & Clematide, 2020) generalize morphological inflection across four European languages under low-resource versus medium-resource supervision.
                """
            )

            with gr.Row():
                with gr.Column(elem_classes=["overview-card"]):
                    gr.Markdown("### Explore\nGenerate inflected forms across English, German, French, and Italian from lemma and feature inputs.")
                with gr.Column(elem_classes=["overview-card"]):
                    gr.Markdown("### Compare\nExamine side-by-side predictions from models trained on 100 versus 1,000 examples.")
                with gr.Column(elem_classes=["overview-card"]):
                    gr.Markdown("### Understand\nInspect whether inputs were observed during training and how edit actions construct the output.")

            with gr.Row():
                start_btn = gr.Button("Start Exploring →", variant="primary", size="lg")

            gr.Markdown(
                """
                *Data: Official CoNLL-SIGMORPHON 2017 Shared Task 1 benchmark data.*
                *Detailed documentation, UniMorph feature tag guides, and data split links are available in the **Documentation** tab.*
                """
            )

        # ----------------------------------------------------
        # Tab 2: Experiment
        # ----------------------------------------------------
        with gr.Tab("Experiment", id="tab_experiment"):

            # Area A: Configure
            gr.Markdown("### Configure")
            with gr.Row():
                lang_dropdown = gr.Dropdown(
                    label="Language",
                    choices=list(LANGUAGE_MAP.keys()),
                    value="English",
                    scale=2,
                )
                sample_btn = gr.Button(
                    "Sample Test Example",
                    variant="secondary",
                    scale=1,
                )

            # Area B: Experiment Inputs
            gr.Markdown("### Experiment")
            with gr.Row():
                lemma_input = gr.Textbox(
                    label="Lemma",
                    placeholder="e.g. sing",
                    value="sing",
                    scale=1,
                )
                with gr.Column(scale=2):
                    feat_input = gr.Textbox(
                        label="Features (UniMorph)",
                        placeholder="e.g. V;PST",
                        value="V;PST",
                    )
                    feat_inline_desc = gr.Markdown(
                        value=explain_features_inline("V;PST"),
                    )

            with gr.Row():
                gold_target_input = gr.Textbox(
                    label="Benchmark Reference (Target Form)",
                    placeholder="Populated when sampling from test set",
                    value="",
                    interactive=False,
                )

            with gr.Row():
                run_btn = gr.Button("Generate Inflection", variant="primary")

            # Area C: Results
            gr.Markdown("### Results")
            with gr.Row():
                with gr.Column():
                    out_100 = gr.Textbox(label="100 Examples (Low-Resource)", interactive=False)
                    stat_100 = gr.Textbox(label="Status (100)", show_label=False, interactive=False)
                with gr.Column():
                    out_1000 = gr.Textbox(label="1,000 Examples (Medium-Resource)", interactive=False)
                    stat_1000 = gr.Textbox(label="Status (1,000)", show_label=False, interactive=False)

            with gr.Accordion("Training Data Exposure", open=False):
                coverage_box = gr.Markdown(
                    "Generate an inflection or sample a test item to inspect training exposure."
                )

            with gr.Accordion("Transduction Steps", open=False):
                trace_box = gr.Code(label="Action Sequence", language="markdown", interactive=False)

            # Quick clickable examples
            gr.Markdown("#### Examples")
            example_rows = []
            for ex in CURATED_EXAMPLES:
                example_rows.append([ex["language"], ex["lemma"], ex["features"], ""])

            gr.Examples(
                examples=example_rows,
                inputs=[lang_dropdown, lemma_input, feat_input, gold_target_input],
                outputs=[out_100, stat_100, out_1000, stat_1000, coverage_box, trace_box, feat_inline_desc],
                fn=run_experiment,
                cache_examples=False,
            )

        # ----------------------------------------------------
        # Tab 3: Documentation
        # ----------------------------------------------------
        with gr.Tab("Documentation", id="tab_docs"):
            gr.Markdown(GLOSSARY_MD)

    # ----------------------------------------------------
    # Event Bindings
    # ----------------------------------------------------
    # Start Exploring button switches to Experiment tab
    start_btn.click(
        fn=lambda: gr.Tabs(selected="tab_experiment"),
        outputs=[tabs],
    )

    # Dynamic feature explanation on change
    feat_input.change(
        fn=explain_features_inline,
        inputs=[feat_input],
        outputs=[feat_inline_desc],
    )

    # Sampling button immediately selects test item and executes the experiment
    sample_btn.click(
        fn=sample_test_and_run,
        inputs=[lang_dropdown],
        outputs=[
            lemma_input,
            feat_input,
            gold_target_input,
            out_100,
            stat_100,
            out_1000,
            stat_1000,
            coverage_box,
            trace_box,
            feat_inline_desc,
        ],
    )

    # Manual run button
    run_btn.click(
        fn=run_experiment,
        inputs=[lang_dropdown, lemma_input, feat_input, gold_target_input],
        outputs=[
            out_100,
            stat_100,
            out_1000,
            stat_1000,
            coverage_box,
            trace_box,
            feat_inline_desc,
        ],
    )

if __name__ == "__main__":
    demo.launch()
