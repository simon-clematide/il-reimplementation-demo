"""
Gradio Web Application for Neural Transducer Morphological Inflection.
Focused exploration of training-size effects with a precomputed test gallery,
live custom inputs, and progressively disclosed model explanations.
"""

import json
import inspect
from html import escape
from types import SimpleNamespace
import os
import gradio as gr
from inference import TransducerSpaceEngine

# Initialize inference engine
engine = TransducerSpaceEngine()

LANGUAGE_MAP = {
    "English": "eng",
    "German": "deu",
    "French": "fra",
    "Italian": "ita",
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


def _pluralize_op(op: str, count: int) -> str:
    forms = {
        "COPY": ("copy", "copies"),
        "SUBST": ("substitution", "substitutions"),
        "INSERT": ("insertion", "insertions"),
        "DELETE": ("deletion", "deletions"),
    }
    sg, pl = forms.get(op, (op.lower(), op.lower() + "s"))
    return f"{count} {sg if count == 1 else pl}"


def _summarize_trace(trace) -> str:
    if not trace:
        return "No actions"
    counts = {}
    for s in trace:
        if s.operation == "END":
            continue
        op = "SUBST" if s.operation == "SUBSTITUTE" else s.operation
        counts[op] = counts.get(op, 0) + 1
    parts = [_pluralize_op(k, v) for k, v in counts.items()]
    return ", ".join(parts) if parts else "no edit actions"


def _format_action_cell(step) -> str:
    if not step:
        return "—"
    op = step.operation
    src = step.source_char
    emitted = step.emitted_char
    if op == "COPY":
        return f"COPY (`{emitted}`)"
    elif op == "SUBSTITUTE":
        return f"**SUBST** (`{src}` → `{emitted}`)"
    elif op == "INSERT":
        return f"INSERT (`{emitted}`)"
    elif op == "DELETE":
        return f"DELETE (`{src}`)"
    elif op == "END":
        return "*END*"
    return f"{op}"


def format_comparative_action_trace(trace_100, trace_1000, trace_10000) -> str:
    if not trace_100 and not trace_1000 and not trace_10000:
        return "Generate an inflection or sample a test item to inspect transduction steps."

    max_len = max(len(trace_100 or []), len(trace_1000 or []), len(trace_10000 or []))

    summary_md = (
        f"- **100 Examples:** {_summarize_trace(trace_100)}\n"
        f"- **1,000 Examples:** {_summarize_trace(trace_1000)}\n"
        f"- **10,000 Examples:** {_summarize_trace(trace_10000)}\n"
    )

    rows = [
        "| Step | 100 Examples (Low) | 1,000 Examples (Medium) | 10,000 Examples (High) | Alignment |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]

    for i in range(max_len):
        s100 = trace_100[i] if i < len(trace_100 or []) else None
        s1000 = trace_1000[i] if i < len(trace_1000 or []) else None
        s10000 = trace_10000[i] if i < len(trace_10000 or []) else None

        c100 = _format_action_cell(s100)
        c1000 = _format_action_cell(s1000)
        c10000 = _format_action_cell(s10000)

        # Check alignment across active steps
        keys = set()
        for s in (s100, s1000, s10000):
            if s is not None:
                keys.add((s.operation, s.source_char, s.emitted_char))
            else:
                keys.add(None)

        alignment = "✓ Match" if len(keys) == 1 else "⚠️ Diverged"

        rows.append(f"| {i+1} | {c100} | {c1000} | {c10000} | {alignment} |")

    return f"{summary_md}\n" + "\n".join(rows)


def build_coverage_markdown(lang_iso: str, lemma: str, features_str: str) -> str:
    if not lemma or not features_str:
        return ""
    cov = engine.check_training_coverage(lang_iso, lemma, features_str)
    low_lemma = "Observed" if cov.get("100", {}).get("lemma_seen") else "Unseen in training"
    med_lemma = "Observed" if cov.get("1000", {}).get("lemma_seen") else "Unseen in training"
    high_lemma = "Observed" if cov.get("10000", {}).get("lemma_seen") else "Unseen in training"

    low_feat = "Observed" if cov.get("100", {}).get("features_seen") else "Unseen combination"
    med_feat = "Observed" if cov.get("1000", {}).get("features_seen") else "Unseen combination"
    high_feat = "Observed" if cov.get("10000", {}).get("features_seen") else "Unseen combination"

    md = f"""
| Input Element | 100 Examples (Low) | 1,000 Examples (Medium) | 10,000 Examples (High) |
| :--- | :--- | :--- | :--- |
| **Lemma** (`{lemma}`) | {low_lemma} | {med_lemma} | {high_lemma} |
| **Features** (`{features_str}`) | {low_feat} | {med_feat} | {high_feat} |
"""
    return md.strip()


# Documentation markdown sections
UNIMORPH_DOC_MD = """
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
"""

ARCHITECTURE_DOC_MD = """
The models implemented here are **transition-based neural string transducers** following Makarov and Clematide (2020):
- **Encoder:** Bidirectional LSTM representing character sequences and feature embeddings.
- **Decoder:** Monotonic transition-based decoder that emits character edit actions: `COPY`, `DELETE`, `INSERT(c)`, and `SUBSTITUTE(c)`.
- **Training Strategy:** Imitation learning using expert policies derived from Stochastic Edit Distance (SED) alignments.
- **Decoding Mode:** Greedy decoding (`beam_width = 1`) for clean, deterministic evaluation.
"""

BENCHMARK_DOC_MD = """
The models are trained and evaluated on official splits from the CoNLL-SIGMORPHON 2017 Shared Task 1:

| Language | Low-Resource (100) | Medium-Resource (1,000) | High-Resource (10,000) | Test Set (1,000) |
| :--- | :--- | :--- | :--- | :--- |
| **English** | [english-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-low) | [english-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-medium) | [english-train-high](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-train-high) | [english-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/english-uncovered-test) |
| **German** | [german-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-low) | [german-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-medium) | [german-train-high](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-train-high) | [german-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/german-uncovered-test) |
| **French** | [french-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-low) | [french-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-medium) | [french-train-high](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-train-high) | [french-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/french-uncovered-test) |
| **Italian** | [italian-train-low](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-low) | [italian-train-medium](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-medium) | [italian-train-high](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-train-high) | [italian-uncovered-test](https://raw.githubusercontent.com/sigmorphon/conll2017/master/all/task1/italian-uncovered-test) |
"""



# The gallery contains real, precomputed test predictions; custom inputs run live.
GALLERY_PATH = os.path.join(os.path.dirname(__file__), "exploration_examples.json")
with open(GALLERY_PATH, encoding="utf-8") as f:
    GALLERY = json.load(f)["examples"]
FOCUSES = ["More data helps", "Models disagree", "Largest model misses", "All models correct", "All examples"]


def gallery_items(language, focus):
    return [s for s in GALLERY if s["language"] == LANGUAGE_MAP[language] and focus in s["groups"]]


def gallery_choices(language, focus):
    return [(f'{s["lemma"]} → {s["target"]} · {s["features"]}', str(GALLERY.index(s)))
            for s in gallery_items(language, focus)]


def present(language, lemma, features, gold, results, source):
    task = f"## {escape(lemma)} → {escape(explain_features_inline(features))}"
    task += f"\n\n{escape(language)} · `{escape(features)}`"
    if gold:
        task += f" · Reference: **{escape(gold)}**"
    rows = []
    for size, result in zip(["100", "1,000", "10,000"], results):
        form = result.predicted_form
        matches = form == gold
        status = (f'<span class="match-status {"correct" if matches else "incorrect"}" '
                  f'role="img" aria-label="{"Correct" if matches else "Incorrect"}" '
                  f'title="{"Correct" if matches else "Incorrect"}">{"✓" if matches else "✗"}</span>') if gold else ""
        status_cell = f'<td>{status}</td>' if gold else ''
        rows.append(f'<tr><td>{size}</td><td class="prediction">{escape(form)}</td>{status_cell}</tr>')
    reference_header = '<th>Reference match</th>' if gold else ''
    table = ('<table class="comparison"><thead><tr><th>Training examples</th><th>Prediction</th>' + reference_header + '</tr></thead>'
             '<tbody>' + ''.join(rows) + '</tbody></table>')
    return task, table, source, build_coverage_markdown(LANGUAGE_MAP[language], lemma, features), format_comparative_action_trace(*[r.action_trace for r in results])


def show_gallery(example_id):
    if example_id is None:
        return "## No examples in this group", "", "Choose another focus or try your own input.", "", ""
    sample = GALLERY[int(example_id)]
    results = [SimpleNamespace(**dict(r, action_trace=[SimpleNamespace(**a) for a in r["action_trace"]])) for r in sample["results"]]
    language = next(label for label, iso in LANGUAGE_MAP.items() if iso == sample["language"])
    return present(language, sample["lemma"], sample["features"], sample["target"], results,
                   "Precomputed test example · Selected cases illustrate behavior, not overall accuracy.")


def change_gallery(language, focus):
    choices = gallery_choices(language, focus)
    selected = choices[0][1] if choices else None
    return (gr.Dropdown(choices=choices, value=selected), *show_gallery(selected))


def custom_run(language, lemma, features):
    lemma, features = lemma.strip(), features.strip()
    if not lemma or not features:
        raise gr.Error("Enter a word and morphological features.")
    try:
        results = [engine.predict(LANGUAGE_MAP[language], size, lemma, features) for size in ["100", "1000", "10000"]]
        return present(language, lemma, features, "", results, "Live prediction · No reference answer supplied.")
    except Exception as exc:
        raise gr.Error(f"Could not generate predictions: {exc}") from exc


# Feature bundles are already precomputed from each training split in data_index.json.
# Keep only observed full bundles, rather than constructing a Cartesian product.
FORM_LABELS = {"V": "Verb", "N": "Noun", "V.PTCP": "Participle", "V.CVB": "Converb / gerund"}


def feature_group(bundle):
    tags = bundle.split(";")
    return next((tag for tag in ("V.PTCP", "V.CVB") if tag in tags), tags[0])


FEATURE_CATALOG = {}
for iso, data in engine._load_data_index().items():
    groups = {}
    for bundle in data["training_coverage"]["10000"]["features"]:
        groups.setdefault(feature_group(bundle), []).append(bundle)
    FEATURE_CATALOG[iso] = groups


def pos_choices(language):
    return [(FORM_LABELS.get(tag, tag), tag) for tag in sorted(FEATURE_CATALOG[LANGUAGE_MAP[language]], key=lambda tag: (tag != "V", tag))]


def form_choices(language, pos):
    return [(f"{explain_features_inline(bundle)} · {bundle}", bundle)
            for bundle in FEATURE_CATALOG[LANGUAGE_MAP[language]].get(pos, [])]


def reset_forms(language, pos):
    choices = form_choices(language, pos)
    value = next((bundle for _, bundle in choices if bundle == "V;PST"), choices[0][1] if choices else "")
    return gr.Dropdown(choices=choices, value=value), value, explain_features_inline(value)


def reset_language_forms(language):
    choices = pos_choices(language)
    pos = choices[0][1]
    return gr.Dropdown(choices=choices, value=pos), *reset_forms(language, pos)


CUSTOM_HEAD = """<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Source+Sans+Pro:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap">"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Source+Sans+Pro:ital,wght@0,300;0,400;0,600;0,700;1,400&display=swap');

/* Calm academic aesthetic with Source Sans 3 / Source Sans Pro */
html, body, .gradio-container, button, input, select, textarea, .prose {
    font-family: 'Source Sans 3', 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}
.gradio-container {width:100% !important; max-width:none !important; margin:0 !important; padding:20px 24px !important; box-sizing:border-box;}
.gradio-container .main, .gradio-container main {padding:0 !important;}
.prose {font-size: 17px !important; line-height: 1.55 !important;}
.prose h1 {font-size: 32px !important; letter-spacing: -.025em;}
.prose h2 {font-size: 24px !important;}
.comparison {border:0 !important; width:100%; border-collapse:collapse; font-size:16px; margin:8px 0 20px;}
.comparison th {text-align:left; font-weight:500; color:var(--body-text-color-subdued);}
.gradio-container .comparison td, .gradio-container .comparison th {border-left:0 !important; border-right:0 !important; border-top:0 !important; padding:12px 8px; border-bottom:1px solid var(--border-color-primary);}
.comparison .match-status {font-size:24px; font-weight:700;}
.comparison .match-status.correct {color:#15803d;}
.comparison .match-status.incorrect {color:#dc2626;}
.comparison .prediction {font-size:26px; font-weight:600;}
@media(max-width:600px) {
  .gradio-container .comparison td, .gradio-container .comparison th {border-left:0 !important; border-right:0 !important; border-top:0 !important; padding:10px 4px; font-size:14px;}
  .comparison .prediction {font-size:22px; overflow-wrap:anywhere;}
}
"""

STYLING = dict(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate", text_size="lg"), css=CUSTOM_CSS, head=CUSTOM_HEAD)
# Gradio 6 moved styling from Blocks to launch; retain Space compatibility.
STYLE_ON_BLOCKS = "theme" in inspect.signature(gr.Blocks.__init__).parameters
with gr.Blocks(title="Morphological Inflection", fill_width=True, **(STYLING if STYLE_ON_BLOCKS else {})) as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Overview", id="overview"):
            gr.Markdown("""
# How much data does it take to learn a language pattern?
Compare neural models trained on **100, 1,000, and 10,000 examples**. See when more data helps—and when it doesn't.

## The task: turn a word into the form you need
Given a dictionary form and grammatical features, predict an inflected word: **sing + past tense → sang**. Explore English, German, French, and Italian, from regular endings to irregular stem changes.

## The models: build words one edit at a time
These transition-based neural transducers learn to **copy, delete, insert, or substitute characters**. Each language has three separately trained models, using the same architecture with different amounts of supervision. Training uses imitation learning with an edit-distance expert.

## The exploration: look beyond a correct answer
Start with test examples where model predictions differ. Compare their outputs, inspect how they build a word, and check whether the word or feature combination appeared in training. Then try your own input.

Can a small model learn a productive rule? Does more data resolve an irregular form? Can the largest model still get it wrong?
""")
            start = gr.Button("Explore the models →", variant="primary")
            gr.Markdown("Based on Makarov & Clematide (2020) · CoNLL–SIGMORPHON 2017 Task 1. Details and data links in Documentation.")
        with gr.Tab("Explore", id="explore"):
            gr.Markdown("# Compare inflections")
            with gr.Row():
                language = gr.Dropdown(list(LANGUAGE_MAP), value="English", label="Language", scale=1, min_width=140)
                focus = gr.Dropdown(FOCUSES, value=FOCUSES[0], label="Explore cases where…", scale=2, min_width=200)
            initial_choices = gallery_choices("English", FOCUSES[0])
            initial_id = initial_choices[0][1] if initial_choices else None
            example = gr.Dropdown(initial_choices, value=initial_id, label="Test example")
            initial = show_gallery(initial_id)
            task = gr.Markdown(initial[0])
            comparison = gr.HTML(initial[1])
            source = gr.Markdown(initial[2])
            with gr.Accordion("How was the word built?", open=False):
                gr.Markdown("Compare edit actions across models. Step numbers are decoding steps; models can consume input at different rates.")
                trace = gr.Markdown(initial[4])
            with gr.Accordion("Was this input seen in training?", open=False):
                coverage = gr.Markdown(initial[3])
            with gr.Accordion("Sample another test example", open=False):
                sample_filter = gr.Radio(["Different (Input ≠ Output)", "Same (Input = Output)", "Any"], value="Different (Input ≠ Output)", label="Word and reference")
                sample_button = gr.Button("Sample and run live")
        with gr.Tab("Try your own word", id="custom"):
            gr.Markdown("# Try your own word\nCompare live predictions. There is no reference answer for custom inputs.")
            custom_language = gr.Dropdown(list(LANGUAGE_MAP), value="English", label="Language")
            gr.Markdown("Choose a form observed in this language’s 10,000-example training set. Availability depends on the dataset; not every form applies to every word.")
            lemma = gr.Textbox(label="Dictionary form", value="sing")
            with gr.Row():
                pos = gr.Dropdown(pos_choices("English"), value="V", label="Part of speech / word form", scale=1, min_width=160)
                form = gr.Dropdown(form_choices("English", "V"), value="V;PST", label="Grammatical form", scale=2, min_width=220)
            description = gr.Markdown(explain_features_inline("V;PST"))
            with gr.Accordion("Edit UniMorph tags manually", open=False):
                features = gr.Textbox(label="Features used for prediction", value="V;PST", info="Overrides the selected form. Separate tags with semicolons.")
            run = gr.Button("Generate inflections", variant="primary")
            custom_task = gr.Markdown()
            custom_comparison = gr.HTML()
            custom_source = gr.Markdown()
            with gr.Accordion("How was the word built?", open=False):
                custom_trace = gr.Markdown("Generate inflections to inspect the edit actions.")
            with gr.Accordion("Was this input seen in training?", open=False):
                custom_coverage = gr.Markdown("Generate inflections to inspect training exposure.")
        with gr.Tab("Documentation", id="docs"):
            gr.Markdown("# About the experiment")
            with gr.Accordion("UniMorph feature guide", open=False):
                gr.Markdown(UNIMORPH_DOC_MD)
            with gr.Accordion("Models and training", open=False):
                gr.Markdown(ARCHITECTURE_DOC_MD)
                gr.Markdown("[Makarov & Clematide (2020)](https://aclanthology.org/2020.sigmorphon-1.19/) · Greedy decoding for every comparison. Training size is a comparison across these trained models, not a guarantee of improvement on every word.")
            with gr.Accordion("Benchmark and data", open=False):
                gr.Markdown(BENCHMARK_DOC_MD)
                gr.Markdown("The gallery uses a deterministic subset of held-out test items whose input differs from the reference. Cases are grouped by actual predictions. These selected examples are not an accuracy estimate. The live sampler also supports unchanged forms.")

    outputs = [task, comparison, source, coverage, trace]
    start.click(lambda: gr.Tabs(selected="explore"), outputs=tabs)
    language.change(change_gallery, [language, focus], [example, *outputs])
    focus.change(change_gallery, [language, focus], [example, *outputs])
    example.input(show_gallery, example, outputs)
    custom_language.change(reset_language_forms, custom_language, [pos, form, features, description])
    pos.input(reset_forms, [custom_language, pos], [form, features, description])
    form.input(lambda value: (value, explain_features_inline(value)), form, [features, description])
    features.input(explain_features_inline, features, description)
    run.click(custom_run, [custom_language, lemma, features], [custom_task, custom_comparison, custom_source, custom_coverage, custom_trace])

    def sample_live(language, sample_type):
        filter_type = "different" if "Different" in sample_type else "same" if "Same" in sample_type else "any"
        sample = engine.get_random_sample(LANGUAGE_MAP[language], split="test", filter_type=filter_type)
        if not sample:
            raise gr.Error("No test examples available for this selection.")
        results = [engine.predict(LANGUAGE_MAP[language], size, sample["lemma"], sample["features"]) for size in ["100", "1000", "10000"]]
        return present(language, sample["lemma"], sample["features"], sample["target"], results, "Live prediction · Random held-out test example.")

    sample_button.click(sample_live, [language, sample_filter], outputs)

if __name__ == "__main__":
    demo.launch(**({} if STYLE_ON_BLOCKS else STYLING))
