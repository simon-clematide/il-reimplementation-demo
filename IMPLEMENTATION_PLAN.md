# Implementation Plan v4: Morphological Inflection Demo for `neural_transducer`

**Target:** `~/pj/2026/il-reimplementation-demo/`
**Library:** `~/pj/2023/il-reimplementation` (package `trans`), pinned to commit `f71833192acd505bc1d3ba47d75798cfb778fc3d` (on `origin/development`, i.e. `github.com/simon-clematide/il-reimplementation`)
**HF account:** `simon-clmtd` (existing sibling Space `romansh-morphology` is used as the config template)
**Status:** v4 (v3 + decisions: names OK, public repo, seed 42, MPS; measured training cost; dependency correction). Adds concrete Hugging Face resources, the user-action checklist, deployment mechanics, and licensing. v2 content (benchmark, loader, training config) is kept and tightened.

## 0. Hugging Face Resources and Your Actions

### 0.1 Resources (names proposed, change if you prefer)
| Kind | ID | Purpose |
|---|---|---|
| Space (Gradio, public) | `simon-clmtd/morphological-inflection` | Interactive demo. Code only, no weights. |
| Model repo (public) | `simon-clmtd/neural-transducer-inflection` | 8 bundles, one subfolder each (`eng_100/` … `ita_1000/`), plus a model card. Release tags `v1.0`, … |
| GitHub repo (public) | `simon-clematide/il-reimplementation-demo` | Source of truth for scripts, notebook, `demo_core`, Space code. Needed for the Colab "Open in Colab" badge. |

One model repo with subfolders (not 8 repos) keeps the revision/tag a single pin for the whole release. `snapshot_download(..., allow_patterns="<bundle>/*")` still fetches only one bundle.

### 0.2 Checklist, in the order needed
| When | Who | Action |
|---|---|---|
| Before Phase 1 | You | Create the GitHub repo `simon-clematide/il-reimplementation-demo` (empty, public). I work in the local folder and you push. |
| Before Phase 4 | You | Create the model repo: `hf repo create simon-clmtd/neural-transducer-inflection --type model` |
| Before Phase 5 | You | Create the Space: `hf repo create simon-clmtd/morphological-inflection --type space --space-sdk gradio` |
| Phase 4 | You | Confirm a **write** token is available. Your CLI is already logged in as `simon-clmtd` (checked with `hf auth whoami`). Do not paste tokens into chat or commit them; the scripts read the CLI login or `HF_TOKEN`. |
| Phase 4 | Me, with your OK | Run `scripts/upload_hf.py` (the upload is outward-facing, so I will ask before running it). |
| Phase 5 | Me, with your OK | Push `hf_space/` to the Space. |
| Phase 6 | You | Open the notebook in a fresh Colab runtime for the final check (I cannot run Colab). |
| Phase 7 | You | Pin the Space or add it to a collection (optional). |

I will not create or modify any remote resource without asking first.

### 0.3 Runtime needs no secrets
Model repo and Space are public, so the Space downloads bundles anonymously. No token is stored in the Space.

## 1. Goals
1. **Space:** inflect a lemma given UniMorph tags in English, German, French and Italian, with models trained on 100 vs 1000 examples, side by side.
2. **Colab notebook:** inference with pretrained bundles by default; trains one language and one regime end to end; train-all is optional.
3. Everything reproducible from a manifest and one command per stage.

## 2. Benchmark Contract
### 2.1 Source (pinned)
- `https://github.com/sigmorphon/conll2017` at commit `519bb1677a7c747b3863d37069ffd395062288d8`. Raw URLs always use the SHA, never `master`.
- Languages: `english`, `german`, `french`, `italian`.
- The "800 train / 100 dev" numbers in `g2p.ipynb` belong to the 2021 G2P task and are not used here.

### 2.2 Splits
| Role | File | Lines (verified, all 4 langs) |
|---|---|---|
| Train, low | `all/task1/<lang>-train-low` | 100 |
| Train, medium | `all/task1/<lang>-train-medium` | 1000 |
| Dev (model selection) | `all/task1/<lang>-dev` | 1000 |
| Gold test | `answers/task1/<lang>-uncovered-test` | counted by the validator |

`*-covered-test` has no targets and is not scored. Format `lemma<TAB>form<TAB>tag1;tag2`, e.g. `stodge	stodged	V;PST`.

### 2.3 Sampling protocol
Official files as released, no own subsampling. The validator reports whether `train-low ⊂ train-medium` and the overlap size; the README states the result so the comparison is described accurately. `data/MANIFEST.json` (checked in) holds, per file, SHA-256, line count, URL and commit.

### 2.4 Validation (`scripts/validate_data.py`, runs before any training)
Files exist and are UTF-8 (NFC check); exactly 3 columns, non-empty lemma, form and tags; counts match the manifest; duplicate triples per split; train/dev/test leakage on triples and lemmas; tag inventory per language, including tags in dev/test unseen in train (expected UNK rate). Exit nonzero on missing file, bad columns or count mismatch.

### 2.5 Data licensing
The SIGMORPHON data is derived from UniMorph/Wiktionary. Before publishing, I will read the license in the conll2017 repo and UniMorph, record it in `data/README.md`, and set the model-card license and attribution to match (likely a share-alike license). The raw data is not redistributed by us: scripts download it from the pinned source.

## 3. Repository Layout
```text
il-reimplementation-demo/
├── README.md  IMPLEMENTATION_PLAN.md  REPRODUCIBILITY.md (generated)  Makefile
├── data/MANIFEST.json  data/README.md      # raw data gitignored
├── configs/train.yaml                       # frozen hyperparameters
├── scripts/  fetch_data.py validate_data.py train_bundle.py eval_bundle.py
│             package_bundle.py upload_hf.py sync_space.py
├── demo_core/  loader.py predict.py features.py   # only inference code
├── hf_space/   app.py requirements.txt README.md (HF YAML card) [demo_core/ synced in]
├── notebooks/morphological_inflection.ipynb
├── model_card/README.md                     # card for the model repo
└── tests/
```
`scripts/sync_space.py` copies `demo_core/` into `hf_space/demo_core/`, so the Space repo is self-contained while there is a single source in GitHub. The sync output is gitignored in GitHub and committed only in the Space repo.

## 4. Training Configuration (`configs/train.yaml`)
Flags checked against `trans/train.py`, `encoders.py`, `optimizers.py`. † = starting point, tuned once on the pilot using dev accuracy only, then frozen for all eight bundles (no per-language tuning).

| Setting | Flag | Value |
|---|---|---|
| Seed | `--pytorch-seed` | **42**, single seed for all 8 bundles (decided; no multi-seed run) |
| Encoder | `--enc-type lstm --enc-hidden-dim 200 --enc-layers 1` † | BiLSTM |
| Enc dropout | `--enc-output-dropout 0.3 --enc-output-dropout-type locked` † | `--enc-dropout` only acts when layers > 1 |
| Dims | `--char-dim --feat-dim --action-dim --dec-hidden-dim --dec-layers` | 100 / 100 / 100 / 200 / 1 † |
| Optimizer | `--optimizer adamw --lr 0.001 --weight-decay 0` † | |
| Scheduler | `--scheduler reduce_on_plateau --factor 0.5 --lrs-patience 4` † | LR-scheduler patience |
| Early stopping | `--patience 20` † | separate from `--lrs-patience` |
| Epochs | `--epochs` | 100 low, 60 medium † |
| Batch | `--batch-size 8 --eval-batch-size 64` † | `--loss-reduction mean` |
| Expert loss | `--expert-loss marginal --expert-temperature 0` | hard set-valued oracle |
| Roll-in | `--rollin-prob 0` | roll-in is an ablation, not in the demo |
| Output feedback | `--output-feedback-dim 0` | |
| SED | `--sed-em-mode damped --sed-em-damping 0.9 --sed-em-iterations 10` | fitted per bundle on its own training set |
| Tokenization | `--source-separator none --target-separator none` | character level |
| Normalization | `--nfd` off, text NFC | |
| Eval decoding | greedy for model selection; `--beam-width 4` reported separately | beam eval runs on CPU for accelerator runs |
| Device | `--device mps` locally when available (decided), else `cpu`; `cuda` on Colab | `train_bundle.py` auto-selects `mps` → `cpu`. MPS was verified to work with `trans-train` but is not faster at this scale (§4.1). |

Dev string accuracy is the selection criterion (symbol accuracy breaks ties). The test set is touched only by `eval_bundle.py`, once, after freezing.

### 4.1 Compute and cost (measured, plus HF pricing checked)
**Measured on this Mac** (English-medium: 1000 examples, the pilot config above, 3 epochs, `trans` at the pinned commit in a clean venv; the 3-epoch dev accuracy of 87–88% shows the pipeline learns):

| Phase | Time | Notes |
|---|---|---|
| SED fit (10 EM iterations) | ≈ 2 min | ≈ 13 s/iteration, single-threaded CPU. **Dominant cost**, independent of `--device`. Roughly linear in training-set size, so the 100-example fit should take seconds (to be confirmed in the pilot). |
| Expert precompute | ≈ 5–7 s | |
| Neural training + dev eval | ≈ 2–3 s per epoch | MPS ≈ CPU at batch size 8: the model is tiny, so MPS gives no real speedup, but it works (seed 42 run: 88.1% after 3 epochs). |

**Estimate:** a 1000-example bundle with 60 epochs ≈ 2 min SED + ≈ 3 min training ≈ 5–6 min; a 100-example bundle is a minute or two. All eight bundles ≈ **under an hour on a laptop** (to be confirmed in the Phase 3 pilot, which replaces these numbers with measured ones).

**Cost options:**
1. **Local Mac (chosen default, `--device mps` with `cpu` fallback): \$0.** Fastest path given the numbers above.
2. **Google Colab free tier: \$0.** This is also what the notebook does for its single bundle. A GPU is not needed; the T4 is unused at this scale, but harmless.
3. **HF Jobs: optional, not needed.** Jobs bill per minute and require a positive credit balance. PRO (\$9/month) includes \$2 of monthly compute credits, which the docs state count toward the Jobs balance. `hf jobs hardware` shows current prices: `cpu-basic` \$0.01/h, `cpu-upgrade` (8 vCPU) \$0.03/h, `t4-small` \$0.40/h. Eight bundles on `cpu-upgrade` would cost cents, well within \$2 even in the worst case. Default Job timeout is 30 min, so use `--timeout`. **Check your balance** at https://huggingface.co/settings/billing before relying on this, because I could not read it with the CLI, and the PRO page labels the \$2 "inference credits".
4. **Not used:** ZeroGPU (applies to Gradio inference Spaces, not training), paid GPU flavors, and Spaces upgrades. The demo Space runs on free `cpu-basic`.

No step in this plan needs paid compute. Inference on the free Space is CPU-only, which is fine for this model size.

## 5. Bundle Format and Loader
### 5.1 Bundle (`eng_100/`, …)
`best.model` (state_dict), `best.model.json` (epoch, dev accuracies, `git_commit`, full `vars(args)`), `sed.pkl`, `sed.pkl.json`, `vocabulary.pkl`, and `bundle_manifest.json` (SHA-256 of every file, `trans` commit, torch version, data commit, seed, train command, dev/test metrics).

### 5.2 `loader.py` contract
1. Verify `bundle_manifest.json` hashes and fail loudly on mismatch.
2. Build the `Namespace` from `best.model.json["args"]`. It is the **only** source of architecture settings; nothing is hard-coded in the app.
3. `FeatureVocabularies.from_pickle`, `StochasticEditDistance.from_pickle`, `OptimalSubstitutionExpert`, `Transducer(vocab, expert, args)`, with `args.device` overridden to the runtime device.
4. `load_state_dict(torch.load(..., map_location=device, weights_only=True))`, then `.eval()`.
5. Assert vocabulary/checkpoint shape compatibility; warn if the installed `trans` commit differs from the manifest.
6. Encode input with `encode_unseen_input` / `encode_unseen_features` (UNK-tolerant), exactly as in `train.py` at test time.

### 5.3 Dependencies
**Corrected after checking:** at the pinned commit, `pyproject.toml` *does* declare the runtime deps (`torch>=2.6`, `editdistance`, `numpy`, `scipy`, `progressbar>=2.5`); the note in AGENTS.md saying they are commented out is stale. A fresh `uv pip install <repo>` into an empty venv gave a working `trans-train` (verified). So the Space and notebook install `trans` from the pinned commit and rely on those declared deps; the package is `progressbar`, **not** `progressbar2` (the Space `requirements.txt` below is updated accordingly). `loader.py` still avoids importing `trans.train`; batching helpers are imported lazily. Phase 1 adds an import smoke test in a clean venv. Suggest (separately, in the library repo) updating the stale AGENTS.md note.

**Pickle safety:** `sed.pkl` and `vocabulary.pkl` are pickles. Load them only from our pinned model-repo revision, after the hash check. Never accept user-supplied bundles.

## 6. Space
### 6.1 Config (modeled on `romansh-morphology`)
`hf_space/README.md` header:
```yaml
---
title: Morphological Inflection
emoji: 🔤
sdk: gradio
sdk_version: 5.48.0        # same as the working sibling Space; re-check at deploy time
app_file: app.py
python_version: "3.12"     # verify the supported value at deploy time
license: <from §2.5>
short_description: Neural transducer inflection (en/de/fr/it)
---
```
`hf_space/requirements.txt` (free `cpu-basic`, 2 vCPU / 16 GB, so CPU-only torch). Runtime deps (`numpy`, `scipy`, `editdistance`, `progressbar`) come in through the package metadata; only the torch index, Hub client and Gradio are added here:
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
huggingface_hub
gradio>=5.48
neural_transducer @ git+https://github.com/simon-clematide/il-reimplementation.git@f71833192acd505bc1d3ba47d75798cfb778fc3d
```
The CPU index avoids the multi-GB CUDA wheels, which would slow builds. The Phase 5 build log is the final check that the install is complete; Phase 1 verifies the same install in a clean venv beforehand.

### 6.2 Behavior
- Inputs: language, regime (100 / 1000 / compare), lemma, tag string, decoding (greedy / beam with width).
- **Lazy loading** with an LRU of at most 2 bundles. At startup, only the default bundle is prefetched (so the first request is fast) and the example list is loaded. Space repo storage stays tiny.
- `MODEL_REPO` and `MODEL_REVISION` (tag, resolved to a commit in the manifest) are constants in `app.py`. Bumping the model release means editing one line.
- Failure handling: if the Hub is unreachable, show a clear message instead of a stack trace.

### 6.3 Feature input
Version 1 is a raw UniMorph textbox (`V;IND;PST;3;SG`). Tags are split on `;` and looked up individually, as `FeatureVocabularies` does. Tags absent from the bundle vocabulary show a warning ("treated as UNK") and are not blocked. A later phase adds language-specific pickers generated from the tag sequences actually observed in the data, so users cannot build orderings never seen in training. Curated examples cover regular, irregular and unseen-lemma cases per language, with gold forms.

### 6.4 Output
- Predicted form per regime, a comparison table, and the gold form if the (lemma, tags) pair is in dev/test.
- **Decoder score:** shown as "average action log-probability (decoder score)", with the note that it is not a calibrated probability of correctness. "Confidence" is not used.
- **Action trace:** `predict.py` converts decoder actions into `{step, op: COPY|DELETE|INSERT|SUBSTITUTE, source_pos, source_char, emitted}` records. No raw IDs are shown.

## 7. Model Repo and Release
- Layout: `eng_100/ … ita_1000/`, a top-level `README.md` (model card from `model_card/`), and `RELEASE.json` (data commit, `trans` commit, summary metrics).
- **Model card contents:** intended use and limits (small-data benchmark models, not production inflection), training data and license, the §4 config, per-bundle dev/test accuracy, the reproducibility table, the decoder-score caveat, and a known-limitations section (UNK tags, unseen lemma types, 100-example instability).
- Release flow: `upload_hf.py` uploads all bundles in a single commit, then creates the tag `v1.0`. The Space pins that tag. No bundle is ever overwritten in place; fixes become `v1.1`.
- Weights, vocabularies and SED files never go into the GitHub demo repo.

## 8. Colab Notebook
1. Intro and benchmark statement (pinned commit).
2. Install the pinned `trans` plus explicit deps (§5.3); the install cell is tested in a fresh runtime; detect the device.
3. Fetch and validate one language and one regime (default Italian, 100).
4. **Inference first:** download that pretrained bundle from the model repo at the pinned tag; predict on dev and custom input, with decoder score and action trace.
5. Train that bundle from the frozen config (runtime stated from the pilot; checkpoints to Google Drive if mounted).
6. Evaluate and compare self-trained against pretrained on dev, then test.
7. Optional collapsed section: train all 8 bundles, with a time warning.
The badge points at `simon-clematide/il-reimplementation-demo`, `notebooks/morphological_inflection.ipynb`.

## 9. Tests
- Bundle smoke test, for all 8: hashes, load via `loader.py`, one fixed prediction, non-empty output.
- Loader contract: architecture comes from `best.model.json` (editing a dim must fail loudly).
- Data validator fixtures: bad columns, missing files, count mismatch.
- Action-presentation mapping on hand-checked examples.
- Space: import `app.py` and call the predict function on every curated example.
- **Space deploy check:** after pushing, call the Space with `gradio_client` for one example per language and compare against `eval_bundle.py` output for the same inputs.

## 10. Reproducibility Table (`REPRODUCIBILITY.md`, generated)
One row per bundle: language, regime, data commit, train/dev/test SHA-256, `trans` commit, torch version, seed, command line, `best.model` SHA-256, best epoch, dev and test string accuracy (greedy and beam). The model card embeds the same table.

## 11. Phases and Gates
0. **Setup:** you create the GitHub repo (§0.2). I `git init` locally and add `.gitignore` and the layout skeleton.
1. **Foundation:** `fetch_data.py`, `MANIFEST.json`, `validate_data.py` with tests; verify dep import names. *Gate:* the validator passes for all 4 languages; the overlap and leakage numbers and data license are recorded.
2. **Loader and inference:** `loader.py`, `predict.py` and tests on a throwaway quick bundle (about 5 epochs). *Gate:* loader-contract and presentation tests pass.
3. **Pilot:** train Italian-100 with the §4 config. Measure runtime and bundle size, tune † values on dev, freeze `train.yaml`. *Gate:* sensible dev accuracy and a documented runtime.
4. **Eight bundles:** train, evaluate test once, package, smoke-test, generate `REPRODUCIBILITY.md`. You create the model repo; with your OK I upload and tag `v1.0`.
5. **Space:** build `app.py` and `requirements.txt`. You create the Space; with your OK I push, then run the deploy check from §9. *Gate:* the build is green and the `gradio_client` results match.
6. **Notebook:** build per §8, which you verify in a fresh Colab runtime. *Gate:* runs top to bottom.
7. **Docs and extras:** README (with `make`, never `remake`, in public docs), validated tag pickers, collection/pinning.

## 12. Risks
| Risk | Mitigation |
|---|---|
| Build fails on a missing runtime dep | Deps come from the pinned package metadata (verified in a clean venv); Phase 1 import smoke test; the Phase 5 build log is the final check. |
| Torch CUDA wheels slow or break the build | CPU extra-index-url. |
| Space sleeps; the first call is slow | Prefetch the default bundle at startup; document the wake-up delay. |
| 100-example models are unstable (single seed 42) | Report the seed in every table; state that the 100-example numbers come from one run and may vary. |
| MPS numerical differences vs CPU/CUDA | Record the device in `bundle_manifest.json`; the smoke test and deploy check run on CPU, so any drift would show up; greedy and beam eval run on CPU. |
| SED fit dominates runtime | Fit once per bundle and cache `sed.pkl` (bundle contents); pilot measures the 100-example time. |
| License of derived models | Resolved in §2.5 before upload. |
| Pinned git dependency becomes unavailable or force-pushed | Pin the full SHA; record it in each bundle manifest. |

## 13. Decisions
**Decided:** resource IDs as in §0.1; GitHub demo repo public from the start; single training seed **42** (no multi-seed run); local training with `--device mps` when available, else `cpu`.
**Still open:**
- Check your HF credit balance (§4.1) only if you want to use HF Jobs; the plan does not depend on it.
- Your OK on the upload and push steps when we reach Phases 4 and 5.
