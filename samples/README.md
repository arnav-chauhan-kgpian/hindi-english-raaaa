# Sample clips

A handful of real clips to test your engine on locally — English and Hindi+English
(code-switch) — each with a reference transcript in [`manifest.json`](manifest.json).

```bash
# transcribe one
python -m solution.transcribe --input samples/<clip_id>.wav --mode auto --output /tmp/out.json

# or score your engine on all of them, offline, like admission does
python preview.py            # point it at samples/manifest.json
```

These are **for your own testing**. The official ranking runs on a larger **hidden** set
(real Indian-English + Hindi+English work speech) you don't get to see — so don't tune to these.

## Attribution
Sample clips are short excerpts from public research speech corpora, included here for challenge
testing, with thanks to the original authors:

- **English:** [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) — CC-BY 4.0.
- **Hindi+English (code-switch):** [OpenSLR-104](https://www.openslr.org/104/) — the MUCS Hindi-English
  code-switching corpus.

If you redistribute, keep the attribution and follow each corpus's license.
