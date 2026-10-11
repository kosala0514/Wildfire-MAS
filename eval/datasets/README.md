# Evaluation datasets

**Golden Part A** (offline KG, 1,001 labelled ForestFire-VQA questions) is generated from ffonto-v2:
`wfmas golden-import` writes `golden_partA.jsonl` (gold intent, gold CQ, route, gold answer, dev/test
split). The planner is evaluated leave-one-out, so a question is never its own few-shot example.

The other files here are small format examples. Replace them with your real sets:

| File | Slot | Fields | Build it from |
|---|---|---|---|
| intent.jsonl | planner | question, intent, cq? | labelled ForestFire-VQA questions (keep a held-out split; the few-shot pool is intent_fewshot.jsonl) |
| sparql.jsonl | sparql | question, gold_cq + gold_params, or gold_sparql | your 59 CQ tests plus paraphrases |
| synth.jsonl | synthesizer | question, intent, urgency, role, evidence[] | evidence packs saved from real runs (`wfmas ask --json`) |
| scene.jsonl + scene_vocab.txt | scene_graph | image, gold {concept: count} | 100-200 hand-annotated images |
| figures.jsonl | figure_reader | image, question, gold_value / gold_trend | figures cropped from qa/rag_docs |
| e2e.jsonl | e2e | question, record/place/lat/lon, intent, hazards[] | 30-50 scenarios; run with `firms.replay_dir` set |

Do not overlap intent.jsonl with intent_fewshot.jsonl, or the planner is tested on its own examples.
