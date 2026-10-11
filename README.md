# wildfire-mas

Multi-agent wildfire screening on top of the FFOnto knowledge graph: a planner, an
orchestrator (LangGraph), worker agents that call tools only through MCP, live NASA FIRMS
hotspots and fire weather, rules R09–R12, and a synthesizer + verifier that cite every
claim. The LLM plans and words the answer; the facts come from the KG, the rules, live
sources, or cited documents.

```
question ─► PLANNER ─► plan (typed DAG) ─► ORCHESTRATOR (LangGraph: parallel waves, timeouts)
              intent, urgency,                │
              slots (few-shot)    ┌───────────┼─────────────┬──────────────┬─────────────┐
                                  ▼           ▼             ▼              ▼             ▼
                               resolve    live_data       risk            kg            rag
                               record→    FIRMS + FWI    rules R09–R12   CQs + IORS,   BM25/bge-m3,
                               site/point (IORS radius)  hazards         NL→SPARQL     CRAG, figures
                                  │           │             │              │             │
                                  └─── MCP: graphdb-mcp · firms-mcp · weather-mcp · rag-mcp ───┘
                                                                │
                     evidence ledger (KG1, LIVE2, R09-1, D3, F1, O1 …) ─► SYNTHESIZER (CRM schema per intent)
                                                                         ─► VERIFIER (cites, safety recall) ─► answer
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,validate]"
cp .env.example .env                # FIRMS_MAP_KEY, FFONTO_V2_PATH, GraphDB URL/repo
ollama pull qwen2.5:14b && ollama pull qwen2.5:7b && ollama pull bge-m3
pytest                              # offline tests; with FFONTO_V2_PATH set, also checks against your project
```

## How it connects to ffonto-v2

wildfire-mas does not replace anything in ffonto-v2. ffonto-v2 builds the knowledge graph (Phase A:
lift, rules R00–R08, validation) and owns the KG question-answering code; wildfire-mas answers
questions (Phase B) and reads the same GraphDB repository.

| From ffonto-v2 | Used by wildfire-mas for |
|---|---|
| `qa/cqs.py` + `queries/CQ*.rq` | the GraphDB agent's CQ01–CQ26, narrowed by record / event / site / date exactly as ffonto-v2 does |
| `qa/pipeline.text_to_sparql` | questions no CQ covers ("plus" mode; `T2S_*` switches still work) |
| `qa/rag.py` + `qa/index` | the RAG agent (bge-m3 over text, tables and figures) when the index exists |
| `qa/examples.py` + `qa/index/examples.jsonl` | the planner's few-shot examples (embedding-nearest labelled questions) |
| `docs/FFVQA_CQ_mapping.xlsx` | golden set Part A (`wfmas golden-import`) |
| `ontology/*.ttl` | the SPARQL guard's term list (read in place, no copying) |

Set `FFONTO_V2_PATH` in `.env`. Without it the system falls back to its own live CQs, guard and BM25 index.
Build ffonto-v2's indexes once if you have not: `cd $FFONTO_V2_PATH && python -m qa.build_index`.

The live module extends FFOnto rather than shadowing it:

| Live data | FFOnto terms it uses | Effect |
|---|---|---|
| FWI days, current weather | `ffonto:WeatherRecord`, `atSite`, `observationDate`, `airTemperatureC`, `relativeHumidityPct`, `windSpeedKmh`, `rainMm`, `ffmc` … `fwi`, `previousRecord` | your CQ08 / CQ09 return live weather too |
| Live hazards (R09–R12) | `ffonto:Hazard`, `hasHazardType`, `inEvent`, `prov:wasDerivedFrom`, `rdfs:comment`; hazard types in `ffontov:HazardScheme` with `mitigatedBy` actions | your CQ24 / CQ25 list satellite hazards and their actions next to image/audio hazards |
| Hotspot classification | `ffonto:ModelObservation`, `ffonto:confidence` (xsd:double) | consistent with your observation model |
| R12 and selected records | `centerWKT`, `captureTime`, `aboutEvent`, `hasObservation` → `predictedClass ffontov:Fire`, `usedThreshold/thresholdValue` | live checks run at the image's own location |

New names avoid clashes with your vocabulary: FWI danger classes are `ffontov:FWIDanger*` (not your
severity values High/Low/Extreme), and R11's hazard is `ElevatedDangerNearActiveFire` (related to your
condition `ElevatedFireDanger`).

### Load the live module into GraphDB

```bash
wfmas ontology load        # ffonto-live.ttl -> graph/ontology, ffonto-live-vocab.ttl -> graph/vocab
wfmas test-cqs             # CQL06 passes once the vocab is loaded
```

### Upgrading from the first version (you loaded the earlier ffonto-live*.ttl)

The first version declared some FFOnto terms again with other ranges (xsd:decimal) and reused the
names High/Low/Extreme. Remove those triples before loading the new files:

```bash
mkdir -p ~/old-live && cp ontology/ffonto-live*.ttl ~/old-live/     # BEFORE replacing the folder
# replace the wildfire-mas folder with the new version (keep your .env), then:
wfmas ontology unload --dir ~/old-live   # removes exactly the old files' triples
wfmas ontology clear-live                # live data written with the old terms
wfmas ontology load
```

### Still to do on your side

- **Site coordinates.** FFOnto sites have `siteName` / grid cells but no coordinates, so R09/R11 only see
  sites the MAS registers for a question (a place, coordinates or a record's `centerWKT`). To include a
  real site, add `ffonto:latitude` / `ffonto:longitude` to it (or a GeoSPARQL WKT point).
- **Images without `centerWKT`** (or audio) get no live checks; the answer says so as a data gap.

### Commands

```bash
wfmas key-status                                     # check the MAP_KEY
wfmas hotspots --brief                               # Sri Lanka (default), last 24 h, stored in the live graph
wfmas hotspots --area=-60,-15,-45,-5 --days 1 --no-persistence --no-store --brief   # note the "=" before a negative bbox
wfmas fwi --lat 7.29 --lon 80.63                     # FWI + danger class, Kandy
wfmas rules                                          # R09–R12 -> inferred-live graph
wfmas ask "Are there any fires near Kandy right now?" --role incident_commander --json trace.json
wfmas ask "What hazards are present and what should we do?" --record "http://purl.org/ffonto/data/record/<uuid>"
wfmas ask "Is the fire score above its threshold?" --record "http://purl.org/ffonto/data/record/<uuid>"
wfmas ask "How does FWI relate to fire spread?"
wfmas ask "..." --transport mcp                      # every tool call over MCP stdio
wfmas test-cqs                                       # CQ tests (entries with an expect block)
wfmas shacl                                          # SHACL check of the live graphs
wfmas golden-import                                  # golden Part A from FFVQA_CQ_mapping.xlsx
streamlit run app/streamlit_mas.py                   # UI with map, plan, evidence
```

## NASA FIRMS: what is and is not a wildfire

FIRMS reports thermal anomalies. `live/classify.py` keeps every detection but labels it,
and only `likely_vegetation_fire` counts as wildfire evidence in rules and answers:

| Check | Category |
|---|---|
| FIRMS `type` column (standard-processing products only): 1 / 2 / 3 | volcanic / static land source / offshore |
| Within 1 km of an entry in `data/static_sources.csv` | static land source or volcanic |
| Outside `land_mask_file` (optional GeoJSON) | offshore |
| Same ~400 m cell detected on ≥ 8 distinct days in 30 | persistent static source (flares, plants) |
| VIIRS confidence `l`, or MODIS confidence < 30 | low confidence |
| none of the above | likely vegetation fire (may still be agricultural burning) |

NRT products have no `type` column, so the persistence check does most of the work. It
costs one extra 30-day request per call (cached 15 minutes; your key allows 5,000
transactions per 10 minutes). Add verified static sources for Sri Lanka to
`data/static_sources.csv` to tighten it further.

The MAP_KEY is read from `.env` and never written to logs, caches, snapshots or the KG.
Set `firms.record_dir` to save every response, and `firms.replay_dir` to replay them, so
evaluation runs are reproducible.

## Where each technique lives

| Technique | Code |
|---|---|
| Planner (intent, urgency, slots, typed plan DAG, template or LLM plans) | `agents/planner.py`, `agents/state.py` |
| Orchestrator (LangGraph; same nodes run without it) | `orchestrator.py` |
| MCP servers / client bus | `mcp_servers/*`, `tools/bus.py` |
| IORS: radius/day ladder for FIRMS, CQ relaxation, CRAG re-query, early stop | `agents/workers.py` |
| CRM: response schema per intent, evidence→section mapping, safety override | `agents/synthesis.py`, `agents/prompts.py` |
| IALT: intent/urgency/role-conditioned prompts | `agents/prompts.py` |
| Verifier (cites exist, every claim cited, every hazard reported; optional LLM check) | `agents/synthesis.py` |
| SPARQL guard (read-only, LIMIT, ontology terms only, no SERVICE) | `kg/guard.py` |
| Bridge to ffonto-v2 (CQ01–CQ26, text-to-SPARQL, RAG, labelled examples) | `kg/ffonto_bridge.py` |
| Rules R09–R12 | `kg/rules_live.py`, `rules/*.rq` |
| FWI (Van Wagner 1987, latitude-adjusted day lengths) | `live/fwi.py` |

## Comparing models

Each LLM job is a slot in `config/models.yaml` (planner, sparql, synthesizer, verifier,
rag_grader, scene_graph, figure_reader). The harness changes the model in one slot and
keeps the others at baseline:

```bash
wfmas golden-import                                   # once: eval/datasets/golden_partA.jsonl (1,001 items)
wfmas-eval --slot planner --pool text --dataset eval/datasets/golden_partA.jsonl --split test --repeats 3
wfmas-eval --slot synthesizer --models qwen2.5:14b,llama3.1:8b,gemma4:12b --dataset eval/datasets/synth.jsonl
wfmas-eval --slot scene_graph --pool vision --dataset eval/datasets/scene.jsonl
wfmas-eval --slot sparql --pool text --dataset eval/datasets/sparql.jsonl     # needs GraphDB
wfmas-eval --slot e2e --models qwen2.5:14b,qwen3:14b --dataset eval/datasets/e2e.jsonl
```

Each run writes `runs/<time>_<slot>/items.jsonl`, `summary.csv` and `summary.md`: mean and
95% bootstrap CI per metric, p50/p95 latency, tokens per item, and exact McNemar tests
against the best model. The datasets in `eval/datasets/` are format examples; replace them
with your labelled sets (see `eval/datasets/README.md`). Check model tags in your Ollama
library before running; DeepSeek-VL2 and InternVL3 run through vLLM.

## Status

Tested here: 30 tests, including integration checks against the ffonto-v2 project (`FFONTO_V2_PATH`):
every term in the live CQs and rule inputs exists in FFOnto + the live module; FFOnto's 27 CQ queries
pass the SPARQL guard; the bridge loads CQ01–CQ26 and narrows CQ17 by record and CQ24 by the record's
event; the planner plans CQ24/CQ25/CQ17 + live checks for a selected record; the 1,001 labelled
questions import. Not run here: FIRMS, Open-Meteo, GraphDB, Ollama (this sandbox cannot reach them).
