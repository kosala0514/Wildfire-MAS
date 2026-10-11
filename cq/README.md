# CQ catalog

Every `*.yaml` file in this folder is loaded into one catalog. The KG agent fills a CQ's
tested SPARQL template with typed parameters, runs it through the guard, and checks
sufficiency (`required_vars`, `min_rows`). If evidence is insufficient it walks the
`relax` ladder (IORS) before falling back to LLM-written SPARQL.

## FFOnto's CQ01–CQ26

They are not copied here. With `ffonto_v2.path` set, the KG agent runs them from ffonto-v2
(`queries/CQ*.rq` through `qa.cqs.run_cq`), narrowed by record / event / site / date. This folder only
holds the live-data CQs (CQL01–CQL09). Add more live CQs in the same format:

Parameter types: `iri` (wrapped in `<>`, validated), `float`, `int`, `string`
(escaped literal), `datetime` (typed `xsd:dateTime`). Two derived parameters are
always available: `since` (now minus `days`) and the bbox `lat_min/lat_max/lon_min/lon_max`
(computed from `lat`, `lon`, `radius_km` when those are given).

Namespaces written as `http://purl.org/ffonto#` / `http://purl.org/ffonto/vocab#`
are replaced at load time with the IRIs in `config/settings.yaml`.

`wfmas test-cqs` runs every CQ here that has an `expect` block (see CQL06). FFOnto's own CQ tests stay
in ffonto-v2 (`pytest tests/test_cq_suite.py`).
