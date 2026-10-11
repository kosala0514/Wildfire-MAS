# Slot: planner

| model | intent_correct | cq_correct | route_correct | valid_json | correct | p50 s | tokens/item |
|---|---|---|---|---|---|---|---|
| qwen3:8b | 0.980 [0.940-1.000] | 0.780 [0.660-0.900] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.780 [0.660-0.900] | 3.981 | 2839.280 |
| qwen2.5:14b | 0.960 [0.900-1.000] | 0.720 [0.600-0.840] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.720 [0.600-0.840] | 1.443 | 2454.560 |
| qwen3:14b (best) | 0.960 [0.900-1.000] | 0.800 [0.680-0.900] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.800 [0.680-0.900] | 7.392 | 2925.860 |
| deepseek-r1:14b | 0.920 [0.840-0.980] | 0.740 [0.620-0.860] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.740 [0.620-0.860] | 7.944 | 2905.880 |
| phi4:14b | 0.960 [0.900-1.000] | 0.780 [0.660-0.900] | 0.980 [0.940-1.000] | 1.000 [1.000-1.000] | 0.780 [0.660-0.900] | 1.582 | 2428.980 |
| mistral-small:24b | 0.940 [0.880-1.000] | 0.780 [0.660-0.880] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.780 [0.660-0.880] | 12.425 | 2491.340 |

WARNING: results below are not valid model comparisons:

- llama3.1:8b: skipped (HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=180))
- gemma4:12b: skipped (HTTP 404: {"error":{"message":"model 'gemma4:12b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- gpt-oss:20b: skipped (HTTP 404: {"error":{"message":"model 'gpt-oss:20b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))

McNemar (exact) vs best, repeat 0:

- qwen3:8b: best-only correct 3, other-only correct 2, p = 1.0000
- qwen2.5:14b: best-only correct 5, other-only correct 1, p = 0.2188
- deepseek-r1:14b: best-only correct 6, other-only correct 3, p = 0.5078
- phi4:14b: best-only correct 3, other-only correct 2, p = 1.0000
- mistral-small:24b: best-only correct 3, other-only correct 2, p = 1.0000
