# Slot: planner

| model | intent_correct | cq_correct | route_correct | valid_json | correct | p50 s | tokens/item |
|---|---|---|---|---|---|---|---|
| llama3.1:8b (best) | 0.960 [0.900-1.000] | 0.780 [0.660-0.900] | 0.980 [0.940-1.000] | 1.000 [1.000-1.000] | 0.780 [0.660-0.900] | 0.794 | 2404.600 |
| qwen2.5:14b | 0.960 [0.900-1.000] | 0.720 [0.600-0.840] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.720 [0.600-0.840] | 1.447 | 2454.560 |

WARNING: results below are not valid model comparisons:

- qwen3:8b: skipped (HTTP 404: {"error":{"message":"model 'qwen3:8b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- qwen3:14b: skipped (HTTP 404: {"error":{"message":"model 'qwen3:14b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- deepseek-r1:14b: skipped (HTTP 404: {"error":{"message":"model 'deepseek-r1:14b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- gemma4:12b: skipped (HTTP 404: {"error":{"message":"model 'gemma4:12b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- phi4:14b: skipped (HTTP 404: {"error":{"message":"model 'phi4:14b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- mistral-small:24b: skipped (HTTP 404: {"error":{"message":"model 'mistral-small:24b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))
- gpt-oss:20b: skipped (HTTP 404: {"error":{"message":"model 'gpt-oss:20b' not found","type":"api_error","param":null,"code":null}}
 (pull it: ollama pull <tag>))

McNemar (exact) vs best, repeat 0:

- qwen2.5:14b: best-only correct 6, other-only correct 3, p = 0.5078
