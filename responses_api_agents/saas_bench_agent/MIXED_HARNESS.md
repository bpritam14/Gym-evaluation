# Mixed OpenCode and browser-use harness

The mixed harness keeps OpenCode harness 2.1.1 / CLI 1.17.11 as the agent loop and
uses browser-use 0.13.7 through the shipped indexed DOM MCP bridge. It is selected
as OpenCode with `saas_bench_browser_backend=browser_use`; it is not the standalone
browser-use agent. The browser tools are implemented in
`benchmark/saas_bench/opencode/browser_use.py`; configuration, screenshot refresh,
planning contract and parsing live alongside it. The matching Gym model adapter
is `responses_api_models/vllm_model/saas_opencode.py`.

Actor input contains effective history/tool schemas, current indexed DOM/browser
state and only the latest browser screenshot. Reasoning is retained separately.
Action-turn content is one JSON object with evaluation_previous_goal, memory,
next_goal, current_plan_item and plan_update; executable MCP calls remain in native
`tool_calls`. Final content adds final_response. See the actual prompt for field
types. Markdown/outer-quote wrapping around a valid planning object is normalized;
text pretending to be a tool call is never executed. Compaction and pruning use
OpenCode's native mechanisms; recorded effective requests remain authoritative.

## Fresh deployment

Use this entire pinned Gym branch, not just the SaaS pipeline repository or one
copied Python file. Install Gym dependencies from its lock and the worker in a
separate Python environment. A dedicated worker must include browser-use 0.13.7,
MCP 1.26.0, Playwright, Pillow, PyYAML, requests and PyMuPDF. Deployment release
records should pin the actual worker dependencies and Chromium binary together.
Do not copy a virtualenv and assume its absolute paths are portable.

Follow README.md for required task data, app images, Enroot/network helpers,
credentials and mounts. Choose writable cache/run roots outside user homes.
The following uses the existing OpenCode benchmark with the explicit mixed option:

```bash
# Export the site paths documented in README.md first.
export SAAS_BENCH_RUN_ROOT=/workspace/runs/mixed-smoke-001
# Optional: select one task for BOTH prepare and run.
gym eval prepare --benchmark saas_bench_opencode \
  +saas_bench_browser_backend=browser_use +saas_bench_tasks=TASK_ID
gym eval run --benchmark saas_bench_opencode --model-type vllm_model --split benchmark \
  +saas_bench_browser_backend=browser_use +saas_bench_tasks=TASK_ID \
  +saas_bench_max_steps=300 +saas_bench_agent_timeout=5400 \
  +policy_base_url=https://MODEL_HOST/v1 +policy_model_name=MODEL_NAME
```

Model reasoning must also be enabled in the selected provider's supported
sampling configuration; it is not inferred from a model name. Supply model and
judge keys through the documented runtime environment, not in command history,
Git, source bundles or screenshots. Native benchmark grading still reads final
application state; a synthetic trajectory's later Gemini review is a separate
stage and cannot replace benchmark verifiers.

Use a fresh prepared bundle after changing code or task data. Preflight must
validate original 106-task inventory, UID helpers, current Chromium/MCP imports,
rootless namespaces, app image registry coverage and actual app startup/login on
the destination compute nodes. Passing schema/unit tests on the source cluster
does not qualify a different cluster's app runtime.

This branch publishes the experimental mixed implementation separately from the
standard two-harness evaluation PR. It does not update that PR, Factory pins or
any existing frozen campaign. See the deployment handoff for release checksums,
portable binary assets and exact validation evidence.
