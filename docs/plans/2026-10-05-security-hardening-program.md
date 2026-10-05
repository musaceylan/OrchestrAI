# OrchestrAI Security Hardening Program Implementation Plan

> **For Hermes:** Execute each slice with one Astra Max implementation owner, strict RED → GREEN → REFACTOR, deterministic verification, and a final independent security review.

**Goal:** Close the confirmed trust-boundary, policy, execution, persistence, and verification defects while preserving OrchestrAI's MCP contracts, provider abstraction, six orchestration modes, stdio behavior, and loopback endpoint semantics.

**Architecture:** Introduce server-owned task context and deep modules for validated configuration, capability identity, eligibility policy, repository paths, execution lifecycle, HTTP security, durable artifacts, and bounded provider work. Request data may narrow server policy but can never become authoritative for trust, identity, filesystem, execution, ownership, or budget decisions.

**Tech stack:** Python 3.11+, Pydantic Settings, MCP Python SDK, Starlette SSE transport, asyncio subprocesses, pytest, Ruff, mypy, GitHub Actions.

**Baseline:** `45b65746f0cdde654c46e64c39f45dc869ebd2d8`; 186 tests passed; 66% coverage.

**Preserve:** all 16 MCP tool names/input schemas, all six mode names, advertised provider model IDs, `BaseProvider`, local stdio framing, and the endpoint trust definition in `tests/test_endpoint_trust.py`.

---

## Task 1: Validate configuration and fail closed

**Files:**
- Modify: `config/default.yaml`
- Modify: `src/orchestrai/config/settings.py`
- Modify: `src/orchestrai/server/tools.py`
- Create: `tests/test_config.py`

**RED:** Add focused tests proving that the shipped YAML loads; an absent default file uses typed defaults; an explicitly configured missing, malformed, non-mapping, unknown-key, or invalid-value file fails; a failed reload retains the previous runtime.

**GREEN:** Make YAML match the typed schema, preserve missing-default-file behavior, raise safe location-only errors for explicit invalid config, and validate reloads before atomic publication. Define deterministic defaults/YAML/environment precedence without logging values.

**REFACTOR:** Keep parsing and publication separate. Do not add fallback branches that substitute permissive defaults after validation failure.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_config.py tests/test_phase4.py
```

**Commit:** `fix: fail closed on invalid configuration`

## Task 2: Make registry identity unambiguous

**Files:**
- Modify: `src/orchestrai/providers/base.py`
- Modify: `src/orchestrai/providers/discovery.py`
- Modify: `src/orchestrai/registry/registry.py`
- Modify: `src/orchestrai/orchestrator/judge.py`
- Create: `tests/test_registry.py`

**RED:** Reproduce duplicate provider overwrite, duplicate model overwrite, identity spoofing, slash-containing model IDs, and ambiguous bare judge overrides.

**GREEN:** Use `(provider_name, model_id)` internally, reject duplicate or mismatched identities, keep advertised model IDs unchanged, support qualified judge overrides through the existing string field, and publish registry refreshes atomically.

**REFACTOR:** Centralize identity parsing/formatting without expanding the provider interface.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_registry.py tests/test_routing.py tests/test_modes.py
```

**Commit:** `fix: reject ambiguous provider identities`

## Task 3: Centralize monotonic task eligibility

**Files:**
- Create: `src/orchestrai/policies/eligibility.py`
- Create: `src/orchestrai/orchestrator/context.py`
- Modify: `src/orchestrai/registry/router.py`
- Modify: `src/orchestrai/orchestrator/judge.py`
- Modify: `src/orchestrai/orchestrator/modes/base_mode.py`
- Modify: `src/orchestrai/server/tools.py`
- Modify: `src/orchestrai/providers/ollama.py`
- Modify: `src/orchestrai/providers/vllm.py`
- Create: `tests/test_policy_eligibility.py`

**RED:** Cover stricter privacy selection, local-only OR, allowlist intersection, denylist union, minimum cost ceiling, empty-intersection denial, preferred-provider ranking without access grants, judge bypass, rerun weakening, and remote native-compatible endpoint labeling.

**GREEN:** Resolve an immutable server-owned policy snapshot. Request preferences may only tighten. Route, judge, compare, rerun, and provider execution must all use the same eligibility result and recheck capability identity immediately before transmission.

**REFACTOR:** Delete duplicated privacy/provider ordering logic after the shared module is green. Preserve the existing endpoint trust tests.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_policy_eligibility.py tests/test_routing.py tests/test_endpoint_trust.py tests/test_openai_compat.py tests/test_modes.py
```

**Commit:** `fix: enforce monotonic task eligibility`

## Task 4: Enforce canonical repository roots

**Files:**
- Create: `src/orchestrai/policies/paths.py`
- Modify: `src/orchestrai/config/settings.py`
- Modify: `src/orchestrai/orchestrator/intake.py`
- Modify: `src/orchestrai/orchestrator/orchestrator.py`
- Modify: `src/orchestrai/server/tools.py`
- Create: `tests/test_repo_roots.py`

**RED:** Cover omitted roots, explicit empty roots, sibling-prefix paths, traversal, symlink escape, sensitive paths, nonexistent roots, and rerun revalidation.

**GREEN:** Resolve allowed roots once at startup, canonicalize with path-component containment, default omitted roots to the startup workspace, define `[]` as deny-all, reject filesystem root as an implicit allowance, and validate before scanning or artifact creation.

**REFACTOR:** Keep repository-boundary knowledge behind one interface used by every filesystem/execution caller.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_repo_roots.py tests/test_intake.py tests/test_tools.py
```

**Commit:** `fix: restrict repository access to allowed roots`

## Task 5: Own subprocess lifetime and resources

**Files:**
- Modify: `src/orchestrai/execution/shell.py`
- Modify: `src/orchestrai/orchestrator/orchestrator.py`
- Modify: affected mode callers
- Create: `tests/test_shell.py`

**RED:** Use harmless real processes to prove descendant survival, inherited credential canaries, output overflow, cancellation, timeout, and test-path option injection.

**GREEN:** Keep `create_subprocess_exec`; use explicit arrays and option boundaries, `stdin=DEVNULL`, minimal allowlisted environment, POSIX process groups, bounded streaming reads, graceful group termination then kill, and guaranteed reap on timeout/cancellation/shutdown. Never invoke installing `npx` behavior.

**REFACTOR:** Centralize process-tree cleanup and output collection. Re-raise cancellation after cleanup.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_shell.py tests/test_modes.py tests/test_tools.py
```

**Commit:** `fix: isolate and bound repository commands`

## Task 6: Secure SSE and task ownership

**Files:**
- Create: `src/orchestrai/server/security.py`
- Create: `src/orchestrai/server/runtime.py`
- Modify: `src/orchestrai/server/main.py`
- Modify: `src/orchestrai/server/tools.py`
- Modify: `src/orchestrai/orchestrator/orchestrator.py`
- Modify: `src/orchestrai/artifacts/schemas.py`
- Create: `tests/test_sse_security.py`
- Create: `tests/test_tool_authorization.py`

**RED:** Cover non-loopback startup refusal, bearer-token failures on both HTTP routes, invalid Host/Origin, cross-owner reads/cancel/rerun/compare, admin-only reload/probe, revoked tokens, oversized bodies, per-principal rate limits, and active-task limits.

**GREEN:** Preserve stdio. Default SSE to `127.0.0.1`; allow tokenless direct loopback compatibility only with loopback peer plus Host/Origin checks. Require bearer auth on every request when configured. Refuse non-loopback unless explicitly enabled, authenticated, and TLS-terminated by configured policy. Bind POST to the authenticated SSE session, assign principal/scopes/roots, enforce task ownership, limit request bytes/rates/concurrency, and keep tokens out of logs/artifacts.

**Protocol note:** This bounded opaque-token resource-server mode does not claim OAuth discovery or authorization-server compliance. Full OAuth requires an external authorization server integration.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_sse_security.py tests/test_tool_authorization.py tests/test_tools.py
```

**Commit:** `fix: authenticate and authorize SSE operations`

## Task 7: Make artifacts durable and safely reopenable

**Files:**
- Modify: `src/orchestrai/artifacts/store.py`
- Modify: `src/orchestrai/artifacts/schemas.py`
- Modify: `src/orchestrai/observability/trace.py`
- Modify: `src/orchestrai/orchestrator/orchestrator.py`
- Modify: artifact inspection handlers
- Extend: `tests/test_artifacts.py`
- Create: `tests/test_trace.py`

**RED:** Cover fresh-instance reopening, unknown-task inspection without directory creation, malformed/oversized files, wrong provenance, symlinks, write failure, permissions, partial writes, and legacy manifests.

**GREEN:** Separate create/open modes, validate IDs and provenance, add versioned task manifests with owner/policy/mode/final reference, write `0600` files inside `0700` directories using same-directory temporary files, fsync and atomic replacement, and publish memory only after durable persistence. Legacy artifacts remain local-inspection-only until migrated.

**REFACTOR:** Share one already-correct atomic I/O helper after both artifact and trace tests are green.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_artifacts.py tests/test_trace.py tests/test_tools.py
```

**Commit:** `fix: persist and rehydrate task artifacts safely`

## Task 8: Align modes, verdicts, and evidence

**Files:**
- Modify: all files in `src/orchestrai/orchestrator/modes/`
- Modify: `src/orchestrai/registry/router.py`
- Modify: `src/orchestrai/orchestrator/judge.py`
- Modify: `src/orchestrai/artifacts/schemas.py`
- Modify: `README.md`
- Extend: `tests/test_modes.py`
- Create: `tests/test_verification_evidence.py`

**RED:** Prove docs/refactor roles are skipped, malformed review text approves, the wrong parallel candidate is reviewed, and checkout tests are incorrectly attributed to unapplied generated code.

**GREEN:** Execute documented role sequences, validate review verdicts fail-closed, choose before reviewing, link review/test evidence to the exact candidate and worktree revision, and represent generated tests as unverified until run against an applied candidate. Preserve all mode names.

**REFACTOR:** Extract shared mode specifications only after behavior is correct.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_modes.py tests/test_verification_evidence.py tests/test_phase4.py
```

**Commit:** `fix: bind orchestration evidence to executed roles`

## Task 9: Bound provider work/scans and add security CI

**Files:**
- Create: `src/orchestrai/orchestrator/provider_calls.py`
- Modify: `src/orchestrai/orchestrator/intake.py`
- Modify: provider adapters and discovery
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/security.yml`
- Create: `.github/dependabot.yml`
- Fix: `Dockerfile`
- Create: `tests/test_provider_limits.py`
- Create: `tests/test_scan_budgets.py`

**RED:** Cover hanging providers, nested retry multiplication, cancellation, global/provider concurrency, unknown remote pricing under hard budget, scan depth/count/time/size limits, and truncation reporting.

**GREEN:** Apply one monotonic task deadline; centralize bounded retries/cancellation and provider-call admission; add completion/discovery ceilings; bound traversal and prune VCS/vendor/generated/sensitive paths; add least-privilege CodeQL/dependency-review/security workflows and Dependabot configuration; fix the Docker health check.

**Verification:**
```bash
.git/orchestrai-venv/bin/python -m pytest -o addopts= -q tests/test_provider_limits.py tests/test_scan_budgets.py tests/test_ollama.py tests/test_phase4.py
```

**Commit:** `fix: bound orchestration resources and add security gates`

## Task 10: Integrated verification and independent review

**Files:**
- Create: `tests/test_mcp_contract.py`
- Create: `tests/test_security_integration.py`
- Create: `tests/fixtures/mcp_tool_schemas.json`
- Modify documentation only where actual behavior changed

**Contract tests:** Freeze all 16 tool names/input schemas, six modes, stdio framing/stderr discipline, task ownership, policy persistence, cancellation/restart, inspect/compare/rerun restrictions, and local endpoint trust.

**Final gates:**
```bash
.git/orchestrai-venv/bin/python -m pytest -p no:cacheprovider tests/ -q --cov=src/orchestrai --cov-report=term-missing
.git/orchestrai-venv/bin/python -m ruff check src/ tests/
.git/orchestrai-venv/bin/python -m mypy src/orchestrai
.git/orchestrai-venv/bin/python -m build
.git/orchestrai-venv/bin/python -m pip_audit

git diff --check
```

Unavailable tools are reported as `BLOCKED`, never as passing. Compare lint/type/security results against the fixed baseline and do not hide pre-existing debt with broad exclusions.

**Independent review:** Freeze the final diff hash. A separate security reviewer must check policy leakage, transport/session/task ownership, filesystem containment, subprocess trees, persistence recovery, resource exhaustion, secret handling, MCP compatibility, and truthful verification evidence.

**Final commit:** `chore: verify security hardening program`

---

## Global constraints

- One mutation owner at a time; Hermes remains sole orchestrator.
- Astra Max implementation provenance: `gpt-6-astra`, reasoning `xhigh`.
- No hardcoded credentials or secret values in tests, logs, artifacts, plans, or MCP arguments.
- No dependency installation without an explicit demonstrated need.
- No deployment or push during implementation.
- No destructive Git commands, force pushes, production changes, or infrastructure mutations.
- Keep remote SSE disabled throughout partial rollout.
- Every production behavior change requires observed RED evidence first.
