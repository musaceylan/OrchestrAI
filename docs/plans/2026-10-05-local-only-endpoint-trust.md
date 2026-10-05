# Local-only Endpoint Trust Implementation Plan

> **For Hermes:** Use the Mastra engineering implementation worker with explicit `gpt-6-astra` and `xhigh`, then independently verify the diff and behavior.

## PRD

**Goal:** Make OrchestrAI's `local_only` and `secret` privacy promises truthful by preventing remote OpenAI-compatible endpoints from being classified as machine-local.

**Problem:** `LocalProviderEndpoint.base_url` accepts arbitrary URLs, while `OpenAICompatProvider` labels every discovered model `PrivacyLevel.SECRET`. The router's `local_only` path allows all `ProviderKind.OPENAI_COMPAT` providers and relies on privacy filtering. A remote OpenAI-compatible URL can therefore receive content that policy says must remain on the machine.

**User value:** Operators can trust that `local_only` means loopback-only, not merely API-compatible.

**Standards basis:** RFC 6761 §6.3 defines `localhost.` and names beneath `.localhost.` as special-use names that resolve to loopback addresses. Python's standard `urllib.parse` and `ipaddress` modules are sufficient; no dependency is needed.

**Acceptance criteria:**

1. `localhost`, `*.localhost`, IPv4 loopback (`127.0.0.0/8`), and IPv6 loopback (`::1`) endpoints remain `PrivacyLevel.SECRET`.
2. Public hostnames, public IPs, RFC1918/LAN addresses, malformed URLs, and URLs without a hostname are not treated as machine-local and are conservatively labeled `PrivacyLevel.PUBLIC`.
3. `local_only` routing excludes remote OpenAI-compatible capabilities through the existing privacy interface.
4. No provider API, MCP tool schema, orchestration mode, or existing configuration key changes.
5. No new dependency.
6. Tests demonstrate RED before implementation and GREEN after implementation.
7. README documents the loopback-only meaning of `local_only`.

**Out of scope:** Endpoint authentication, SSE authentication, artifact persistence hardening, provider failover, global lint cleanup, type-check debt, deployment, and dependency upgrades.

## Architecture

The `LocalProviderEndpoint` configuration module owns endpoint trust classification because it has locality: the parsed URL and its declared role are together. Add one small read-only interface such as `is_loopback` that hides URL parsing and IP-address rules behind the configuration module.

`OpenAICompatProvider` consumes that interface when building `ModelCapability`. It emits `SECRET` only for machine-loopback endpoints and `PUBLIC` otherwise. The existing `CapabilityRegistry.capabilities_for_role` privacy interface and `RoutingEngine` then enforce the policy without a second hostname parser or a new routing seam.

This is a deepening change: endpoint trust logic is concentrated behind one interface, while routing remains unchanged. Deleting the helper would reintroduce duplicated or implicit trust logic, so it passes the deletion test.

## TDD task list

### Task 1: Define endpoint locality behavior

**Files:**
- Modify: `tests/test_ollama.py` or create a focused provider test file if that improves locality.
- Modify: `src/orchestrai/config/settings.py`.

1. Add parameterized tests for `localhost`, subdomains of `.localhost`, `127.0.0.1`, another `127/8` address, and `[::1]`.
2. Add parameterized negative tests for a public hostname, public IP, RFC1918 IP, malformed URL, and missing hostname.
3. Run only the new tests and capture the expected RED failure.
4. Implement the minimum immutable locality property using `urllib.parse.urlsplit` and `ipaddress.ip_address`.
5. Run only the new tests and capture GREEN.

### Task 2: Enforce privacy labeling

**Files:**
- Modify: focused provider tests.
- Modify: `src/orchestrai/providers/openai_compat.py`.

1. Add a test showing a loopback endpoint creates `SECRET` capabilities and a remote endpoint creates `PUBLIC` capabilities.
2. Run the focused test and capture RED.
3. Replace the unconditional privacy label with the endpoint-locality result.
4. Run focused tests and capture GREEN.

### Task 3: Prove routing behavior

**Files:**
- Modify: `tests/test_routing.py` or the focused provider test file.

1. Add an integration-level routing test where both local and remote OpenAI-compatible capabilities exist.
2. Assert `local_only=True` assigns only the loopback capability and never the remote one.
3. Run the test RED, make only the minimum correction if needed, then run GREEN.

### Task 4: Document and verify

**Files:**
- Modify: `README.md`.

1. Document that `local_only` accepts only machine-loopback providers; LAN or remote compatible endpoints remain available only when the selected privacy policy permits them.
2. Run focused tests.
3. Run the full pytest suite with coverage.
4. Run Ruff only on changed Python files because the repository has a pre-existing global lint backlog.
5. Run `git diff --check` and a secret scan.
6. Perform one independent final review focused on privacy semantics, IPv4/IPv6 parsing, regressions, and scope.

## Known baseline debt

- 110 tests pass before this change; project coverage is 64%.
- Current GitHub CI fails at `ruff check src/` on Python 3.11 and 3.12.
- The repository has 185 Ruff findings across the full tree and 73 strict mypy errors.
- Those broad failures predate this slice and are not silently fixed here.
