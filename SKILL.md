---
name: codex-rate-reset
description: Query and use ChatGPT/Codex rate-limit reset credits, referral invite eligibility, eligibility rules, and invite sending for OpenAI OAuth accounts. Use when Codex needs to inspect or consume Codex reset credits, send Codex referral invitations, or reproduce TokenRouter-style Codex invite/reset backend calls without deploying TokenRouter, especially against a Sub2API deployment with OpenAI OAuth accounts.
---

# Codex Rate Reset

Use this skill to operate the Codex Desktop invite/reset backend flow directly from stored OpenAI OAuth account credentials. This skill intentionally does not deploy TokenRouter and does not modify local Sub2API account state unless the user explicitly asks for that separate cleanup.

## Core Rules

- Treat OpenAI OAuth access tokens, refresh tokens, ID tokens, and `chatgpt-account-id` values as secrets. Do not print them.
- Prefer read-only `status`, `rules`, and `eligibility` before `consume` or `invite`.
- Consume a reset credit only for accounts the user explicitly targets or for accounts identified as quota-full by local evidence.
- Do not claim this is a public OpenAI API. It uses ChatGPT backend endpoints observed from Codex Desktop behavior.
- Do not clear Sub2API `rate_limited_at`, `rate_limit_reset_at`, Redis scheduler keys, or local quota caches as part of this skill unless the user separately asks for local state cleanup.
- If the user asks to “reset Codex rate” for Sub2API accounts, find the affected OpenAI OAuth accounts first, then run `status`, then `consume` only where `available_count > 0`.

## Script

Use `scripts/codex_rate_reset.mjs` for deterministic calls. Resolve the script path relative to this skill directory, for example:

```bash
SKILL_DIR="${CODEX_HOME:-$HOME/.codex}/skills/codex-rate-reset"
node "$SKILL_DIR/scripts/codex_rate_reset.mjs" --help
```

For all script arguments, JSON input format, and JSON Lines output fields, read `references/script-reference.md`.

Common Sub2API examples:

```bash
node "$SKILL_DIR/scripts/codex_rate_reset.mjs" \
  --source sub2api --ids 46,53 --action status
```

```bash
node "$SKILL_DIR/scripts/codex_rate_reset.mjs" \
  --source sub2api --ids 46,53 --action consume --yes
```

```bash
node "$SKILL_DIR/scripts/codex_rate_reset.mjs" \
  --source sub2api --ids 46 --action invite --emails user@example.com,other@example.com --yes
```

Generic JSON account file:

```bash
node "$SKILL_DIR/scripts/codex_rate_reset.mjs" \
  --source file --file /path/to/accounts.json --action status
```

JSON format:

```json
[
  {
    "id": 46,
    "name": "account label",
    "email": "owner@example.com",
    "access_token": "eyJ...",
    "chatgpt_account_id": "optional-account-id"
  }
]
```

## Supported Actions

- `status`: query invite eligibility, eligibility rules, and reset credits.
- `credits`: query only `/wham/rate-limit-reset-credits`.
- `eligibility`: query only `/referrals/invite/eligibility`.
- `rules`: query only `/wham/referrals/eligibility_rules`.
- `consume`: query credits and consume the first available credit unless `--credit-id` is provided; requires `--yes`.
- `invite`: send Codex referral invitations through `/wham/referrals/invite`; requires `--emails` and `--yes`.

For `--source sub2api`, `consume` and `invite` require `--ids` so the script cannot accidentally mutate all OpenAI OAuth accounts.

## Backend Endpoints

The script uses `https://chatgpt.com/backend-api`:

- `GET /referrals/invite/eligibility?referral_key=codex_referral_persistent_invite`
- `GET /wham/referrals/eligibility_rules?referral_key=codex_referral_persistent_invite`
- `GET /wham/rate-limit-reset-credits`
- `POST /wham/rate-limit-reset-credits/consume`
- `POST /wham/referrals/invite`

Headers emulate Codex Desktop:

- `Authorization: Bearer <access_token>`
- `originator: Codex Desktop`
- `X-OpenAI-Attach-Auth: 1`
- `X-OpenAI-Attach-Integrity-State: 1`
- `User-Agent: Codex Desktop/0.0.0 (Linux; x86_64)` unless `--user-agent` is set
- `chatgpt-account-id` when available

## Finding Quota-Full Sub2API Accounts

For a live Sub2API Docker Compose deployment, first confirm the containers and schema:

```bash
docker compose ps
docker exec sub2api-postgres psql -U sub2api -d sub2api -Atc \
  "select id, name, credentials->>'email', extra->>'codex_5h_used_percent', extra->>'codex_7d_used_percent', extra->>'codex_7d_reset_at', rate_limit_reset_at from accounts where deleted_at is null and platform='openai' and type='oauth' order by id;"
```

Targets are usually OpenAI OAuth accounts where `extra->>'codex_7d_used_percent' = '100'`, `extra->>'codex_5h_used_percent' = '100'`, or `rate_limit_reset_at > now()`. Verify with `status` before consuming credits.

## Output Handling

The script prints JSON lines with account id, name, email, eligibility status, available count, credit ids, and action result. It never prints tokens. Summarize results to the user without exposing raw credentials.
