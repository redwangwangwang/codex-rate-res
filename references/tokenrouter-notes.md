# TokenRouter Codex Invite Reset Notes

TokenRouter implements a Codex invite/reset service that calls ChatGPT backend endpoints with an OpenAI OAuth access token. The important observed constants and paths are:

- Base URL: `https://chatgpt.com/backend-api`
- Referral key: `codex_referral_persistent_invite`
- Eligibility: `GET /referrals/invite/eligibility?referral_key=codex_referral_persistent_invite`
- Rules: `GET /wham/referrals/eligibility_rules?referral_key=codex_referral_persistent_invite`
- Credits: `GET /wham/rate-limit-reset-credits`
- Consume: `POST /wham/rate-limit-reset-credits/consume` with `credit_id` and `redeem_request_id`
- Invite: `POST /wham/referrals/invite` with `referral_key` and `emails`

Observed request headers:

- `Authorization: Bearer <access_token>`
- `Accept: application/json`
- `Content-Type: application/json` for POST
- `OAI-Language: zh-CN`
- `originator: Codex Desktop`
- `X-OpenAI-Attach-Auth: 1`
- `X-OpenAI-Attach-Integrity-State: 1`
- `User-Agent: Codex Desktop/...`
- `chatgpt-account-id: <id>` when present

The feature consumes official banked rate-limit reset credits. It does not create credits by itself.
