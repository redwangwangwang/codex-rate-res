# Script Reference

## Requirements

- Node.js 18 or newer for built-in `fetch`.
- Docker CLI only when using `--source sub2api`.
- Network access to `https://chatgpt.com`.

## Arguments

- `--source sub2api|file`: Load accounts from a Sub2API Postgres container or from a JSON file.
- `--ids 1,2`: Restrict accounts by id. Required for `consume` and `invite` with `--source sub2api`.
- `--file path`: JSON account file for `--source file`.
- `--action status|credits|eligibility|rules|consume|invite`: Operation to run.
- `--emails a@example.com,b@example.com`: Invite recipients. Required for `invite`.
- `--credit-id id`: Consume a specific credit. If omitted, the first available credit is used.
- `--yes`: Required before the script sends invite or consume POST requests.
- `--base-url url`: Override backend base URL. Defaults to `https://chatgpt.com/backend-api`.
- `--user-agent value`: Override Codex Desktop user agent.
- `--psql-container name`: Defaults to `sub2api-postgres`.
- `--db-user name`: Defaults to `sub2api`.
- `--db-name name`: Defaults to `sub2api`.

## JSON Account Input

`--source file` accepts either one object or an array:

```json
[
  {
    "id": 1,
    "name": "OpenAI OAuth account",
    "email": "owner@example.com",
    "access_token": "eyJ...",
    "chatgpt_account_id": "optional"
  }
]
```

The script also accepts camelCase `accessToken` and `chatgptAccountId`.

## Output

The script emits JSON Lines. Common fields:

- `id`, `name`, `email`
- `action`
- `ok`
- `available_count`
- `available_credit_ids`
- `credit_statuses`
- `code` for consume responses
- `failed_emails` and `message` for invite responses
- `error` for request failures

Secrets are never intentionally emitted.
