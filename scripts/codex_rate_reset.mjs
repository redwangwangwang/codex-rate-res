#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";

const REFERRAL_KEY = "codex_referral_persistent_invite";
const DEFAULT_BASE_URL = "https://chatgpt.com/backend-api";
const DEFAULT_USER_AGENT = "Codex Desktop/0.0.0 (Linux; x86_64)";
const MAX_INVITE_EMAILS = 5;

function parseArgs(argv) {
  const args = {
    source: "sub2api",
    action: "status",
    ids: [],
    file: "",
    emails: [],
    creditId: "",
    yes: false,
    baseUrl: DEFAULT_BASE_URL,
    userAgent: DEFAULT_USER_AGENT,
    psqlContainer: "sub2api-postgres",
    dbUser: "sub2api",
    dbName: "sub2api",
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`missing value for ${arg}`);
      return argv[i];
    };
    switch (arg) {
      case "--source":
        args.source = next();
        break;
      case "--action":
        args.action = next();
        break;
      case "--ids":
        args.ids = splitList(next()).map((value) => Number.parseInt(value, 10)).filter(Number.isFinite);
        break;
      case "--file":
        args.file = next();
        break;
      case "--emails":
        args.emails = splitList(next());
        break;
      case "--credit-id":
        args.creditId = next().trim();
        break;
      case "--yes":
        args.yes = true;
        break;
      case "--base-url":
        args.baseUrl = next().replace(/\/+$/, "");
        break;
      case "--user-agent":
        args.userAgent = next();
        break;
      case "--psql-container":
        args.psqlContainer = next();
        break;
      case "--db-user":
        args.dbUser = next();
        break;
      case "--db-name":
        args.dbName = next();
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return args;
}

function splitList(value) {
  return String(value || "")
    .split(/[,\s;]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function printHelp() {
  console.log(`Usage:
  codex_rate_reset.mjs --source sub2api --ids 46,53 --action status
  codex_rate_reset.mjs --source sub2api --ids 46 --action consume --yes
  codex_rate_reset.mjs --source sub2api --ids 46 --action invite --emails a@example.com --yes
  codex_rate_reset.mjs --source file --file accounts.json --action status

Actions: status, credits, eligibility, rules, consume, invite
Add --yes for consume and invite.
`);
}

function psql(args, sql) {
  return execFileSync(
    "docker",
    [
      "exec",
      args.psqlContainer,
      "psql",
      "-U",
      args.dbUser,
      "-d",
      args.dbName,
      "-At",
      "-F",
      "\t",
      "-c",
      sql,
    ],
    { encoding: "utf8", maxBuffer: 20 * 1024 * 1024 },
  );
}

function loadAccounts(args) {
  if (args.source === "file") {
    if (!args.file) throw new Error("--file is required for --source file");
    const data = JSON.parse(fs.readFileSync(args.file, "utf8"));
    const rows = Array.isArray(data) ? data : [data];
    return rows.map(normalizeAccount).filter((account) => {
      return args.ids.length === 0 || args.ids.includes(Number(account.id));
    });
  }
  if (args.source !== "sub2api") {
    throw new Error("--source must be sub2api or file");
  }
  const idFilter = args.ids.length > 0 ? `and id = any(array[${args.ids.join(",")}])` : "";
  const sql = `
select
  id,
  name,
  coalesce(credentials->>'email', extra->>'email', '') as email,
  credentials->>'access_token' as access_token,
  coalesce(credentials->>'chatgpt_account_id', '') as chatgpt_account_id
from accounts
where deleted_at is null
  and platform = 'openai'
  and type = 'oauth'
  ${idFilter}
order by id;
`;
  return psql(args, sql)
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [id, name, email, accessToken, chatgptAccountId] = line.split("\t");
      return normalizeAccount({
        id: Number(id),
        name,
        email,
        access_token: accessToken,
        chatgpt_account_id: chatgptAccountId,
      });
    });
}

function normalizeAccount(raw) {
  return {
    id: raw.id ?? raw.account_id ?? "",
    name: raw.name ?? "",
    email: raw.email ?? "",
    accessToken: raw.access_token ?? raw.accessToken ?? "",
    chatgptAccountId: raw.chatgpt_account_id ?? raw.chatgptAccountId ?? "",
  };
}

function requestHeaders(args, account) {
  const headers = {
    Authorization: `Bearer ${account.accessToken}`,
    Accept: "application/json",
    "Content-Type": "application/json",
    "OAI-Language": "zh-CN",
    originator: "Codex Desktop",
    "X-OpenAI-Attach-Auth": "1",
    "X-OpenAI-Attach-Integrity-State": "1",
    "User-Agent": args.userAgent,
  };
  if (account.chatgptAccountId) headers["chatgpt-account-id"] = account.chatgptAccountId;
  return headers;
}

async function requestJson(args, account, method, path, body) {
  const response = await fetch(`${args.baseUrl}${path}`, {
    method,
    headers: requestHeaders(args, account),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  let json = {};
  if (text.trim()) {
    try {
      json = JSON.parse(text);
    } catch {
      json = { raw_text: text.slice(0, 500) };
    }
  }
  if (!response.ok) {
    const message = json?.detail || json?.message || json?.error?.message || text.slice(0, 500) || response.statusText;
    const error = new Error(`${response.status} ${response.statusText}: ${message}`);
    error.status = response.status;
    throw error;
  }
  return json;
}

function publicAccount(account) {
  return {
    id: account.id,
    name: account.name,
    email: account.email,
  };
}

function availableCredits(payload) {
  const credits = Array.isArray(payload?.credits) ? payload.credits : [];
  return credits.filter((credit) => {
    const status = String(credit?.status || "available").toLowerCase();
    return credit?.id && (!status || status === "available");
  });
}

function normalizeEmails(emails) {
  const seen = new Set();
  const result = [];
  for (const raw of emails) {
    const email = String(raw).trim();
    if (!email) continue;
    const key = email.toLowerCase();
    if (seen.has(key)) continue;
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      throw new Error(`invalid email: ${email}`);
    }
    seen.add(key);
    result.push(email);
  }
  if (result.length === 0) throw new Error("--emails is required for invite");
  if (result.length > MAX_INVITE_EMAILS) throw new Error(`at most ${MAX_INVITE_EMAILS} emails can be invited at once`);
  return result;
}

async function safeCall(label, fn) {
  try {
    return { ok: true, value: await fn() };
  } catch (error) {
    return { ok: false, error: String(error.message || error), label };
  }
}

async function handleAccount(args, account) {
  if (!account.accessToken) {
    console.log(JSON.stringify({ ...publicAccount(account), action: args.action, ok: false, error: "missing_access_token" }));
    return;
  }

  if (args.action === "credits") {
    const credits = await requestJson(args, account, "GET", "/wham/rate-limit-reset-credits");
    console.log(JSON.stringify(formatCredits(account, credits)));
    return;
  }

  if (args.action === "eligibility") {
    const eligibility = await requestJson(args, account, "GET", `/referrals/invite/eligibility?referral_key=${encodeURIComponent(REFERRAL_KEY)}`);
    console.log(JSON.stringify({ ...publicAccount(account), action: "eligibility", ok: true, eligibility }));
    return;
  }

  if (args.action === "rules") {
    const rules = await requestJson(args, account, "GET", `/wham/referrals/eligibility_rules?referral_key=${encodeURIComponent(REFERRAL_KEY)}`);
    console.log(JSON.stringify({ ...publicAccount(account), action: "rules", ok: true, rules }));
    return;
  }

  if (args.action === "status") {
    const eligibility = await safeCall("eligibility", () => requestJson(args, account, "GET", `/referrals/invite/eligibility?referral_key=${encodeURIComponent(REFERRAL_KEY)}`));
    const rules = await safeCall("rules", () => requestJson(args, account, "GET", `/wham/referrals/eligibility_rules?referral_key=${encodeURIComponent(REFERRAL_KEY)}`));
    const credits = await requestJson(args, account, "GET", "/wham/rate-limit-reset-credits");
    console.log(JSON.stringify({
      ...formatCredits(account, credits),
      action: "status",
      eligibility: eligibility.ok ? eligibility.value : { error: eligibility.error },
      rules: rules.ok ? rules.value : { error: rules.error },
    }));
    return;
  }

  if (args.action === "consume") {
    const creditsPayload = await requestJson(args, account, "GET", "/wham/rate-limit-reset-credits");
    const credits = availableCredits(creditsPayload);
    const creditId = args.creditId || credits[0]?.id || "";
    console.log(JSON.stringify(formatCredits(account, creditsPayload)));
    if (!creditId) {
      console.log(JSON.stringify({ ...publicAccount(account), action: "consume", ok: false, reason: "no_available_credit" }));
      return;
    }
    if (!args.yes) {
      console.log(JSON.stringify({
        ...publicAccount(account),
        action: "consume",
        ok: false,
        reason: "confirmation_required",
        hint: "rerun with --yes to consume this credit",
        credit_id: creditId,
      }));
      return;
    }
    const result = await requestJson(args, account, "POST", "/wham/rate-limit-reset-credits/consume", {
      credit_id: creditId,
      redeem_request_id: crypto.randomUUID(),
    });
    console.log(JSON.stringify({
      ...publicAccount(account),
      action: "consume",
      ok: true,
      credit_id: creditId,
      code: result?.code ?? null,
      available_count: result?.available_count ?? null,
      remaining_credit_count: Array.isArray(result?.credits) ? result.credits.length : null,
    }));
    return;
  }

  if (args.action === "invite") {
    const emails = normalizeEmails(args.emails);
    if (!args.yes) {
      console.log(JSON.stringify({
        ...publicAccount(account),
        action: "invite",
        ok: false,
        reason: "confirmation_required",
        hint: "rerun with --yes to send invitations",
        emails,
      }));
      return;
    }
    const result = await requestJson(args, account, "POST", "/wham/referrals/invite", {
      referral_key: REFERRAL_KEY,
      emails,
    });
    console.log(JSON.stringify({
      ...publicAccount(account),
      action: "invite",
      ok: true,
      invited_count: Array.isArray(result?.invites) ? result.invites.length : null,
      failed_emails: Array.isArray(result?.failed_emails) ? result.failed_emails : [],
      message: result?.message ?? "",
    }));
    return;
  }

  throw new Error(`unsupported action: ${args.action}`);
}

function formatCredits(account, payload) {
  const credits = Array.isArray(payload?.credits) ? payload.credits : [];
  const available = availableCredits(payload);
  return {
    ...publicAccount(account),
    action: "credits",
    ok: true,
    available_count: payload?.available_count ?? available.length,
    credit_count: credits.length,
    available_credit_ids: available.map((credit) => credit.id),
    credit_statuses: credits.map((credit) => ({
      id: credit?.id ?? "",
      status: credit?.status ?? "",
      title: credit?.title ?? "",
      description: credit?.description ?? "",
    })),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if ((args.action === "consume" || args.action === "invite") && args.source === "sub2api" && args.ids.length === 0) {
    throw new Error("--ids is required for consume and invite with --source sub2api");
  }
  const accounts = loadAccounts(args);
  if (accounts.length === 0) throw new Error("no accounts found");
  for (const account of accounts) {
    try {
      await handleAccount(args, account);
    } catch (error) {
      console.log(JSON.stringify({ ...publicAccount(account), action: args.action, ok: false, error: String(error.message || error) }));
    }
  }
}

main().catch((error) => {
  console.error(String(error.message || error));
  process.exit(1);
});
