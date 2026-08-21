/**
 * Cloudflare Worker: the read-only middleman between the browser build of
 * the FFT analyser and the X2 Cloud API.
 *
 * Why this exists: a page in a browser cannot hold a password (anyone can
 * read it) and cannot call another origin's API (the browser forbids it).
 * This Worker runs on Cloudflare's servers instead, so it can hold the
 * credentials as encrypted secrets and talk to X2 server-side. The page
 * then only ever calls its own origin.
 *
 * READ-ONLY BY CONSTRUCTION
 * ------------------------
 * This is deliberately NOT a path proxy. The client cannot choose what
 * gets requested upstream: this Worker exposes a fixed set of operations,
 * and each one builds its own upstream URL from a constant. There is no
 * code path that can reach the X2 mutating endpoints
 * (POST .../control/{cmd}, POST .../config), so no request from a browser
 * — crafted or otherwise — can disturb a sensor.
 *
 * The only non-GET request made upstream is the login the API requires for
 * a session cookie; it does not reach any device.
 *
 * Routes (all require the access token):
 *   GET /api/health                     config sanity, no secrets returned
 *   GET /api/sites                      every site you can see, with its ID
 *   GET /api/monitors                   vibration monitors at the site
 *   GET /api/waveform?address=<addr>    newest uploaded .bin, raw bytes
 *   GET /api/files?address=<addr>       what that monitor has uploaded
 *
 * /api/sites deliberately does not need X2_SITE_ID: it is how you find out
 * what to put there.
 *
 * Anything else falls through to the static assets in ./public
 */

/**
 * Bumped on every change that alters behaviour. /api/health returns it and
 * the page footer shows its own copy, so "which version am I actually
 * running?" is answerable by looking, not guessing. If the page footer and
 * /api/health disagree, one of the two is stale.
 */
const WORKER_VERSION = "8-vibration-view";

const DEFAULT_BASE = "https://api.x2wireless.com";

/** Product type codes that produce vibration waveforms. */
const VIBRATION_TYPES = new Set([21, 202]);
const VIBRATION_NAMES = ["VIBRATION", "MLT_PURE"];

/* The session cookie is cached in the isolate. Workers reuse isolates
   between requests, so this usually saves a login round-trip; when it does
   not, or when the cookie has expired, we simply log in again. */
let cachedCookie = null;
let cachedCookieAt = 0;
const COOKIE_TTL_MS = 15 * 60 * 1000;

const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extra,
    },
  });

class UpstreamError extends Error {
  constructor(message, status = 502) {
    super(message);
    this.status = status;
  }
}

function baseUrl(env) {
  return (env.X2_BASE_URL || DEFAULT_BASE).replace(/\/+$/, "");
}

/** Constant-time-ish comparison so the token cannot be guessed by timing. */
function tokenMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string") return false;
  if (given.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < given.length; i++) diff |= given.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}

function authorize(request, env) {
  const expected = env.ACCESS_TOKEN;
  if (!expected) {
    throw new UpstreamError(
      "ACCESS_TOKEN is not set on this Worker. Set it before exposing live " +
      "data: npx wrangler secret put ACCESS_TOKEN", 503);
  }
  const header = request.headers.get("authorization") || "";
  const bearer = header.toLowerCase().startsWith("bearer ")
    ? header.slice(7).trim() : "";
  const url = new URL(request.url);
  const given = bearer || url.searchParams.get("token") || "";
  if (!tokenMatches(given, expected)) {
    throw new UpstreamError("Not authorised. Supply the access token.", 401);
  }
}

async function login(env) {
  const body = new URLSearchParams({
    u: env.X2_USERNAME || "",
    p: env.X2_PASSWORD || "",
  });
  const resp = await fetch(`${baseUrl(env)}/login`, {
    method: "POST",
    body,
    headers: { "content-type": "application/x-www-form-urlencoded" },
  });
  if (!resp.ok) {
    throw new UpstreamError(`X2 login returned HTTP ${resp.status}`, 502);
  }
  let payload;
  try {
    payload = await resp.json();
  } catch {
    throw new UpstreamError("X2 login response was not JSON", 502);
  }
  if (payload.Status !== "login_ok") {
    throw new UpstreamError(`X2 rejected the login (${payload.Status})`, 401);
  }
  const cookies = typeof resp.headers.getSetCookie === "function"
    ? resp.headers.getSetCookie()
    : [resp.headers.get("set-cookie")].filter(Boolean);
  if (!cookies.length) {
    throw new UpstreamError("X2 login returned no session cookie", 502);
  }
  // keep just name=value from each cookie
  cachedCookie = cookies.map(c => String(c).split(";")[0]).join("; ");
  cachedCookieAt = Date.now();
  return cachedCookie;
}

async function sessionCookie(env, forceFresh = false) {
  const fresh = cachedCookie && (Date.now() - cachedCookieAt) < COOKIE_TTL_MS;
  if (fresh && !forceFresh) return cachedCookie;
  return login(env);
}

/**
 * Perform one of this Worker's fixed upstream reads.
 *
 * `path` is always built by the caller from a constant plus, at most, a
 * URL-encoded sensor address — never from a client-supplied path.
 */
async function x2Get(env, path, params = {}) {
  const url = new URL(`${baseUrl(env)}/${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  }

  const attempt = async cookie =>
    fetch(url.toString(), { method: "GET", headers: { cookie } });

  let resp = await attempt(await sessionCookie(env));
  if (resp.status === 401 || resp.status === 403) {
    // the cached session probably expired: log in once more, then give up
    resp = await attempt(await sessionCookie(env, true));
  }
  if (resp.status === 403) {
    throw new UpstreamError(
      "X2 refused the request (403). The account may lack permission for " +
      "this site.", 403);
  }
  if (!resp.ok) {
    throw new UpstreamError(`X2 returned HTTP ${resp.status} for ${path}`, 502);
  }
  try {
    return await resp.json();
  } catch {
    throw new UpstreamError(`X2 response for ${path} was not JSON`, 502);
  }
}

/** The API returns either a bare list or an object wrapping one. */
function asList(payload, key) {
  if (Array.isArray(payload)) return payload;
  if (payload && typeof payload === "object") {
    for (const candidate of [key, key.toLowerCase(), "data"]) {
      if (Array.isArray(payload[candidate])) return payload[candidate];
    }
    return [payload];
  }
  return [];
}

/** Mirrors fft_analyser.x2_client._is_vibration. */
function isVibration(address) {
  for (const key of ["TypeInt", "Type", "ProductType", "AddressType"]) {
    const value = address[key];
    if (typeof value === "number" && VIBRATION_TYPES.has(Math.trunc(value))) return true;
    if (typeof value === "string") {
      const upper = value.toUpperCase();
      if (VIBRATION_NAMES.some(n => upper.includes(n))) return true;
    }
  }
  const extras = address.ExtraInfo || [];
  for (const extra of extras) {
    const key = String(extra.ExtraRawKey || "");
    if (key.startsWith("extra_mlt_")) return true;
    if (key === "extra_device_type") {
      const head = String(extra.ExtraValue || "").split(";")[0].trim().toUpperCase();
      if (VIBRATION_NAMES.includes(head) || VIBRATION_TYPES.has(Number(head))) return true;
    }
  }
  return false;
}

/**
 * Which site to read: the request's ?site= parameter wins, falling back to
 * the X2_SITE_ID variable. Letting the client pick is safe because the
 * site ID is not a secret and X2 enforces the real boundary server-side -
 * the account can only ever read sites it has permission for (anything
 * else comes back 403). The page uses /api/sites to offer the choice.
 */
function siteId(env, url) {
  const fromQuery = url && url.searchParams.get("site");
  const site = (fromQuery && fromQuery.trim()) || env.X2_SITE_ID;
  if (!site) {
    throw new UpstreamError(
      "No site selected. Call /api/sites to see which sites this account " +
      "can read, then pass ?site=<id> (the page does this for you) or set " +
      "the X2_SITE_ID variable to pin one.", 503);
  }
  return encodeURIComponent(String(site).trim());
}

/**
 * Every site the account can see. Needs no configuration beyond the
 * credentials, which is what makes it the way to discover X2_SITE_ID.
 */
async function listSites(env) {
  const payload = await x2Get(env, "site");
  const sites = asList(payload, "Sites").map(s => ({
    siteId: s.SiteID ?? s.siteId ?? null,
    name: s.Name || "(unnamed)",
    description: s.Description || null,
    location: s.Location || null,
    sensorCount: s.AddressCount ?? null,
  }));
  return {
    count: sites.length,
    sites,
    next: sites.length
      ? "Set X2_SITE_ID to the siteId you want, then call /api/monitors."
      : "This account cannot see any sites. Check the username and password.",
  };
}

async function listMonitors(env, url) {
  const site = siteId(env, url);
  const payload = await x2Get(env, `site/${site}/address`);
  const addresses = asList(payload, "Addresslist");
  const monitors = addresses.filter(isVibration).map(a => ({
    address: String(a.Address ?? ""),
    name: String(a.Name || a.Address || "unnamed"),
    type: a.TypeInt ?? a.Type ?? null,
    lastUpload: (a.ExtraInfo || [])
      .find(e => e.ExtraRawKey === "extra_last_upload")?.ExtraValue ?? null,
  }));
  return { site: decodeURIComponent(site), count: monitors.length, monitors,
           totalAddresses: addresses.length };
}

async function listFiles(env, url, address) {
  const payload = await x2Get(
    env, `site/${siteId(env, url)}/address/${encodeURIComponent(address)}/files`);
  return asList(payload, "Files")
    .map(f => ({
      name: String(f.name || ""),
      size: Number(f.size || 0),
      changed: f.changed || null,
      link: String(f.pre_signed_link || f.direct_link || ""),
    }))
    .filter(f => f.name.toLowerCase().endsWith(".bin"))
    .sort((a, b) => String(b.changed || "").localeCompare(String(a.changed || "")));
}

async function fetchWaveform(env, url, address) {
  const files = await listFiles(env, url, address);
  if (!files.length) {
    throw new UpstreamError("This monitor has not uploaded a waveform yet.", 404);
  }
  const newest = files[0];
  if (!newest.link) {
    throw new UpstreamError("The newest waveform has no download link.", 502);
  }
  // Pre-signed links expire an hour after listing, which is why we always
  // list immediately before downloading rather than caching the URL.
  const resp = await fetch(newest.link);
  if (!resp.ok) {
    throw new UpstreamError(
      `Downloading ${newest.name} failed (HTTP ${resp.status}). Pre-signed ` +
      "links expire one hour after listing.", 502);
  }
  const bytes = await resp.arrayBuffer();
  return { bytes, meta: newest };
}

async function handleApi(request, env, url) {
  authorize(request, env);

  if (url.pathname === "/api/health") {
    const site = env.X2_SITE_ID || null;
    return json({
      ok: true,
      version: WORKER_VERSION,
      base: baseUrl(env),
      site,
      credentialsConfigured: Boolean(env.X2_USERNAME && env.X2_PASSWORD),
      next: site
        ? "Call /api/monitors to list the vibration monitors at this site."
        : "No default site pinned — the page picks one from /api/sites; " +
          "set X2_SITE_ID only if you want to pin a single site.",
      note: "Read-only. This Worker can only list sites and monitors, and " +
            "download waveforms the sensors already uploaded.",
    });
  }

  if (url.pathname === "/api/sites") {
    return json(await listSites(env));
  }

  if (url.pathname === "/api/monitors") {
    return json(await listMonitors(env, url));
  }

  const address = url.searchParams.get("address");
  if (url.pathname === "/api/files") {
    if (!address) throw new UpstreamError("address parameter is required", 400);
    return json({ address, files: await listFiles(env, url, address) });
  }

  if (url.pathname === "/api/waveform") {
    if (!address) throw new UpstreamError("address parameter is required", 400);
    const { bytes, meta } = await fetchWaveform(env, url, address);
    return new Response(bytes, {
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "no-store",
        "x-waveform-name": meta.name,
        "x-waveform-changed": String(meta.changed || ""),
        "x-waveform-size": String(meta.size || bytes.byteLength),
      },
    });
  }

  throw new UpstreamError("No such endpoint", 404);
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (!url.pathname.startsWith("/api/")) {
      // everything else is the static analyser page
      return env.ASSETS.fetch(request);
    }

    if (request.method !== "GET") {
      return json({ error: "This API is read-only; only GET is accepted." },
                  405, { allow: "GET" });
    }

    try {
      return await handleApi(request, env, url);
    } catch (err) {
      const status = err instanceof UpstreamError ? err.status : 500;
      // never echo the upstream body or any secret back to the browser
      return json({ error: err.message || "Unexpected error" }, status);
    }
  },
};
