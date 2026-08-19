var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// worker/index.js
var DEFAULT_BASE = "https://api.x2wireless.com";
var VIBRATION_TYPES = /* @__PURE__ */ new Set([21, 202]);
var VIBRATION_NAMES = ["VIBRATION", "MLT_PURE"];
var cachedCookie = null;
var cachedCookieAt = 0;
var COOKIE_TTL_MS = 15 * 60 * 1e3;
var json = /* @__PURE__ */ __name((body, status = 200, extra = {}) => new Response(JSON.stringify(body, null, 2), {
  status,
  headers: {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...extra
  }
}), "json");
var UpstreamError = class extends Error {
  static {
    __name(this, "UpstreamError");
  }
  constructor(message, status = 502) {
    super(message);
    this.status = status;
  }
};
function baseUrl(env) {
  return (env.X2_BASE_URL || DEFAULT_BASE).replace(/\/+$/, "");
}
__name(baseUrl, "baseUrl");
function tokenMatches(given, expected) {
  if (typeof given !== "string" || typeof expected !== "string") return false;
  if (given.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < given.length; i++) diff |= given.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}
__name(tokenMatches, "tokenMatches");
function authorize(request, env) {
  const expected = env.ACCESS_TOKEN;
  if (!expected) {
    throw new UpstreamError(
      "ACCESS_TOKEN is not set on this Worker. Set it before exposing live data: npx wrangler secret put ACCESS_TOKEN",
      503
    );
  }
  const header = request.headers.get("authorization") || "";
  const bearer = header.toLowerCase().startsWith("bearer ") ? header.slice(7).trim() : "";
  const url = new URL(request.url);
  const given = bearer || url.searchParams.get("token") || "";
  if (!tokenMatches(given, expected)) {
    throw new UpstreamError("Not authorised. Supply the access token.", 401);
  }
}
__name(authorize, "authorize");
async function login(env) {
  const body = new URLSearchParams({
    u: env.X2_USERNAME || "",
    p: env.X2_PASSWORD || ""
  });
  const resp = await fetch(`${baseUrl(env)}/login`, {
    method: "POST",
    body,
    headers: { "content-type": "application/x-www-form-urlencoded" }
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
  const cookies = typeof resp.headers.getSetCookie === "function" ? resp.headers.getSetCookie() : [resp.headers.get("set-cookie")].filter(Boolean);
  if (!cookies.length) {
    throw new UpstreamError("X2 login returned no session cookie", 502);
  }
  cachedCookie = cookies.map((c) => String(c).split(";")[0]).join("; ");
  cachedCookieAt = Date.now();
  return cachedCookie;
}
__name(login, "login");
async function sessionCookie(env, forceFresh = false) {
  const fresh = cachedCookie && Date.now() - cachedCookieAt < COOKIE_TTL_MS;
  if (fresh && !forceFresh) return cachedCookie;
  return login(env);
}
__name(sessionCookie, "sessionCookie");
async function x2Get(env, path, params = {}) {
  const url = new URL(`${baseUrl(env)}/${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v !== void 0 && v !== null && v !== "") url.searchParams.set(k, v);
  }
  const attempt = /* @__PURE__ */ __name(async (cookie) => fetch(url.toString(), { method: "GET", headers: { cookie } }), "attempt");
  let resp = await attempt(await sessionCookie(env));
  if (resp.status === 401 || resp.status === 403) {
    resp = await attempt(await sessionCookie(env, true));
  }
  if (resp.status === 403) {
    throw new UpstreamError(
      "X2 refused the request (403). The account may lack permission for this site.",
      403
    );
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
__name(x2Get, "x2Get");
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
__name(asList, "asList");
function isVibration(address) {
  for (const key of ["TypeInt", "Type", "ProductType", "AddressType"]) {
    const value = address[key];
    if (typeof value === "number" && VIBRATION_TYPES.has(Math.trunc(value))) return true;
    if (typeof value === "string") {
      const upper = value.toUpperCase();
      if (VIBRATION_NAMES.some((n) => upper.includes(n))) return true;
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
__name(isVibration, "isVibration");
function siteId(env) {
  const site = env.X2_SITE_ID;
  if (!site) throw new UpstreamError("X2_SITE_ID is not configured", 503);
  return encodeURIComponent(String(site));
}
__name(siteId, "siteId");
async function listMonitors(env) {
  const payload = await x2Get(env, `site/${siteId(env)}/address`);
  const addresses = asList(payload, "Addresslist");
  const monitors = addresses.filter(isVibration).map((a) => ({
    address: String(a.Address ?? ""),
    name: String(a.Name || a.Address || "unnamed"),
    type: a.TypeInt ?? a.Type ?? null,
    lastUpload: (a.ExtraInfo || []).find((e) => e.ExtraRawKey === "extra_last_upload")?.ExtraValue ?? null
  }));
  return {
    site: env.X2_SITE_ID,
    count: monitors.length,
    monitors,
    totalAddresses: addresses.length
  };
}
__name(listMonitors, "listMonitors");
async function listFiles(env, address) {
  const payload = await x2Get(
    env,
    `site/${siteId(env)}/address/${encodeURIComponent(address)}/files`
  );
  return asList(payload, "Files").map((f) => ({
    name: String(f.name || ""),
    size: Number(f.size || 0),
    changed: f.changed || null,
    link: String(f.pre_signed_link || f.direct_link || "")
  })).filter((f) => f.name.toLowerCase().endsWith(".bin")).sort((a, b) => String(b.changed || "").localeCompare(String(a.changed || "")));
}
__name(listFiles, "listFiles");
async function fetchWaveform(env, address) {
  const files = await listFiles(env, address);
  if (!files.length) {
    throw new UpstreamError("This monitor has not uploaded a waveform yet.", 404);
  }
  const newest = files[0];
  if (!newest.link) {
    throw new UpstreamError("The newest waveform has no download link.", 502);
  }
  const resp = await fetch(newest.link);
  if (!resp.ok) {
    throw new UpstreamError(
      `Downloading ${newest.name} failed (HTTP ${resp.status}). Pre-signed links expire one hour after listing.`,
      502
    );
  }
  const bytes = await resp.arrayBuffer();
  return { bytes, meta: newest };
}
__name(fetchWaveform, "fetchWaveform");
async function handleApi(request, env, url) {
  authorize(request, env);
  if (url.pathname === "/api/health") {
    return json({
      ok: true,
      base: baseUrl(env),
      site: env.X2_SITE_ID || null,
      credentialsConfigured: Boolean(env.X2_USERNAME && env.X2_PASSWORD),
      note: "Read-only. This Worker can only list monitors and download waveforms the sensors already uploaded."
    });
  }
  if (url.pathname === "/api/monitors") {
    return json(await listMonitors(env));
  }
  const address = url.searchParams.get("address");
  if (url.pathname === "/api/files") {
    if (!address) throw new UpstreamError("address parameter is required", 400);
    return json({ address, files: await listFiles(env, address) });
  }
  if (url.pathname === "/api/waveform") {
    if (!address) throw new UpstreamError("address parameter is required", 400);
    const { bytes, meta } = await fetchWaveform(env, address);
    return new Response(bytes, {
      headers: {
        "content-type": "application/octet-stream",
        "cache-control": "no-store",
        "x-waveform-name": meta.name,
        "x-waveform-changed": String(meta.changed || ""),
        "x-waveform-size": String(meta.size || bytes.byteLength)
      }
    });
  }
  throw new UpstreamError("No such endpoint", 404);
}
__name(handleApi, "handleApi");
var worker_default = {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }
    if (request.method !== "GET") {
      return json(
        { error: "This API is read-only; only GET is accepted." },
        405,
        { allow: "GET" }
      );
    }
    try {
      return await handleApi(request, env, url);
    } catch (err) {
      const status = err instanceof UpstreamError ? err.status : 500;
      return json({ error: err.message || "Unexpected error" }, status);
    }
  }
};

// ../../../root/.npm/_npx/c943b712072b77c4/node_modules/wrangler/templates/middleware/middleware-ensure-req-body-drained.ts
var drainBody = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } finally {
    try {
      if (request.body !== null && !request.bodyUsed) {
        const reader = request.body.getReader();
        while (!(await reader.read()).done) {
        }
      }
    } catch (e) {
      console.error("Failed to drain the unused request body.", e);
    }
  }
}, "drainBody");
var middleware_ensure_req_body_drained_default = drainBody;

// ../../../root/.npm/_npx/c943b712072b77c4/node_modules/wrangler/templates/middleware/middleware-miniflare3-json-error.ts
function reduceError(e) {
  return {
    name: e?.name,
    message: e?.message ?? String(e),
    stack: e?.stack,
    cause: e?.cause === void 0 ? void 0 : reduceError(e.cause)
  };
}
__name(reduceError, "reduceError");
var jsonError = /* @__PURE__ */ __name(async (request, env, _ctx, middlewareCtx) => {
  try {
    return await middlewareCtx.next(request, env);
  } catch (e) {
    const error = reduceError(e);
    const body = JSON.stringify(error);
    const headers = {
      "Content-Type": "application/json",
      "MF-Experimental-Error-Stack": "true"
    };
    const encoded = encodeURIComponent(body);
    if (encoded.length <= 8192) {
      headers["MF-Experimental-Error-Stack-Payload"] = encoded;
    }
    return new Response(body, { status: 500, headers });
  }
}, "jsonError");
var middleware_miniflare3_json_error_default = jsonError;

// .wrangler/tmp/bundle-j1kLAk/middleware-insertion-facade.js
var __INTERNAL_WRANGLER_MIDDLEWARE__ = [
  middleware_ensure_req_body_drained_default,
  middleware_miniflare3_json_error_default
];
var middleware_insertion_facade_default = worker_default;

// ../../../root/.npm/_npx/c943b712072b77c4/node_modules/wrangler/templates/middleware/common.ts
var __facade_middleware__ = [];
function __facade_register__(...args) {
  __facade_middleware__.push(...args.flat());
}
__name(__facade_register__, "__facade_register__");
function __facade_invokeChain__(request, env, ctx, dispatch, middlewareChain) {
  const [head, ...tail] = middlewareChain;
  const middlewareCtx = {
    dispatch,
    next(newRequest, newEnv) {
      return __facade_invokeChain__(newRequest, newEnv, ctx, dispatch, tail);
    }
  };
  return head(request, env, ctx, middlewareCtx);
}
__name(__facade_invokeChain__, "__facade_invokeChain__");
function __facade_invoke__(request, env, ctx, dispatch, finalMiddleware) {
  return __facade_invokeChain__(request, env, ctx, dispatch, [
    ...__facade_middleware__,
    finalMiddleware
  ]);
}
__name(__facade_invoke__, "__facade_invoke__");

// .wrangler/tmp/bundle-j1kLAk/middleware-loader.entry.ts
var __Facade_ScheduledController__ = class ___Facade_ScheduledController__ {
  constructor(scheduledTime, cron, noRetry) {
    this.scheduledTime = scheduledTime;
    this.cron = cron;
    this.#noRetry = noRetry;
  }
  scheduledTime;
  cron;
  static {
    __name(this, "__Facade_ScheduledController__");
  }
  #noRetry;
  noRetry() {
    if (!(this instanceof ___Facade_ScheduledController__)) {
      throw new TypeError("Illegal invocation");
    }
    this.#noRetry();
  }
};
function wrapExportedHandler(worker) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return worker;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  const fetchDispatcher = /* @__PURE__ */ __name(function(request, env, ctx) {
    if (worker.fetch === void 0) {
      throw new Error("Handler does not export a fetch() function.");
    }
    return worker.fetch(request, env, ctx);
  }, "fetchDispatcher");
  return {
    ...worker,
    fetch(request, env, ctx) {
      const dispatcher = /* @__PURE__ */ __name(function(type, init) {
        if (type === "scheduled" && worker.scheduled !== void 0) {
          const controller = new __Facade_ScheduledController__(
            Date.now(),
            init.cron ?? "",
            () => {
            }
          );
          return worker.scheduled(controller, env, ctx);
        }
      }, "dispatcher");
      return __facade_invoke__(request, env, ctx, dispatcher, fetchDispatcher);
    }
  };
}
__name(wrapExportedHandler, "wrapExportedHandler");
function wrapWorkerEntrypoint(klass) {
  if (__INTERNAL_WRANGLER_MIDDLEWARE__ === void 0 || __INTERNAL_WRANGLER_MIDDLEWARE__.length === 0) {
    return klass;
  }
  for (const middleware of __INTERNAL_WRANGLER_MIDDLEWARE__) {
    __facade_register__(middleware);
  }
  return class extends klass {
    #fetchDispatcher = /* @__PURE__ */ __name((request, env, ctx) => {
      this.env = env;
      this.ctx = ctx;
      if (super.fetch === void 0) {
        throw new Error("Entrypoint class does not define a fetch() function.");
      }
      return super.fetch(request);
    }, "#fetchDispatcher");
    #dispatcher = /* @__PURE__ */ __name((type, init) => {
      if (type === "scheduled" && super.scheduled !== void 0) {
        const controller = new __Facade_ScheduledController__(
          Date.now(),
          init.cron ?? "",
          () => {
          }
        );
        return super.scheduled(controller);
      }
    }, "#dispatcher");
    fetch(request) {
      return __facade_invoke__(
        request,
        this.env,
        this.ctx,
        this.#dispatcher,
        this.#fetchDispatcher
      );
    }
  };
}
__name(wrapWorkerEntrypoint, "wrapWorkerEntrypoint");
var WRAPPED_ENTRY;
if (typeof middleware_insertion_facade_default === "object") {
  WRAPPED_ENTRY = wrapExportedHandler(middleware_insertion_facade_default);
} else if (typeof middleware_insertion_facade_default === "function") {
  WRAPPED_ENTRY = wrapWorkerEntrypoint(middleware_insertion_facade_default);
}
var middleware_loader_entry_default = WRAPPED_ENTRY;
export {
  __INTERNAL_WRANGLER_MIDDLEWARE__,
  middleware_loader_entry_default as default
};
//# sourceMappingURL=index.js.map
