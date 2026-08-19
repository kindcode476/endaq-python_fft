# Setting up the live-data middleman

This turns the deployed analyser page into something that can pull
waveforms from your real monitors.

## Why a middleman is needed at all

The page runs in a browser, and a browser page can't do two things it
would need to: it can't keep your X2 password secret (anyone who opens the
page can read whatever is in it), and it isn't allowed to call another
company's API directly (browsers block that unless the API explicitly
permits it, which X2 doesn't).

A Worker runs on Cloudflare's servers instead of in the browser. It can
hold the password safely and call X2 on the page's behalf. The page then
only ever talks to its own address.

**It stays read-only.** The Worker exposes a fixed set of operations — list
monitors, list files, download a waveform. It is not a general proxy: the
browser cannot ask it to call an arbitrary X2 address, so there is no route
to the endpoints that change device settings or operate relays. Nothing
asks a sensor to measure; it reads files the sensors already uploaded.

---

## Step 1 — Get a terminal in the repo

```bash
git clone https://github.com/kindcode476/endaq-python_fft.git
cd endaq-python_fft
```

You need Node installed (for `npx`). Nothing else.

## Step 2 — Log in to Cloudflare

```bash
npx wrangler login
```

A browser window opens; approve the access. This links the terminal to your
Cloudflare account.

## Step 3 — Set the four values

Three are secret and one is not, but they all go in the same place:
**Cloudflare**. None of them goes in a file on your computer, and none of
them belongs in this repo.

| Name | What it is | Secret? |
|---|---|---|
| `X2_USERNAME` | your X2 login name | yes |
| `X2_PASSWORD` | your X2 password | yes |
| `ACCESS_TOKEN` | a password you invent, to protect your own page | yes |

**You do not need to know your site number.** After connecting, the page
lists every site the account can see and you pick one with a click. Set an
`X2_SITE_ID` variable (Text, not secret) only if you want to pin a default
site; `/api/sites?token=YOUR_TOKEN` shows the IDs.

### Either: in the browser (no terminal needed)

1. Go to **dash.cloudflare.com** and sign in.
2. Open **Workers & Pages**, then click **endaq-python-fft**.
3. Open the **Settings** tab, then **Variables and Secrets**.
4. For each of the four rows above: **Add**, type the name exactly as
   written, paste the value, and choose **Secret** (or tick **Encrypt**)
   for the three marked secret. Choose **Text** for `X2_SITE_ID`.
5. **Save** / **Deploy**.

Once saved, a secret can never be read back — not by you, not by anyone
with repo access. You can only replace it. That is the point.

### Or: from a terminal

```bash
npx wrangler secret put X2_USERNAME     # prompts, then encrypts
npx wrangler secret put X2_PASSWORD
npx wrangler secret put ACCESS_TOKEN
```

Each command asks for the value rather than taking it on the command line,
so it never lands in your shell history. Set `X2_SITE_ID` in the dashboard
as above, or add it to `wrangler.jsonc` under `vars` if you would rather
keep it in version control.

> **Careful:** anything listed under `vars` in `wrangler.jsonc` is rewritten
> on every deploy. `X2_SITE_ID` is deliberately not listed there, so a
> deploy will not wipe the value you set in the dashboard.

`ACCESS_TOKEN` is the password for *your page*. Without it, anyone who
found the URL could read your machines' vibration data. Invent a long
random one — this generates a good one:

```bash
openssl rand -base64 24
```

Keep a copy; you'll paste it into the page in step 7.

**Until `ACCESS_TOKEN` is set, live data simply refuses to work.** The
static page still loads and still analyses files you drag onto it. That is
deliberate: it fails closed, not open.

## Step 5 — Deploy

```bash
npx wrangler deploy
```

The `name` in `wrangler.jsonc` is already set to `endaq-python-fft`, which
matches the Worker you deployed before, so this updates it rather than
creating a second one at a new URL.

## Step 6 — Check it can reach X2

Each of these can be opened straight in a browser by adding
`?token=YOUR_ACCESS_TOKEN`, or called with curl:

```bash
BASE=https://endaq-python-fft.restless-bush-9121.workers.dev
AUTH="Authorization: Bearer YOUR_ACCESS_TOKEN"

curl -H "$AUTH" $BASE/api/health      # is it configured?
curl -H "$AUTH" $BASE/api/sites       # which sites can this account see?
curl -H "$AUTH" $BASE/api/monitors    # which monitors at the chosen site?
```

What you should see: a list of your vibration monitors with their
addresses.

What might go wrong:

| Response | Meaning |
|---|---|
| `Not authorised` | the token doesn't match what you set in step 4 |
| `X2 rejected the login` | wrong X2 username or password |
| `X2_SITE_ID is not configured` | step 3 was skipped |
| `X2 refused the request (403)` | the account can't see that site |
| `count: 0` but `totalAddresses` > 0 | it reached your site but didn't recognise any sensor as a vibration monitor — see below |

That last one is the most likely thing to need a fix, because nothing here
has ever seen your real account. The Worker decides a sensor is a vibration
monitor if its type is 21 or 202, or if it reports any `extra_mlt_*`
readings. If your sensors are tagged differently, send me the output of
`/api/monitors` and it's a one-line change.

## Step 7 — Use it

Open the page, and in the **Live monitors** panel:

1. Paste your access token and press **Connect** — it lists your monitors.
2. Pick one and press **Fetch latest waveform**.
3. Optionally set **Auto-refresh**.

The Worker downloads the newest waveform the sensor uploaded; your browser
decodes and analyses it. Switching monitors fetches automatically.

---

## Things worth knowing

**"Live" is not a live stream.** These sensors upload a waveform on their
own schedule (their `config_mlt_auto_interval` setting), typically minutes
to hours apart. Auto-refresh checks for a *newer upload*; polling faster
than the sensors upload just costs API calls, which is why the shortest
interval offered is 5 minutes.

**Download links expire after an hour**, so the Worker always asks for a
fresh one immediately before downloading rather than remembering it.

**The token is the only thing protecting your data.** If you'd rather have
proper logins, put Cloudflare Access in front of the Worker (Zero Trust →
Access → Applications) and you get email or SSO sign-in instead.

**To rotate the token**, run `npx wrangler secret put ACCESS_TOKEN` again
with a new value and re-enter it in the page. The old one stops working
immediately.

**To turn live data off entirely**, delete the secret:
`npx wrangler secret delete ACCESS_TOKEN`. The page keeps working for
files you open by hand.

## Testing it without touching your real site

The whole chain can be exercised against a stand-in API — useful if you
want to see it work before pointing it at production:

```bash
python tests/worker/fake_x2.py &          # a pretend X2 on :9099
cp worker/.dev.vars.example .dev.vars     # then edit if you like
npx wrangler dev
```

Open http://127.0.0.1:8787 and connect with the token from `.dev.vars`.
`.dev.vars` is gitignored.
