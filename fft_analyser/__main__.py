"""
Run the FFT analyser UI: ``python -m fft_analyser``.

Credentials may be supplied through the environment instead of typed into
the browser - which is what you want for anything long-running:

===================  =====================================================
``X2_BASE_URL``      API host (default ``https://api.x2wireless.com``)
``X2_USERNAME``      API username
``X2_PASSWORD``      API password
``X2_SITE_ID``       site to open
``X2_MONITORS``      comma-separated addresses to load at startup (max 5)
``X2_POLL_SECONDS``  live-feed poll interval (default 300)
===================  =====================================================

With ``X2_USERNAME``/``X2_PASSWORD``/``X2_SITE_ID`` set, ``--connect``
logs in at startup and pre-fills the monitor list, so the page opens
already pointed at the site.  Reading those from the environment (or a
``.env`` file that is not committed) keeps the password out of the
browser, out of the URL, and out of shell history.

Use ``--serve`` for anything beyond a desktop session: it runs the app
under Waitress, a production WSGI server, instead of Flask's development
server.
"""

import argparse
import os
import sys

from .app import create_app
from .monitors import MAX_MONITORS


def _env_bootstrap(app, connect: bool) -> None:
    """Optionally log in and preload monitors from the environment."""
    from .app import BANK

    user = os.environ.get("X2_USERNAME")
    password = os.environ.get("X2_PASSWORD")
    site = os.environ.get("X2_SITE_ID")
    base = os.environ.get("X2_BASE_URL", "https://api.x2wireless.com")
    if not connect:
        return
    if not (user and password and site):
        print("--connect needs X2_USERNAME, X2_PASSWORD and X2_SITE_ID; "
              "starting without a connection.", file=sys.stderr)
        return

    try:
        found = BANK.connect(user, password, base, site)
    except Exception as exc:
        print(f"Startup connection failed: {exc}", file=sys.stderr)
        print("The UI still starts; connect from the Live monitors panel.",
              file=sys.stderr)
        return

    print(f"Connected read-only to site {site}: "
          f"{len(found)} vibration monitor(s).")
    for a in found:
        print(f"  {a.get('Name') or '(unnamed)'}  {a.get('Address')}")

    wanted = [a.strip() for a in
              os.environ.get("X2_MONITORS", "").split(",") if a.strip()]
    if not wanted:
        return
    if len(wanted) > MAX_MONITORS:
        print(f"X2_MONITORS lists {len(wanted)} addresses; loading the first "
              f"{MAX_MONITORS}.", file=sys.stderr)
    try:
        loaded = BANK.load_from_cloud(wanted[:MAX_MONITORS])
    except Exception as exc:
        print(f"Preloading waveforms failed: {exc}", file=sys.stderr)
        return
    for m in loaded:
        state = (f"loaded {len(m.waveform.data)} samples @ {m.waveform.fs:g} Hz"
                 if m.ok else f"FAILED - {m.error}")
        print(f"  {m.name}: {state}")


def main():
    parser = argparse.ArgumentParser(
        description="FFT analyser UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; use 0.0.0.0 to accept "
                             "connections from other machines")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true",
                        help="Flask debug mode (never use with --serve)")
    parser.add_argument("--serve", action="store_true",
                        help="run under Waitress, a production WSGI server, "
                             "instead of the Flask development server")
    parser.add_argument("--threads", type=int, default=8,
                        help="worker threads when using --serve")
    parser.add_argument("--connect", action="store_true",
                        help="log in at startup using the X2_* environment "
                             "variables and preload X2_MONITORS")
    args = parser.parse_args()

    app = create_app()
    _env_bootstrap(app, args.connect)

    if args.serve:
        try:
            from waitress import serve
        except ImportError:
            sys.exit("--serve needs Waitress: pip install waitress")
        print(f"Serving on http://{args.host}:{args.port}")
        serve(app.server, host=args.host, port=args.port, threads=args.threads)
    else:
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
