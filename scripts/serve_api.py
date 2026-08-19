#!/usr/bin/env python
"""Serve the MARSAD 813 alert feed over HTTP, using the standard library only.

Usage (from the repo root, after running the pipeline once):

    ".venv/Scripts/python" scripts/serve_api.py [--port 8813] [--results PATH]

Why stdlib ``http.server`` and no web framework: the delivery promise in the
PRD is a "dashboard + alert API", and a hackathon deliverable that a judge can
run with zero installs is worth more than one that needs a package index. The
API is read-only, single-purpose, and serves a file that the pipeline already
wrote, so a framework would add dependencies without adding capability. This
is a demonstration endpoint, not a hardened public service: it binds to
localhost by default and performs no authentication.

Scientific honesty (binding, see docs/CONTRACTS-V2.md)
------------------------------------------------------
Everything served here comes from ``synth.py``, our own physics-based forward
model of Gulf Case-2 water. The payloads are a synthetic physics-based
simulation, a self-consistency demonstration of the pipeline, and never an
observation of real water or independent validation. Each response repeats
that in its ``data_basis`` field.
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

# Allow running the script directly without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from marsad.alerts import (  # noqa: E402
    DATA_BASIS,
    SOURCE,
    alert_feed,
    alerts_from_results,
)

DEFAULT_PORT = 8813
DEFAULT_HOST = "127.0.0.1"
DEFAULT_RESULTS = _REPO_ROOT / "outputs" / "results.json"

# (method, path, description) - printed on startup and echoed in 404 bodies so
# an operator who mistypes a URL immediately sees the whole surface.
ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/health", "service status, results freshness, asset count"),
    ("GET", "/v1/alerts", "operator alert feed (AMBER and above; ?min_level=RED to narrow)"),
    ("GET", "/v1/intakes", "all monitored asset records"),
    ("GET", "/v1/intakes/<name>", "one asset by name, URL-encoded (e.g. /v1/intakes/Khor%20Fakkan)"),
    ("GET", "/v1/metrics", "Stage 1 and Stage 2 model metrics"),
)

_RUN_DEMO_HINT = (
    "Run the pipeline first, from the repo root:\n"
    "    .venv/Scripts/python scripts/run_demo.py\n"
    "then start the API again (or point --results at an existing results.json)."
)


class ResultsUnavailableError(RuntimeError):
    """Raised when the results file is missing or unreadable.

    Carries an operator-actionable message: the API cannot invent an alert
    feed, so the only useful thing it can do is say exactly which file it
    wanted and which command produces it.
    """


def load_results(path: str | Path) -> dict:
    """Load ``outputs/results.json`` written by the pipeline.

    Raises ``ResultsUnavailableError`` with a message that names the missing
    path and the command that creates it, rather than a bare traceback.
    """
    p = Path(path)
    if not p.exists():
        raise ResultsUnavailableError(
            f"results file not found: {p}\n{_RUN_DEMO_HINT}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultsUnavailableError(
            f"results file is not valid JSON: {p}\n"
            f"    {exc}\n{_RUN_DEMO_HINT}") from exc
    except OSError as exc:
        raise ResultsUnavailableError(
            f"could not read results file: {p}\n    {exc}\n{_RUN_DEMO_HINT}") from exc
    if not isinstance(data, dict) or "intakes" not in data:
        raise ResultsUnavailableError(
            f"results file has no 'intakes' key, so it was not written by this "
            f"pipeline: {p}\n{_RUN_DEMO_HINT}")
    return data


def _normalise(name: str) -> str:
    """Loose asset-name key: case, spaces, hyphens and underscores collapse.

    Operators type intake names by hand into a URL bar, so "Khor%20Fakkan",
    "khor-fakkan" and "khor_fakkan" all resolve to the same asset.
    """
    cleaned = unquote(str(name)).replace("-", " ").replace("_", " ")
    return " ".join(cleaned.split()).casefold()


def _envelope(results: dict, **payload: Any) -> dict:
    """Wrap a response body with the provenance fields every payload carries."""
    body = {
        "generated_utc": str(results.get("generated_utc", "")),
        "source": SOURCE,
        "data_basis": DATA_BASIS,
    }
    body.update(payload)
    return body


def route(results: dict, path: str, query: str = "") -> tuple[int, dict]:
    """Resolve one GET request to an (HTTP status, JSON body) pair.

    Pure function of the parsed request and the loaded results, so the whole
    routing surface is testable without opening a socket.
    """
    clean = path.rstrip("/") or "/"
    params = parse_qs(query)

    if clean in ("/", "/health"):
        return 200, {
            "status": "ok",
            "service": f"{SOURCE} alert API",
            "source": SOURCE,
            "data_basis": DATA_BASIS,
            "results_generated_utc": str(results.get("generated_utc", "")),
            "intakes": len(results.get("intakes", [])),
            "routes": [f"{method} {p}" for method, p, _ in ROUTES],
        }

    if clean == "/v1/alerts":
        feed = alert_feed(results)
        requested = params.get("min_level", [None])[0]
        if requested is not None:
            try:
                feed["alerts"] = [a.to_dict()
                                  for a in alerts_from_results(results, requested)]
            except ValueError as exc:
                return 400, {"error": "bad_request", "message": str(exc),
                             "data_basis": DATA_BASIS}
        return 200, feed

    if clean == "/v1/intakes":
        intakes = list(results.get("intakes", []))
        return 200, _envelope(results, count=len(intakes), intakes=intakes)

    if clean.startswith("/v1/intakes/"):
        wanted = _normalise(clean[len("/v1/intakes/"):])
        for intake in results.get("intakes", []):
            if _normalise(intake.get("name", "")) == wanted:
                return 200, _envelope(results, intake=intake)
        return 404, {
            "error": "not_found",
            "message": f"no monitored asset matches {wanted!r}",
            "path": path,
            "available": [str(i.get("name", "")) for i in results.get("intakes", [])],
            "data_basis": DATA_BASIS,
        }

    if clean == "/v1/metrics":
        return 200, _envelope(results, model_metrics=results.get("model_metrics", {}))

    return 404, {
        "error": "not_found",
        "message": f"unknown route {path!r}",
        "path": path,
        "routes": [f"{method} {p}" for method, p, _ in ROUTES],
        "data_basis": DATA_BASIS,
    }


def make_handler(results: dict, quiet: bool = False) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one already-loaded results dict."""

    class AlertAPIHandler(BaseHTTPRequestHandler):
        server_version = "MARSAD813/0.2"
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: dict, with_body: bool = True) -> None:
            payload = json.dumps(body, default=float).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # Open CORS: the dashboard may be opened straight from file://,
            # whose origin is "null", and this feed is public demo data.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if with_body:
                self.wfile.write(payload)

        def _resolve(self) -> tuple[int, dict]:
            parts = urlsplit(self.path)
            return route(results, parts.path, parts.query)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            status, body = self._resolve()
            self._send(status, body)

        def do_HEAD(self) -> None:  # noqa: N802
            status, body = self._resolve()
            self._send(status, body, with_body=False)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(200, {"ok": True, "allow": ["GET", "OPTIONS"]})

        def _method_not_allowed(self) -> None:
            self._send(405, {"error": "method_not_allowed",
                             "message": "this alert API is read-only; use GET",
                             "data_basis": DATA_BASIS})

        do_POST = do_PUT = do_DELETE = do_PATCH = _method_not_allowed  # type: ignore[assignment]

        def log_message(self, fmt: str, *args: Any) -> None:
            if not quiet:
                sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    return AlertAPIHandler


def make_server(results: dict, port: int = DEFAULT_PORT, host: str = DEFAULT_HOST,
                quiet: bool = False) -> ThreadingHTTPServer:
    """Create (but do not start) the alert API server.

    Pass ``port=0`` to bind an ephemeral port; the chosen port is then
    ``server.server_address[1]``. Callers own the lifecycle:
    ``serve_forever`` in a thread, then ``shutdown()`` and ``server_close()``.
    """
    server = ThreadingHTTPServer((host, int(port)), make_handler(results, quiet=quiet))
    server.daemon_threads = True
    return server


def print_route_table(host: str, port: int, results_path: Path, results: dict) -> None:
    """Print the served route table and the honest provenance of the data."""
    base = f"http://{host}:{port}"
    print(f"{SOURCE} alert API")
    print(f"  results   : {results_path}")
    print(f"  generated : {results.get('generated_utc', 'unknown')} "
          f"({len(results.get('intakes', []))} monitored assets)")
    print(f"  data basis: {DATA_BASIS} - our own forward model of Gulf Case-2")
    print("              water, a self-consistency demo and NOT real observations.")
    print(f"  listening : {base}")
    print("\n  Routes")
    width = max(len(p) for _, p, _ in ROUTES)
    for method, path, description in ROUTES:
        print(f"    {method:<4} {path:<{width}}  {description}")
    print(f"\n  Try: {base}/v1/alerts")
    print("  Ctrl+C to stop.\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS),
                        help="path to the pipeline's results.json "
                             "(default: outputs/results.json)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"interface to bind (default: {DEFAULT_HOST}; "
                             "use 0.0.0.0 to expose on the LAN)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not log individual requests")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    try:
        results = load_results(results_path)
    except ResultsUnavailableError as exc:
        print(f"{SOURCE} alert API: cannot start.\n{exc}", file=sys.stderr)
        return 2

    try:
        server = make_server(results, port=args.port, host=args.host, quiet=args.quiet)
    except OSError as exc:
        print(f"{SOURCE} alert API: cannot bind {args.host}:{args.port} ({exc}).\n"
              f"    Another process may already hold the port; retry with "
              f"--port {args.port + 1}.", file=sys.stderr)
        return 2

    print_route_table(args.host, server.server_address[1], results_path.resolve(), results)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping alert API.")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
