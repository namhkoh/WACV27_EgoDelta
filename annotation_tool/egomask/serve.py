from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

from .storage import (
    AccessDenied,
    AnnotationStore,
    InvalidAnnotation,
    RevisionConflict,
)
from .review import FinalExportReview


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        handler,
        store: AnnotationStore,
        final_review: FinalExportReview | None = None,
    ):
        super().__init__(address, handler)
        self.store = store
        self.final_review = final_review


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/bootstrap":
                annotator = self._authenticate(query)
                payload = self.server.store.bootstrap(annotator)
                payload["final_review"] = (
                    self.server.final_review.summary(annotator)
                    if self.server.final_review
                    else None
                )
                self._send_json(payload)
            elif parsed.path == "/api/task":
                annotator = self._authenticate(query)
                task_id = _one(query, "id")
                self._send_json(self.server.store.task_payload(annotator, task_id))
            elif parsed.path == "/api/image":
                annotator = self._authenticate(query)
                self._send_file(self.server.store.image_path(annotator, _one(query, "id")), cache=False)
            elif parsed.path == "/api/mask":
                annotator = self._authenticate(query)
                task_id = _one(query, "id")
                path = (
                    self.server.store.initial_mask_path(annotator, task_id)
                    if query.get("initial") == ["1"]
                    else self.server.store.mask_path(annotator, task_id)
                )
                self._send_file(path, cache=False)
            elif parsed.path == "/api/reference-mask":
                annotator = self._authenticate(query)
                self._send_file(
                    self.server.store.reference_mask_path(
                        annotator, _one(query, "id")
                    ),
                    cache=False,
                )
            elif parsed.path == "/api/final/bootstrap":
                annotator = self._authenticate(query)
                self._send_json(self._final_review().bootstrap(annotator))
            elif parsed.path == "/api/final/image":
                self._authenticate(query)
                self._send_file(
                    self._final_review().image_path(_one(query, "id")), cache=False
                )
            elif parsed.path == "/api/final/mask":
                self._authenticate(query)
                self._send_file(
                    self._final_review().mask_path(_one(query, "id")), cache=False
                )
            elif parsed.path in {"/", "/index.html"}:
                self._send_file(STATIC_ROOT / "index.html", cache=False)
            elif parsed.path == "/app.js":
                self._send_file(STATIC_ROOT / "app.js", cache=True)
            elif parsed.path == "/styles.css":
                self._send_file(STATIC_ROOT / "styles.css", cache=True)
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        except AccessDenied as exc:
            self._send_error(HTTPStatus.FORBIDDEN, str(exc))
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, f"Missing file: {exc}")
        except (KeyError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Server error: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path not in {"/api/save", "/api/final/flag"}:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            annotator = self._authenticate(query)
            content_length = int(self.headers.get("Content-Length", "0"))
            maximum_length = 10_000 if parsed.path == "/api/final/flag" else 12_500_000
            if content_length <= 0 or content_length > maximum_length:
                raise InvalidAnnotation("Invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            if parsed.path == "/api/final/flag":
                task_id = str(payload.get("task_id", ""))
                flagged = payload.get("flagged")
                if not isinstance(flagged, bool):
                    raise InvalidAnnotation("flagged must be a boolean")
                state = self._final_review().set_flag(task_id, annotator, flagged)
                self._send_json({"ok": True, "state": state})
            else:
                record = self.server.store.save(annotator, payload)
                self._send_json({"ok": True, "record": record})
        except AccessDenied as exc:
            self._send_error(HTTPStatus.FORBIDDEN, str(exc))
        except RevisionConflict as exc:
            self._send_json(
                {"error": str(exc), "current_revision": exc.current_revision},
                status=HTTPStatus.CONFLICT,
            )
        except (InvalidAnnotation, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Server error: {exc}")

    def _authenticate(self, query: dict[str, list[str]]) -> str:
        return self.server.store.authenticate(_one(query, "token"))

    def _final_review(self) -> FinalExportReview:
        if self.server.final_review is None:
            raise FileNotFoundError("Final-annotation review is not configured")
        return self.server.final_review

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, *, cache: bool) -> None:
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'",
        )

    def log_message(self, format: str, *args) -> None:
        # Avoid writing private annotator tokens from query strings to server logs.
        sanitized = list(args)
        if sanitized and isinstance(sanitized[0], str):
            sanitized[0] = sanitized[0].split("?", 1)[0]
        super().log_message(format, *sanitized)


def _one(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"Missing query parameter: {key}")
    return values[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve an EgoDelta mask-annotation run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--final-export",
        type=Path,
        help="Optional frozen EgoMask export exposed through the read-only review UI.",
    )
    parser.add_argument(
        "--review-flags",
        type=Path,
        help=(
            "Optional persistent redo-flag JSON. Defaults to final_review_flags.json "
            "inside --run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        store = AnnotationStore(args.run)
        final_review = (
            FinalExportReview(
                args.final_export,
                flags_path=args.review_flags
                or store.run_root / "final_review_flags.json",
            )
            if args.final_export
            else None
        )
        server = AnnotationHTTPServer(
            (args.host, args.port), AnnotationHandler, store, final_review
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"EgoMask server: http://{args.host}:{args.port}/")
    print(f"Private access paths: {(store.run_root / 'ACCESS_LINKS.txt')}")
    if final_review:
        print(
            f"Read-only final review: {final_review.export_root} "
            f"({len(final_review.tasks)} masks)"
        )
    print("Press Ctrl-C to stop. Completed annotations are saved immediately.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
