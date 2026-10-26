"""Host-side client for the Isaac Sim ``isaacsim.code_editor.vscode`` socket.

The extension runs a TCP server (default 127.0.0.1:8226) that executes whatever
UTF-8 source it receives in Kit's *persistent* global scope, then replies with a
JSON document ``{"status","output","traceback","ename","evalue"}`` and closes the
connection.

Because the server treats each TCP ``data_received`` chunk as a complete program,
payloads must arrive in a single packet. Keep code small: to run a large module,
write it to disk and send a one-line bootstrap that ``exec``\\s the file (see
``--bootstrap``).

Usage:
    python isaac_client.py --file snippet.py
    echo "print('hi')" | python isaac_client.py
    python isaac_client.py --bootstrap D:/path/to/module.py "start_image_publisher()"

Stdlib only; runs under any Python.
"""
import argparse
import json
import socket
import sys


def send_code(code: str, host: str = "127.0.0.1", port: int = 8226, timeout: float = 60.0) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    try:
        s.sendall(code.encode("utf-8"))
        # Do NOT half-close the write side: sending EOF makes the server's
        # asyncio transport close before _process_code writes its JSON reply.
        # The server closes the connection itself once the reply is sent.
        chunks = []
        while True:
            try:
                b = s.recv(65536)
            except socket.timeout:
                break
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks).decode("utf-8", "replace")
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Execute Python in a live Isaac Sim via the VS Code code-editor socket.")
    ap.add_argument("--file", help="read code from this file (default: stdin)")
    ap.add_argument(
        "--bootstrap",
        metavar="MODULE",
        help="exec MODULE inside Isaac Sim, then run any trailing positional code (avoids the single-chunk size limit)",
    )
    ap.add_argument("rest", nargs="*", help="extra code to append (used with --bootstrap)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8226)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    if a.bootstrap:
        mod = a.bootstrap.replace("\\", "/")
        trailing = "\n".join(a.rest)
        code = (
            "exec(compile(open(r'{p}','r',encoding='utf-8').read(), r'{p}', 'exec'), globals())\n{t}".format(
                p=mod, t=trailing
            )
        )
    elif a.file:
        with open(a.file, "r", encoding="utf-8") as fh:
            code = fh.read()
    elif a.rest:
        # inline code passed as positional args
        code = "\n".join(a.rest)
    else:
        code = sys.stdin.read()

    raw = send_code(code, a.host, a.port, a.timeout)
    try:
        rep = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return 0

    out = rep.get("output", "")
    if out:
        print(out)
    if rep.get("status") != "ok":
        sys.stderr.write("\n--- ISAAC EXCEPTION ---\n")
        sys.stderr.write("{}: {}\n".format(rep.get("ename"), rep.get("evalue")))
        for t in rep.get("traceback", []):
            sys.stderr.write(t + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
