"""Prompt and decode speed of one engine configuration, measured through the engine's own protocol.

Starts the engine a setup config describes (strata-*.json), optionally on other GPUs, with a layer split, extra engine
arguments or environment switches, then sends prompts of the given lengths (built from this repository's source, so
any PC can rebuild the same ones) and reads the engine's own DONE numbers: prompt tokens read and the time it took,
tokens generated and the time it took.  Every request starts with its own line (--tag, the length, the repeat), so no
conversation checkpoint is reused and every prompt is read in full; runs with the same --tag read the same prompts
whatever their --label, so their tokens can be compared.

    python tools/multi_gpu_bench.py strata-iq3_s.json --gpus 0,1,2,3 --lengths 4096,32768 --max-new 256
    python tools/multi_gpu_bench.py strata-iq3_s.json --gpus all --env STRATA_SPLIT_PIPELINE=0 --label old-pipeline
    python tools/multi_gpu_bench.py strata-iq3_s.json --gpus all --tokens-out a.json      # the generated ids too
    python tools/multi_gpu_bench.py strata-iq3_s.json --gpus all --set "--ple-io ram" --set --mmap-experts
    python tools/multi_gpu_bench.py --compare a.json b.json                             # same tokens? (greedy A/B)
    python tools/multi_gpu_bench.py strata-iq3_s.json --lengths 250000 --repeats 1 --env STRATA_CUDA_PROFILE=240000:40 \
        --wrap "nsys profile -o decode250k -f true -t cuda --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop"

One JSON line per request goes to --out (appended), a table to the terminal.  Greedy decoding (temperature 0) unless
--temperature is given.
"""
from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

CORPUS_GLOBS = ("src/**/*.cpp", "src/**/*.cu", "include/**/*.hpp", "serve/*.py", "tools/*.py", "docs/*.md")
CHAT_PREFIX = "<|im_start|>user\nRun {tag}. Read this source code, then summarize what it does in five bullet points.\n\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load_tokenizer(path: Path):
    import strata_tokenizer as ST
    vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (path / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((path / "token_type.json").read_text()))


def corpus_ids(tok, need: int) -> list[int]:
    """At least `need` token ids of this repository's source, in a fixed file order (cut later)."""
    ids: list[int] = []
    files = sorted({p for g in CORPUS_GLOBS for p in ROOT.glob(g) if p.is_file()})
    while len(ids) < need:
        for p in files:
            ids += tok.encode(f"\n\n// ---- {p.relative_to(ROOT).as_posix()}\n" + p.read_text(encoding="utf-8", errors="replace"))
            if len(ids) >= need:
                break
        if not files:
            raise SystemExit("no source files found for the prompts")
    return ids


def prompt(tok, body: list[int], length: int, tag: str) -> list[int]:
    head = tok.encode(CHAT_PREFIX.format(tag=tag), parse_special=True)
    tail = tok.encode(CHAT_SUFFIX, parse_special=True)
    n = max(0, length - len(head) - len(tail))
    return head + body[:n] + tail


def visible_gpus() -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + (2 if i + 1 < len(out) and not out[i + 1].startswith("--") else 1)]
    if value is not None:
        out += [flag] if value == "" else [flag, value]
    return out


def compare(a: Path, b: Path) -> int:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    keys = sorted(set(ra) & set(rb))
    same = 0
    for k in keys:
        x, y = ra[k], rb[k]
        n = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if n is None and len(x) == len(y):
            same += 1
            print(f"  {k}: identical ({len(x)} tokens)")
        else:
            print(f"  {k}: differ from token {n if n is not None else min(len(x), len(y))} ({len(x)} / {len(y)} tokens)")
    print(f"{same} of {len(keys)} requests identical")
    return 0 if keys and same == len(keys) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="a setup config, strata-<model>.json")
    ap.add_argument("--gpus", help='the cards, as nvidia-smi numbers them: "0,1,2" or "all" (default: the config\'s)')
    ap.add_argument("--layer-split", help='"auto" (default with several cards) or "K1,K2,.."')
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine (repeatable)")
    ap.add_argument("--set", action="append", default=[], help='an engine flag: "--spec 6" or "--prefill 4096" (repeatable)')
    ap.add_argument("--lengths", default="4096,32768", help="prompt lengths in tokens (comma separated)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="", help="names the run in the output (not part of the prompts)")
    ap.add_argument("--tag", default="bench", help="the word every prompt starts with: two runs with the same tag read "
                                                    "the same prompts, so --compare can match their tokens")
    ap.add_argument("--out", default="multi_gpu_bench.jsonl", help="JSON lines appended here")
    ap.add_argument("--tokens-out", help="the generated ids of every request, as JSON (for --compare)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two --tokens-out files and exit")
    ap.add_argument("--wrap", help='run the engine under this command (a profiler: "nsys profile -o x ..."); the '
                                   'engine and its arguments follow it')
    # `--set --mmap-experts` (a value that starts with "--") is taken as --set=--mmap-experts
    argv, i = [], 0
    raw = sys.argv[1:]
    while i < len(raw):
        if raw[i] == "--set" and i + 1 < len(raw):
            argv.append("--set=" + raw[i + 1])
            i += 2
        else:
            argv.append(raw[i])
            i += 1
    a = ap.parse_args(argv)
    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]))
    if not a.config:
        ap.error("a config is needed (or --compare)")

    from serve.server import StrataEngine, child_env, engine_args
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    if a.gpus:
        cfg["gpu"] = visible_gpus() if a.gpus.strip().lower() == "all" else [int(x) for x in a.gpus.split(",")]
    if a.layer_split:
        cfg["layer_split"] = a.layer_split
    cfg["env"] = dict(cfg.get("env") or {})
    for kv in a.env:
        k, _, v = kv.partition("=")
        cfg["env"][k] = v
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    args = engine_args(cfg)
    for s in a.set:
        parts = s.split(maxsplit=1)
        args = with_arg(args, parts[0], parts[1] if len(parts) > 1 else "")
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 0
    if ctx and max(lengths) + a.max_new + 64 > ctx:
        args = with_arg(args, "--max-context", str(max(lengths) + a.max_new + 1024))
        print(f"--max-context raised to {max(lengths) + a.max_new + 1024} for the longest prompt")
    tok = load_tokenizer(Path(cfg["tokenizer"]))
    body = corpus_ids(tok, max(lengths))
    gpus = cfg.get("gpu")
    label = a.label or f"gpus={gpus}"
    print(f"[{label}] engine args: {' '.join(args)}")
    if cfg["env"]:
        print(f"[{label}] env: {cfg['env']}")
    exe = cfg["exe"]
    if a.wrap:   # a small script, so the engine's own command line (`exe --serve args`) stays as the server builds it
        wrapper = Path(a.out).resolve().with_suffix(".wrap.sh")
        wrapper.write_text(f"#!/bin/sh\nexec {a.wrap} {shlex.quote(str(Path(exe).resolve()))} \"$@\"\n")
        wrapper.chmod(0o755)
        exe = str(wrapper)
        print(f"[{label}] engine wrapped: {a.wrap}")
    t0 = time.time()
    eng = StrataEngine(exe, args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    if not eng.alive():
        print(f"[{label}] the engine did not start; its log: {cfg.get('log')}")
        return 1
    load_s = time.time() - t0
    print(f"[{label}] engine ready in {load_s:.0f} s, context {eng.max_context}")
    rows, tokens_by_req = [], {}

    def one(ids: list[int], key: str) -> dict:
        sampling = {"temperature": a.temperature} if a.temperature > 0 else {"temperature": 0}
        out = [t for t in eng.generate(ids, a.max_new, sampling, threading.Event()) if t is not None]
        d = dict(eng.last or {})
        read = d.get("prompt_read", d.get("prompt_tokens", len(ids)) - d.get("reused", 0))
        r = {"label": label, "gpus": gpus, "env": cfg["env"], "set": a.set, "request": key,
             "prompt_tokens": d.get("prompt_tokens"), "read": read, "prompt_ms": d.get("prompt_ms"),
             "prompt_tps": round(1000.0 * read / d["prompt_ms"], 1) if d.get("prompt_ms") else None,
             "generated": d.get("generated", len(out)), "decode_ms": d.get("decode_ms"),
             "decode_tps": round(1000.0 * d.get("generated", len(out)) / d["decode_ms"], 2) if d.get("decode_ms") else None,
             "drafts_accepted": d.get("drafts_accepted"), "drafts_offered": d.get("drafts_offered"),
             "hits": d.get("hits"), "lookups": d.get("lookups"), "finish": d.get("finish"), "load_s": round(load_s)}
        tokens_by_req[key] = out
        return r

    try:
        one(prompt(tok, body, 512, f"{a.tag} warm-up"), "warm-up")
        for L in lengths:
            for rep in range(a.repeats):
                r = one(prompt(tok, body, L, f"{a.tag} {L} #{rep}"), f"{L}#{rep}")
                rows.append(r)
                print(f"[{label}] {L:>7} tokens #{rep}: prompt {r['read']} tok in {r['prompt_ms']:.0f} ms = "
                      f"{r['prompt_tps']} tok/s | decode {r['generated']} tok = {r['decode_tps']} tok/s "
                      f"(drafts {r['drafts_accepted']}/{r['drafts_offered']}, finish {r['finish']})")
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r) + "\n")
    finally:
        try:
            eng.proc.stdin.write("QUIT\n")
            eng.proc.stdin.flush()
            eng.proc.wait(120)
        except Exception:
            eng.proc.kill()
    if a.tokens_out:
        Path(a.tokens_out).write_text(json.dumps(tokens_by_req))
    print(f"\n[{label}] medians:")
    for L in lengths:
        rs = [r for r in rows if r["request"].startswith(f"{L}#")]
        pt = [r["prompt_tps"] for r in rs if r["prompt_tps"]]
        dt = [r["decode_tps"] for r in rs if r["decode_tps"]]
        print(f"  {L:>7} tokens: prompt {statistics.median(pt) if pt else 0:,.0f} tok/s, "
              f"decode {statistics.median(dt) if dt else 0:.1f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
