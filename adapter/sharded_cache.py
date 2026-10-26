"""Weight cache for the TP target: each rank's loaded weights, copied straight back on later boots.

DSV41_SHARDED_CACHE=<directory inside the container> turns it on (the TP4 profile uses
/root/.cache/dsv41-sharded, which is ~/.cache/dsv41-sharded on each Spark, chattr +C).

A normal boot reads and slices the checkpoint (DSV41_FAST_LOAD's eager reads: ~155 GB a rank
for ~83 GB kept), runs every load hook, then the quant post-process (b12x repack, MXFP8
swizzles, padding). This adapter saves the rank's state between those two steps: after
model.load_weights() and its wrappers (the APE reorder, post_load_weights, the wo_a fp8 twins),
before process_weights_after_loading. A later boot whose fingerprint has a complete copy on
every rank skips the checkpoint: it builds the model, copies the saved bytes in, marks the
compressors' APE as already converted, calls model.load_weights() with no weights (so the
model's own post-load tail and every adapter wrapper run exactly as on a normal boot), then
runs the same quant post-process on the same bytes.

The fingerprint covers the adapters, SGLang, b12x, the checkpoint's config and index, the
launch_server arguments that are not plainly runtime-only, and every DSV41_/SGLANG_/B12X_
variable not listed below as runtime-only: a change re-saves instead of loading a stale copy.
The first boot after a change pays the normal load plus the write (~83 GB a rank).

Every save records a position-sensitive digest of each tensor, before and after the quant
post-process. DSV41_SHARDED_VERIFY=1 recomputes both on a cached boot and refuses to serve when
a weight or wo_a twin differs (other plain tensors only warn: some are uninitialized
workspaces).
"""
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

_TAG = "[sharded_cache]"
_TARGET = "DeepseekV4ForCausalLM"
_FORMAT = 1
_ALIGN = 4096
_PART_BYTES = 4 << 30
_CHUNK = 256 << 20
_READERS = int(os.environ.get("DSV41_SHARDED_READERS", "8"))
_KEEP = int(os.environ.get("DSV41_SHARDED_KEEP", "1"))
_VERIFY = os.environ.get("DSV41_SHARDED_VERIFY", "0").strip() not in ("0", "", "off", "false")

# variables that only address, pace or log the runtime; every other DSV41_/SGLANG_/B12X_ counts
_ENV_PREFIXES = ("DSV41_", "SGLANG_", "B12X_")
_ENV_RUNTIME = ("DSV41_SHARDED_", "DSV41_STATS_", "DSV41_LOOP_", "DSV41_ROCE_", "SGLANG_ROCE_",
                "B12X_ROCE_", "B12X_COMPILE_CACHE_DIR", "DSV41_IO_THREADS", "DSV41_FAST_LOAD_INFLIGHT_GB",
                "DSV41_FAST_LOAD_DEBUG", "DSV41_FAST_LOAD_MEMLOG", "SGLANG_BUILD_", "SGLANG_IMAGE_TAG",
                "SGLANG_RUN_ID")
# launch_server flags that only size or address the runtime (dropped with their values)
_ARG_RUNTIME = {"--host", "--port", "--served-model-name", "--mem-fraction-static", "--context-length",
                "--max-running-requests", "--chunked-prefill-size", "--cuda-graph-max-bs",
                "--cuda-graph-max-bs-decode", "--max-total-tokens", "--dist-init-addr", "--node-rank",
                "--log-level", "--log-level-http", "--log-requests", "--log-requests-level",
                "--enable-metrics", "--watchdog-timeout"}
_SECRET = ("KEY", "TOKEN", "SECRET", "PASS")

_pending = {}      # id(model) -> (directory, rank, fingerprint, inputs) for a save between load and post-process
_cached = {}       # id(model) -> (directory, rank) of a model restored from the cache (verify after post-process)


def _log(msg):
    print(f"{_TAG} {msg}", flush=True)


# ---------------------------------------------------------------- fingerprint
def _file_digest(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
    except OSError:
        h.update(b"missing")
    return h.digest()


def _tree(h, root):
    if not os.path.isdir(root):
        h.update(f"missing:{root}".encode())
        return
    for top, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith((".py", ".so", ".cpp", ".cu", ".cuh", ".h")):
                path = os.path.join(top, name)
                h.update(os.path.relpath(path, root).encode())
                h.update(_file_digest(path))


def _launch_args():
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            argv = [a.decode() for a in f.read().split(b"\0") if a]
    except OSError:
        argv = list(sys.argv)
    if "sglang.launch_server" not in argv:
        argv = list(sys.argv)
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            if not a.startswith("--"):
                continue
        flag = a.split("=", 1)[0]
        if flag in _ARG_RUNTIME or any(s in flag.upper() for s in _SECRET):
            skip = "=" not in a
            continue
        out.append(a)
    return out


def _fingerprint(model_config, world):
    import sglang
    env = {k: v for k, v in os.environ.items()
           if k.startswith(_ENV_PREFIXES) and not k.startswith(_ENV_RUNTIME)
           and not any(s in k for s in _SECRET)}
    inputs = {"format": _FORMAT, "class": _TARGET, "world": world, "torch": torch.__version__,
              "dtype": str(model_config.dtype), "env": dict(sorted(env.items())), "args": _launch_args()}
    h = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode())
    for name in ("config.json", "model.safetensors.index.json"):
        h.update(name.encode())
        h.update(_file_digest(os.path.join(model_config.model_path, name)))
    for root in (os.path.dirname(sglang.__file__), os.path.dirname(os.path.abspath(__file__)),
                 "/opt/b12x_next", "/opt/b12x"):
        _tree(h, root)
    return h.hexdigest()[:24], inputs


# ---------------------------------------------------------------- what is saved
def _wo_a_w8():
    mod = sys.modules.get("wo_a_w8")
    return mod if mod is not None and getattr(mod, "_TWINS", None) is not None else None


def _weights(model, loader_module):
    """name -> tensor of every weight that owns its storage (the ShardedStateLoader filter), with each
    wo_a whose bf16 copy wo_a_w8 released replaced by its bit-identical dequantized twin."""
    sd = loader_module.ShardedStateLoader._filter_subtensors(model.state_dict())
    w8 = _wo_a_w8()
    out = {}
    for name, t in sd.items():
        if w8 is not None and t.data_ptr() in w8._DROPPED:
            t = w8.dequant_into_scratch(w8._TWINS[t.data_ptr()], t.device).view(t.shape)
        elif t.numel() > 1 and 0 in t.stride():
            raise RuntimeError(f"{name} is an expanded view (stride {t.stride()}): nothing here knows how to save it")
        out[name] = t
    return out


def _extras(model, weights):
    """Plain tensors hung on modules (not parameters or buffers, not views of a weight)."""
    storages = {(t.device, t.untyped_storage().data_ptr()) for t in weights.values() if t.numel()}
    out = {}
    for mname, module in model.named_modules():
        for attr, val in list(vars(module).items()):
            if not isinstance(val, torch.Tensor) or isinstance(val, torch.nn.Parameter) or not val.numel():
                continue
            if (val.device, val.untyped_storage().data_ptr()) in storages:
                continue
            if val.numel() > 1 and 0 in val.stride():
                continue
            out[f"{mname}.{attr}" if mname else attr] = val
    return out


def _twins(model):
    """wo_a fp8 twins by weight name (wo_a_w8 keeps them by data_ptr, outside the module)."""
    w8 = _wo_a_w8()
    if w8 is None:
        return {}
    out = {}
    for mname, module in model.named_modules():
        w = getattr(getattr(module, "wo_a", None), "weight", None)
        if w is None:
            continue
        twin = w8._TWINS.get(w.data_ptr())
        if twin is not None:
            out[f"{mname}.wo_a.twin.w8"], out[f"{mname}.wo_a.twin.scale"] = twin
    return out


def _flat_bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8)


def _digest(t):
    """Position-sensitive: the plain sum and an index-weighted sum (mod 65521) of the tensor's bytes
    read as int32 words (or bytes)."""
    b = _flat_bytes(t)
    x = b.view(torch.int32) if b.numel() % 4 == 0 else b
    s1 = s2 = 0
    step = 1 << 24
    for i in range(0, x.numel(), step):
        c = x[i:i + step].to(torch.int64)
        w = torch.arange(i, i + c.numel(), dtype=torch.int64, device=c.device) % 65521 + 1
        s1 += int(c.sum())
        s2 += int(((c % 65521) * w).sum())
    return f"{b.numel()}:{s1}:{s2}"


def _digests(model, loader_module):
    weights = _weights(model, loader_module)
    return ({k: _digest(v) for k, v in weights.items()} | {k: _digest(v) for k, v in _twins(model).items()},
            {k: _digest(v) for k, v in _extras(model, weights).items()})


# ---------------------------------------------------------------- files
def _paths(directory, rank):
    return (os.path.join(directory, f"rank{rank}.json"), os.path.join(directory, f"rank{rank}.ready"))


def _complete(directory, rank):
    manifest, ready = _paths(directory, rank)
    if not os.path.isfile(ready):
        return False
    try:
        with open(manifest) as f:
            m = json.load(f)
        return all(os.path.getsize(os.path.join(directory, p["file"])) == p["size"] for p in m["parts"])
    except (OSError, ValueError, KeyError):
        return False


def _drop_cache(fd):
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _save(model, loader_module, directory, rank, fp, inputs):
    t0 = time.perf_counter()
    root = os.path.dirname(directory)
    os.makedirs(directory, exist_ok=True)
    manifest_path, ready_path = _paths(directory, rank)
    for p in (ready_path, manifest_path):
        if os.path.exists(p):
            os.remove(p)
    weights = _weights(model, loader_module)
    extras = _extras(model, weights)
    entries, parts = [], []
    f = None
    try:
        for kind, table in (("weight", weights), ("extra", extras)):
            for name, t in table.items():
                b = _flat_bytes(t)
                n = b.numel()
                if f is None or (f.tell() and f.tell() + n > _PART_BYTES):
                    if f is not None:
                        f.flush(); os.fsync(f.fileno()); _drop_cache(f.fileno())
                        parts[-1]["size"] = f.tell(); f.close()
                    fname = f"rank{rank}-part{len(parts):03d}.bin"
                    f = open(os.path.join(directory, fname), "wb")
                    parts.append({"file": fname, "size": 0})
                off = f.tell()
                for i in range(0, n, _CHUNK):
                    f.write(memoryview(b[i:i + _CHUNK].cpu().numpy()))
                pad = -f.tell() % _ALIGN
                if pad:
                    f.write(b"\0" * pad)
                entries.append({"kind": kind, "name": name, "dtype": str(t.dtype).removeprefix("torch."),
                                "shape": list(t.shape), "device": t.device.type, "part": len(parts) - 1,
                                "offset": off, "nbytes": n, "digest": _digest(b)})
        if f is not None:
            f.flush(); os.fsync(f.fileno()); _drop_cache(f.fileno())
            parts[-1]["size"] = f.tell(); f.close(); f = None
        twins = {k: _digest(v) for k, v in _twins(model).items()}
        with open(manifest_path, "w") as m:
            json.dump({"fingerprint": fp, "rank": rank, "inputs": inputs, "parts": parts, "entries": entries,
                       "twins": twins}, m)
        with open(ready_path, "w") as m:
            m.write(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    except BaseException:
        if f is not None:
            f.close()
        for p in parts:
            try:
                os.remove(os.path.join(directory, p["file"]))
            except OSError:
                pass
        raise
    keep = {p["file"] for p in parts}
    for name in os.listdir(directory):
        if name.startswith(f"rank{rank}-part") and name not in keep:
            os.remove(os.path.join(directory, name))
    total = sum(p["size"] for p in parts)
    _log(f"rank {rank}: saved {len(entries)} tensors ({len([e for e in entries if e['kind'] == 'extra'])} plain), "
         f"{total / 2**30:.1f} GiB in {len(parts)} parts to {directory} in {time.perf_counter() - t0:.0f}s")
    if _KEEP > 0:
        others = sorted((d for d in os.listdir(root) if d != fp and len(d) == 24
                         and os.path.isfile(os.path.join(root, d, f"rank{rank}.json"))),
                        key=lambda d: os.path.getmtime(os.path.join(root, d)), reverse=True)
        for d in others[_KEEP - 1:]:
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            _log(f"rank {rank}: removed the older copy {d}")


def _restore(model, loader_module, directory, rank):
    t0 = time.perf_counter()
    with open(_paths(directory, rank)[0]) as f:
        m = json.load(f)
    weights = _weights(model, loader_module)
    extras = _extras(model, weights)
    saved = {e["name"] for e in m["entries"] if e["kind"] == "weight"}
    if saved != set(weights):
        missing, unknown = sorted(set(weights) - saved), sorted(saved - set(weights))
        raise RuntimeError(f"{_TAG} the saved weights do not match this model: {len(missing)} missing "
                           f"{missing[:5]}, {len(unknown)} unknown {unknown[:5]}; delete {directory}")
    device = next((t.device for t in weights.values() if t.device.type == "cuda"), torch.device("cpu"))
    dests, placed = {}, 0
    for e in m["entries"]:
        dtype = getattr(torch, e["dtype"])
        shape = tuple(e["shape"])
        if e["kind"] == "weight":
            t = weights[e["name"]]
            if t.dtype != dtype or tuple(t.shape) != shape:
                raise RuntimeError(f"{_TAG} {e['name']}: saved {e['dtype']} {shape}, model {t.dtype} {tuple(t.shape)}")
        else:
            t = extras.get(e["name"])
            if t is None or t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
                t = torch.empty(shape, dtype=dtype, device=device if e["device"] == "cuda" else "cpu")
                mname, _, attr = e["name"].rpartition(".")
                setattr(model.get_submodule(mname) if mname else model, attr, t)
                placed += 1
        dests[e["name"]] = t
    by_part = {}
    for e in m["entries"]:
        by_part.setdefault(e["part"], []).append(e)

    def read(part):
        if device.type == "cuda":
            torch.cuda.set_device(device)
        buf = torch.empty(_CHUNK, dtype=torch.uint8)
        view = memoryview(buf.numpy())
        with open(os.path.join(directory, m["parts"][part]["file"]), "rb", buffering=0) as f:
            for e in by_part[part]:
                dst = dests[e["name"]]
                flat = dst.reshape(-1).view(torch.uint8) if dst.is_contiguous() else None
                tmp = None if flat is not None else torch.empty(e["nbytes"], dtype=torch.uint8, device=dst.device)
                f.seek(e["offset"])
                for i in range(0, e["nbytes"], _CHUNK):
                    n = min(_CHUNK, e["nbytes"] - i)
                    got = f.readinto(view[:n])
                    if got != n:
                        raise RuntimeError(f"{_TAG} short read in {m['parts'][part]['file']} at {e['name']}")
                    (flat if flat is not None else tmp)[i:i + n].copy_(buf[:n])
                if tmp is not None:
                    dst.copy_(tmp.view(dst.dtype).view(dst.shape))
            _drop_cache(f.fileno())
        if device.type == "cuda":
            torch.cuda.synchronize()

    with ThreadPoolExecutor(max_workers=max(1, min(_READERS, len(by_part)))) as pool:
        list(pool.map(read, sorted(by_part)))
    total = sum(p["size"] for p in m["parts"])
    took = time.perf_counter() - t0
    _log(f"rank {rank}: restored {len(m['entries'])} tensors ({placed} plain tensors placed anew), "
         f"{total / 2**30:.1f} GiB in {took:.0f}s ({total / 2**30 / max(took, 1e-9):.1f} GiB/s)")
    return m


def _check(rank, label, expected_weights, expected_twins, expected_extras, model, loader_module):
    got_weights, got_extras = _digests(model, loader_module)
    want = dict(expected_weights) | dict(expected_twins)
    bad = sorted(k for k in set(want) | set(got_weights) if want.get(k) != got_weights.get(k))
    soft = sorted(k for k in set(expected_extras) & set(got_extras) if expected_extras[k] != got_extras[k])
    if soft:
        _log(f"rank {rank}: {label}: {len(soft)} plain tensors differ (workspaces?): {soft[:6]}")
    if bad:
        raise RuntimeError(f"{_TAG} rank {rank}: {label}: {len(bad)} weights differ from the saved copy: {bad[:8]}")
    _log(f"rank {rank}: {label}: all {len(want)} weights and twins match the saved digests")


# ---------------------------------------------------------------- the loader hooks
def _agree(have):
    import torch.distributed as dist
    from sglang.srt.distributed import get_tp_group
    t = torch.tensor([1 if have else 0], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
    return bool(t.item())


def _mark_ape_converted(model):
    n = 0
    for module in model.modules():
        if hasattr(module, "ape_converted"):
            module.ape_converted = True
            n += 1
    return n


class _Quiet:
    """Silence 'Some weights are not initialized' while load_weights() runs with no weights."""

    def __enter__(self):
        import logging
        self._logger = logging.getLogger("sglang.srt.models.deepseek_v4")
        self._level = self._logger.level
        self._logger.setLevel(logging.ERROR)

    def __exit__(self, *exc):
        self._logger.setLevel(self._level)


def install(loader_module):
    root = os.environ.get("DSV41_SHARDED_CACHE", "").strip()
    if not root:
        return
    cls = loader_module.DefaultModelLoader
    if getattr(cls.load_model, "_dsv41_sharded", False):
        return
    orig_load_model = cls.load_model
    orig_postprocess = cls.postprocess_weights

    def load_model(self, *, model_config, device_config):
        model_class, _ = loader_module.get_model_architecture(model_config)
        if model_class.__name__ != _TARGET or getattr(model_config, "modelopt_quant", False):
            return orig_load_model(self, model_config=model_config, device_config=device_config)
        from sglang.srt.distributed import get_tp_group
        group = get_tp_group()
        rank, world = group.rank_in_group, group.world_size
        fp, inputs = _fingerprint(model_config, world)
        directory = os.path.join(root, fp)
        mine = _complete(directory, rank)
        if not _agree(mine):
            _log(f"rank {rank}: copy {fp} {'complete here' if mine else 'not here'}, not on every rank: "
                 f"loading the checkpoint, then saving")
            _pending[id(self)] = (directory, rank, fp, inputs)
            try:
                return orig_load_model(self, model_config=model_config, device_config=device_config)
            finally:
                _pending.pop(id(self), None)
        _log(f"rank {rank}: loading copy {fp} instead of the checkpoint")
        target_device = torch.device(device_config.device)
        quant_config = loader_module._get_quantization_config(model_config, self.load_config)
        with loader_module.set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = loader_module._initialize_model(model_config, self.load_config, quant_config)
            if getattr(model, "wo_a_fp8", False):
                raise RuntimeError(f"{_TAG} wo_a_fp8 is on: its post-load scale setup is not known to be "
                                   f"repeatable; unset DSV41_SHARDED_CACHE")
            m = _restore(model, loader_module, directory, rank)
            n = _mark_ape_converted(model)
            t0 = time.perf_counter()
            with _Quiet():
                model.load_weights(iter(()))
            _log(f"rank {rank}: post-load tail and wrappers replayed ({n} compressors marked converted) "
                 f"in {time.perf_counter() - t0:.1f}s")
            if _VERIFY:
                saved = {e["name"]: e["digest"] for e in m["entries"] if e["kind"] == "weight"}
                extras = {e["name"]: e["digest"] for e in m["entries"] if e["kind"] == "extra"}
                _check(rank, "before post-process", saved, m.get("twins", {}), extras, model, loader_module)
                _cached[id(model)] = (directory, rank)
            cls.postprocess_weights(model, target_device)
        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()

    def postprocess_weights(model, target_device):
        job = next(iter(_pending.values()), None)
        if job is not None and type(model).__name__ == _TARGET:
            directory, rank, fp, inputs = job
            try:
                _save(model, loader_module, directory, rank, fp, inputs)
            except Exception as exc:  # noqa: BLE001 -- a failed save must not stop the boot
                _log(f"rank {rank}: save failed, serving from the checkpoint load: {exc!r}")
                job = None
        orig_postprocess(model, target_device)
        if job is not None and type(model).__name__ == _TARGET:
            directory, rank, fp, inputs = job
            try:
                w, x = _digests(model, loader_module)
                with open(os.path.join(directory, f"rank{rank}.final.json"), "w") as f:
                    json.dump({"weights": w, "extras": x}, f)
            except Exception as exc:  # noqa: BLE001
                _log(f"rank {rank}: post-process digests not recorded: {exc!r}")
        elif id(model) in _cached:
            directory, rank = _cached.pop(id(model))
            path = os.path.join(directory, f"rank{rank}.final.json")
            if os.path.isfile(path):
                with open(path) as f:
                    final = json.load(f)
                _check(rank, "after post-process", final["weights"], {}, final["extras"], model, loader_module)

    load_model._dsv41_sharded = True
    cls.load_model = load_model
    cls.postprocess_weights = staticmethod(postprocess_weights)
    _log(f"armed: weight cache in {root} (verify {'on' if _VERIFY else 'off'}, {_READERS} readers)")
