"""CPU check of adapter/sharded_cache.py: save/restore round trip, digests, fingerprint.

Run inside the image, off the GPU:
    CUDA_VISIBLE_DEVICES= python3 tests/test_sharded_cache_cpu.py
"""
import json, os, sys, tempfile, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.environ.get("DSV41_ADAPTER_DIR", "/opt/dsv41/adapter"))
import torch
import sharded_cache as sc
import sglang.srt.model_loader.loader as L

sc._PART_BYTES = 6000          # force several parts
sc._CHUNK = 1000               # and chunked reads/writes

def make(seed):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16), torch.nn.Embedding(10, 16))
    m.head = torch.nn.Linear(16, 10, bias=False)
    m.head.weight = m[2].weight               # tied: one storage, saved once
    m[0].cache = torch.randn(7, 3)             # a plain tensor on a module
    m[1].scale8 = torch.randn(4).to(torch.float8_e4m3fn)
    return m

d = tempfile.mkdtemp()
fp = "a" * 24
src = make(0)
sc._save(src, L, os.path.join(d, fp), 0, fp, {"test": True})
assert sc._complete(os.path.join(d, fp), 0)
dst = make(1)
dst[0].cache = torch.zeros(7, 3)
m = sc._restore(dst, L, os.path.join(d, fp), 0)
for (k, a), (_, b) in zip(src.state_dict().items(), dst.state_dict().items()):
    assert torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a, b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b), k
assert torch.equal(src[0].cache, dst[0].cache)
assert torch.equal(src[1].scale8.view(torch.uint8), dst[1].scale8.view(torch.uint8))
assert dst.head.weight.data_ptr() == dst[2].weight.data_ptr()
saved = {e["name"]: e["digest"] for e in m["entries"] if e["kind"] == "weight"}
extras = {e["name"]: e["digest"] for e in m["entries"] if e["kind"] == "extra"}
sc._check(0, "roundtrip", saved, {}, extras, dst, L)
# a reorder must change the digest (the APE double-conversion case)
w = dst[0].weight.data
w.copy_(torch.cat([w[16:], w[:16]]))
try:
    sc._check(0, "tampered", saved, {}, extras, dst, L)
    raise SystemExit("tamper not caught")
except RuntimeError as e:
    print("caught:", str(e)[:100])
# missing / unknown weights refuse
other = torch.nn.Sequential(torch.nn.Linear(64, 32))
try:
    sc._restore(other, L, os.path.join(d, fp), 0)
    raise SystemExit("mismatch not caught")
except RuntimeError as e:
    print("refused:", str(e)[:100])
# a second save under another fingerprint removes the older copy
fp2 = "b" * 24
sc._save(src, L, os.path.join(d, fp2), 0, fp2, {})
assert not os.path.exists(os.path.join(d, fp)) and sc._complete(os.path.join(d, fp2), 0)
# fingerprint: stable, env-sensitive, runtime flags ignored
cfg = types.SimpleNamespace(dtype=torch.bfloat16, model_path="/models/DeepSeek-V4.1-Flash")
a, _ = sc._fingerprint(cfg, 4)
os.environ["DSV41_STATS_SECONDS"] = "999"; b, _ = sc._fingerprint(cfg, 4)
os.environ["DSV41_TEST_KNOB"] = "1"; c, inputs = sc._fingerprint(cfg, 4)
assert a == b != c, (a, b, c)
assert not any("KEY" in k or "TOKEN" in k for k in inputs["env"])
sys.argv = ["x", "-m", "sglang.launch_server", "--port", "8888", "--tp", "4", "--api-key", "s3cret", "--mem-fraction-static=0.9", "--trust-remote-code"]
os.getppid = lambda: 999999999
print("args:", sc._launch_args())
assert sc._launch_args() == ["x", "-m", "sglang.launch_server", "--tp", "4", "--trust-remote-code"]
print("ALL OK")
