# muse_bf16_proj notes (2026-11-11)

Board: leader 1.7277 (gogowhy); ours 1.156 (job c2503232, Oct 22). Challenge ends Nov 19.

## Measured on H200 (scratch/v2/h200.py harness: graph replay, 512 MB flush, empty graph subtracted)
- Harness reads ~0.8-1.7 us higher than the judge for the same kernel.
- Stream floor (ldg, 132x1024 thr): 6.3 MB 4.06 us, 12.6 MB 5.6-5.9, 28.3 MB 9.4-9.8, 50.3 MB 14.2-14.6, 67 MB 18.1.
- Empty kernel ~1.9-2.0 us. Dynamic smem carveout: +0.15 us @32 KB, +0.5 @64 KB, +1.1 @200 KB.
- 66 CTAs x 1024 thr ~= 132 CTAs for streaming; 512-thread CTAs at 66 are much worse.
- L2 bulk prefetch (cp.async.bulk.prefetch.L2): always worse.
- Mixing x loads (ld.global.nc, even 4 lanes) into a warp's W register ring: catastrophic.
- ptxas puts ALL ldg in a loop on one scoreboard and waits for zero: "compute stage d, reload d" rings
  serialize to one latency per stage. Use batch / double-buffer (issue next, compute current, promote)
  or cp.async smem rings with wait_group N (old hgemv did this and was the best m=1 pooler: 15.55 judge).
- Runtime-selected accumulator arrays (`mma(cond ? acc1 : acc0)`) go to local memory: big slowdown.
- Code size / forward jumps / kernel params: no measurable cold-start cost.

## Format
12-bit: raw low byte + nibble (sign, 3-bit code of exp>>1 vs window). Fixed-base decode (0x38|c):
exact 4^(base-56) rescale at output; ~10 int ops + 4 prmt per 8 weights. Exceptions (~1e-4) patched
into A registers (per-lane slots for split-K GEMV; per-tile sorted list for full-K wgmma).

## wgmma m>=128 (scratch/wg/wg1..wg4.py): all CORRECT, all slower than cuBLAS
- wgmma RS A-fragment layout = mma.m16n8k16 A per warp (verified), SW128 desc (LBO 1, SBO 1024, +32 B per k16),
  natural-K packing for this path (k = 64u + 32p + 16h + 8cc2 + 2tq + cc1).
- Best: unpooler m=128 28.7 us vs cuBLAS 20.6 (wg3: 2 WGs same rows split K). ~280-300 TFLOPs; consumers
  starved (x ring shallow), compute pipeline not saturated. Needs a proper CUTLASS-style design.

## Public writeups (techniques only, no code copied)
thethings.ai/p/okc/muse-bf16-proj-report: same 12-bit idea, 1.34x; their wgmma attempt lost to cuBLAS.
github NathenFernandes/muse-bf16-proj (1.52x), shashank-100/okc (1.46x), operatorstack (1.31x), ruwwww (1.51x).

## Hybrid (scratch/hybrid/hybrid1.py) = old submission + v2 GEMV for some m<=8 shapes; works, harness ~same.
