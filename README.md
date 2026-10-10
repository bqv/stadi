# stadi

A GPU **slot broker** for a single card shared by two engines that cannot share it.

`stadi` sits in front of a [llama.cpp](https://github.com/ggml-org/llama.cpp) router and
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and owns the card, so
neither engine can OOM the other.

## The problem

On a one-tenant card this is not a scheduling nicety, it is a correctness problem:

- The llama router keeps a resident model — 17.6–19.7 GiB of a 20 GiB card.
- A Qwen-Image DiT wants ~13 GiB, plus its VAE and a text encoder.
- **`sd-server` cannot unload.** There is no flag and no endpoint (grep
  `examples/server/*.cpp` for `unload`, `evict`, `release` — there is no such route); only the
  *process exiting* frees the weights. Measured: 12.5 GiB → 0.03 GiB.
- Neither engine knows the other exists, so whichever loads second OOMs mid-load.

So a plain always-on `sd-server` holds ~12.5 GiB until a human stops it, and **a local model
cannot generate an image by itself**: to call the image model it would have to unload the very
model making the call.

## What stadi does

It listens on `127.0.0.1:55558` and fronts both engines:

| request | what happens |
| :-- | :-- |
| **media** — `/v1/images/*`, and *only* `/sdapi/v1/txt2img` and `/img2img` | unload the router's models, spawn an `sd-server` child, proxy to it |
| **chat** — everything else, e.g. `/v1/chat/completions`, `/models/load` | stop the child, **wait for the VRAM to actually return**, proxy to the router |
| **metadata** — `/v1/models`, `/props` | proxy untouched; the router loads nothing for these, and a picker polls them constantly |
| **observation** — other `/sdapi/*` (`/progress`, `/options`, `/sd-models`, …) | proxy only if loaded, otherwise `503 sd is not loaded` |

Plus a self-inflicted `keep_alive`: after `STADI_IDLE` with no media request the child is
stopped and the card frees itself. That is the feature sd.cpp does not have.

The generation/observation split is not cosmetic. Without it a monitoring loop polling
`/sdapi/v1/progress` would spawn the child and pin the card forever, defeating the idle timer
entirely.

**State needs no flags: the child's existence *is* the state.** `GET /status` reports
`sd_child_up`, so "is sd loaded?" is answered by process liveness rather than a boolean that
can desync.

## Measured

Numbers from the machine it was written for — a 20 GiB RX 7900 XT, 4-thread Haswell, Vulkan:

| | |
| :-- | :-- |
| cold render, 20 steps | **508.6 s** (weights still page-cached) |
| cold render, **1** step | **706.6 s** |
| warm render | ~148.6 s ≈ 20 steps × 7 s — **pure sampling** |
| idle release | fired **exactly** 5 min after the render ended; **12.136 → 0.026 GiB in 1 s** |
| eviction on a chat request | child stopped and card released **before** the router was asked; the chat then answered in 51 s |

The 1-step run being *slower* than the 20-step run is the interesting one: **the cost is the
load, not the sampling.** ~700 s is weights and init whatever the step count, which is why the
idle timeout is the lever that matters rather than the step count.

## Build

Stdlib only — no dependencies, no module cache, no network:

```sh
go build -o stadi .
```

## Configure

Everything comes from the environment:

| variable | default |
| :-- | :-- |
| `STADI_LISTEN` | `127.0.0.1:55558` |
| `STADI_LLM` | `http://127.0.0.1:55555` |
| `STADI_SD` | `http://127.0.0.1:55557` — where the **child** comes up |
| `STADI_IDLE` | `5m` — stop the child after this long with no media request |
| `STADI_BUSY_WAIT` | `2m` — how long a chat request waits for a render before answering 503 |
| `STADI_VRAM_WAIT` | `45s` — bound on waiting for the card to come back |
| `STADI_LOAD_WAIT` | `15m` — bound on the child's port appearing |
| `STADI_SD_BIN` | `/usr/bin/sd-server` |
| `STADI_SD_ARGS` | **required** — the child's whole command line |
| `STADI_SD_LOG_DIR` | `/var/log` — the child's stdout/stderr |

Note on `STADI_SD_ARGS`: do **not** pass `--params-backend` for stable-diffusion.cpp. Its
`--help` says an explicit `--params-backend` **disables `--auto-fit`**, and auto-fit is what
performs the graph segmentation that tiles a VAE decode too large for the card. Pinning the
*params* split throws away the fallback; pinning `--backend` is safe and is preserved.

## Design notes worth keeping

Two bugs found while building it, both of which would have been intermittent and miserable:

1. **The idle timer must be armed from the *end* of a request, not the start.** It is first
   armed when the child is ensured — before the render — and a cold render can take ten
   minutes. A five-minute timer armed there kills the child mid-load. It now re-arms when the
   request completes, and refuses to fire while a render is in flight.
2. **The child must be killed on SIGTERM.** It is started in its own process group
   (`Setpgid`) so that stopping it takes any children with it — which also means it does *not*
   die when stadi does. Without a signal handler, a service restart orphans ~12.5 GiB that
   nothing owns any more.

And one operational trap: a `supervise-daemon` user service is stopped with SIGTERM, so a
broker that spawns long-lived children **must** handle that signal. Cheap to do, expensive to
skip.

## Licence

MPL-2.0 (see `LICENSE`). The Mozilla Public License is FILE-LEVEL copyleft: you may combine
this with differently-licensed code, but modifications to MPL-covered *files* must stay MPL.
