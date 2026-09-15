# autoresearch

This is an experiment to have the LLM do its own research to improve a Triton kernel.

The repo supports **multiple tasks**, each in its own subfolder. Every task has `prepare*.py` scripts (the read-only evaluation harness) plus editable files containing implementations, and helper code. The test is run using `prepare.py`. For example:

- `<task>/` — `prepare*.py` (read-only harness) + `code_*.py` and other helper files (editable; discover by listing the folder)
- `lib_sparse/` — `prepare.py`, `prepare_layer.py` (read-only harness) + `code_pack.py`, `code_unpack.py`, `code_main.py` (editable)

## Setup

To set up a new experiment, work with the user to:

1. **Pick a task**: e.g. `lib_sparse`. The task subfolder contains the code for that experiment. Ask the user which task they want to work on.
2. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun6`). The branch `autoresearch/<task>/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<task>/<tag>` from current main.
4. **Read the in-scope files**: List the task folder to find the exact filenames, then read all relevant files for full context:
   - `{task}/prepare*.py` — the evaluation harness. These are the scripts you **run**, but **do not modify**.
   - `{task}/code_*.py` and any other `.py` files in the task folder NOT starting with `prepare` — these are the editable files. They contain the Triton kernel(s) and helper code under test.
5. **Initialize results.tsv**: Create `results_{task}.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. You launch it from the repo root as: `python3 {task}/prepare.py` (after ensuring the mamba env `optimiser` is active). The prepare script imports the editable modules via local imports, so run it from the repo root — the working directory matters.

**What you CAN do:**
- Modify any file in `{task}/` **except** files starting with `prepare` (e.g. `prepare.py`, `prepare_layer.py`). Almost everything else is fair game, just no cheating.
- Change Triton kernels — block size, grid strategy, memory access patterns, use of atomics, etc.
- Change host-side helper functions — different ways to compute block prefixes, different launch parameters, etc.
- Add new Triton kernels.
- Use autotune configuration to try different launch parameters, to a reasonable extent.

**What you CANNOT do:**
- Modify `{task}/prepare*.py`. These are read-only. They contain the fixed evaluation, data generation, and correctness checks.
- Install new packages or add dependencies. You can only use what's already in this env.
- Change the evaluation harness or the data generation.

**The goal is simple: make the kernel faster.** The evaluation runs the kernel on a range of shapes, times it, and checks correctness. Additional constraints may also need to be satisfied (e.g. efficiency). Total runtime across all shapes is the key metric — lower is better. Correctness is a hard requirement; if the kernel produces wrong results, the run is a failure.

**VRAM** is a soft constraint. It should not blow up dramatically and crash the program.

**Simplicity criterion**: All else being equal, simpler is better. A small speed improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better speed is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A ~1% speedup that adds 20 lines of hacky code? Probably not worth it. A 10% speedup from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the prepare script as is.

## Output format

Once the script finishes it prints timing for each shape, then a total:

```
Shape (128, 2048): Time 1.3 ms
Shape (128, 5000): Time 2.1 ms
...
passed
Total time: 4.567
```

The key metric is the **Total time** printed at the end. You can extract it directly from the log:

```
grep "^Total time:" run.log
```

NOTE: Lower is better. The total time is the sum of per-shape average times (each averaged over n runs).

## Logging results

When an experiment is done, log it to `results_{task}.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Each task gets its own results file.

The TSV has a header row and 4 columns:

```
commit   time   status   description
```

1. git commit hash (short, 7 chars)
2. Total time (e.g. 0.04567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	time	status	description
a1b2c3d	0.0457	keep	baseline
b2c3d4e	0.0420	keep	increase BLOCK size to 512
c3d4e5f	0.0510	discard	use atomic_add for prefix
d4e5f6g	0.0000	crash	add triton prefix kernel (invalid memory access)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/lib_sparse/jun6`).
Before starting, make sure any needed env is activated.
LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune the editable files in `{task}/` (everything except files starting with `prepare`) with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python3 {task}/prepare.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: Check that "passed" appears (correctness). Then get the total: `grep "^Total time:" run.log`. The rest of the log may be useful.
6. If the output is missing "passed" or errors appear, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (`results_{task}.tsv`). Commit all results, including suboptimal results. — NOTE: do not commit the results.tsv file, leave it untracked by git.
8. If Total time improved (lower), you "advance" the branch, keeping the git commit.
9. If Total time is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this sparingly.

**Timeout**: Each experiment should be quick (a few seconds). If a run exceeds 2 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, invalid memory access, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read Triton documentation, re-read the in-scope files for new angles, try combining previous near-misses, try more radical kernel redesigns. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!
