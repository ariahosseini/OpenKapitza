# Cloud runbook: the CPU job set on one rented machine

For running the remaining CPU jobs without a university cluster: one large
cloud machine, four jobs side by side, about half a day. The jobs, why each
one is run and what it answers: [`scripts/prepare_cloud.py`](../scripts/prepare_cloud.py)
(KB §42). For a SLURM cluster use [`cluster-runbook.md`](cluster-runbook.md);
for GPU (machine-learned potentials) see section 7 below.

| | |
|---|---|
| machine | AWS **c7a.48xlarge**: 192 AMD EPYC (Zen 4) cores, no hyperthreading, 384 GB |
| layout | 4 jobs × 48 MPI ranks (`slots=4 np=48`) |
| job set | 14 jobs: Si/Al and Si/Cu (MEAM), incoherent Si/Ge, r07 re-run, 4 drive and damping follow-ups, 5 disordered interfaces |
| estimate | about 10–11 h wall (the longest job, Si/Cu, sets it); the smoke run in step 3 replaces this with a measured figure |
| cost | spot about $2.8/h → **≈ $30–40** with setup; on-demand about $9.9/h → ≈ $110 |

Prices are us-east-1, checked November 2026; spot prices move, so look at
the current one when launching (EC2 console → Spot Requests → Pricing history).

## 0. This week, before the day (one-time)

1. **Raise the vCPU quota. Do this first: it can take a day or two.** New
   AWS accounts are allowed only a handful of vCPUs, and this machine needs
   192. Service Quotas → Amazon EC2 → *All Standard (A, C, D, H, I, M, R, T, Z)
   Spot Instance Requests* → request **192** (and the same for *Running
   On-Demand Standard … instances*, as a fallback). Pick the region now
   (us-east-1 or us-east-2 are usually cheapest) and request it there.
2. Create a key pair (EC2 → Key Pairs, `.pem`), save it as
   `~/.ssh/openkapitza.pem` on the Mac and run `chmod 400 ~/.ssh/openkapitza.pem`.
3. Set a budget alarm (Billing → Budgets, e.g. $60) so a forgotten machine
   is noticed.

## 1. Launch (10 minutes)

EC2 → Launch instance:

- **Image:** Ubuntu Server 24.04 LTS, 64-bit (x86).
- **Instance type:** c7a.48xlarge.
- **Key pair:** the one from step 0.
- **Network:** allow SSH from *My IP* only.
- **Storage:** 100 GiB gp3 (the jobs write a few GB, mostly checkpoints; the
  LAMMPS build and Python need about 5 GB).
- **Advanced → Purchasing option:** Spot instances, request type
  **persistent**, interruption behaviour **Stop** (not terminate): if AWS
  takes the machine back, the disk is kept and the runs resume from their
  checkpoints when it restarts (step 5).

Copy the public IPv4 address from the instance page.

## 2. Set up (about 20–30 minutes, mostly the LAMMPS build and the smoke run)

```bash
ssh -i ~/.ssh/openkapitza.pem ubuntu@<IP>
git clone https://github.com/ariahosseini/OpenKapitza.git
cd OpenKapitza
bash scripts/cloud_setup.sh
```

[`scripts/cloud_setup.sh`](../scripts/cloud_setup.sh) installs the system
packages, builds LAMMPS stable_30Sep2026 (the validated commit) with MPI and
every package the decks use, installs OpenKapitza in `.venv`, runs the test
suite, writes the job set into `~/ok_cloud`, and then runs a **smoke copy of
every job** (a few hundred steps per stage) to the end on this machine. It
stops at the first problem. It finishes by printing the exact commands for
step 4.

## 3. Read the smoke result

The script prints one line per job, `PASS` or `FAIL`, then
`14/14 passed` and the speed against the cost model. **Start the real runs
only if all 14 passed.** A failure prints its reason (details in
`~/ok_smoke/smoke_report.json`); stop the machine (step 6) and send the report.
The measured speed is fed back into the wall-time estimates, which the script
prints last.

## 4. Run

```bash
tmux new -s ok                       # survives a dropped SSH connection
source ~/OpenKapitza/.venv/bin/activate
python3 -m openkapitza.agent
> jobs ~/ok_cloud
> submit all local lmp=~/lammps-openkapitza/build/lmp np=48 slots=4
> watch
```

- The longest jobs start first; the queue starts each next job by itself as
  one finishes, so closing the agent or the SSH session does not stop it.
- Detach with `Ctrl-b d`, log out; later `ssh` in again and `tmux attach -t ok`.
- `jobs` (in the agent) shows what is running, finished and measured. A
  failed job is diagnosed with the knowledge-base entry that explains it.
- `htop` should show about 192 busy `lmp` processes.

## 5. If the spot machine was interrupted

AWS stops the machine (disk kept) and, with a persistent request, starts it
again when capacity returns; the IP address may change. Then:

```bash
ssh -i ~/.ssh/openkapitza.pem ubuntu@<new IP>
tmux new -s ok
source ~/OpenKapitza/.venv/bin/activate
python3 -m openkapitza.agent
> jobs ~/ok_cloud                    # finds the killed jobs and restarts them
> resubmit all                       # only if any is still shown as failed
> watch
```

The agent recognises that the machine restarted (it does not trust process
ids from before the restart), and a job killed by the machine is restarted
from its last checkpoint by itself. `resubmit all` restarts only jobs that
died; finished and running ones are left alone. A job loses at
most the time since its last stage checkpoint.

## 6. Bring the results home and stop paying

From the Mac, when `jobs` shows nothing running:

```bash
rsync -av --exclude '*.bin' --exclude 'restart.*' --exclude '*.dump' \
  -e "ssh -i ~/.ssh/openkapitza.pem" ubuntu@<IP>:ok_cloud/ ~/ok_jobs/cloud/
```

This copies the logs, profiles, tallies, specs and the job book (a few
hundred MB), and leaves the large checkpoint files behind. Check that
`~/ok_jobs/cloud/heavy/sicu_pp/log.lammps` ends with `OPENKAPITZA_DONE`. Then
**terminate** the instance *and cancel the spot request* (EC2 → Spot
Requests → Cancel; with a persistent request, terminating alone lets AWS start
a new one), and delete the volume if it was not deleted with the instance.
The EC2 dashboard should show 0 running instances and 0 volumes.

Analysis at home: `python3 -m openkapitza.agent`, then `jobs ~/ok_jobs/cloud`
measures and ingests every finished run.

## 7. GPU (machine-learned potentials)

Not part of this set. Test first on the free Colab T4 with
[`notebooks/colab_mlip_gpu.ipynb`](../notebooks/colab_mlip_gpu.ipynb); the
rented GPU (A100 class, Vast.ai or RunPod) comes after that works. The GPU
build is the KOKKOS-CUDA one in [`cluster-runbook.md`](cluster-runbook.md#0b-a-second-gpu-build-for-tier-2-mlips-gpu-nodes-only).

## Other machines

Anything with 64+ physical cores and Ubuntu works: `cloud_setup.sh` counts
the physical cores and sets `np` = cores / 4 (override with `NP=… SLOTS=…`).
Avoid counting hyperthreads as cores: on machines sold by "vCPU" with SMT
(most Intel instances, Hetzner CCX), 192 vCPU are 96 cores, and the run takes
about twice as long.
