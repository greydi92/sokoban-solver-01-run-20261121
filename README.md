[![make.fun](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Rank](https://make.fun/badge/sokoban-solver-01-run-20261121/greydi92/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![Reward](https://make.fun/badge/sokoban-solver-01-run-20261121/greydi92/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

# Push-Optimal Room Solver — Sokoban, 5/5 at P*

A push-optimal Sokoban solver for the [make.fun bounty "Build the best Sokoban solver, 21 November 2026"](https://make.fun/bounty/sokoban-solver-01-run-20261121).

The generator lays a 1-wide hallway with 1–4 small rooms off it, and a box pushed out of a room through its door can never come back — so every box stays in its room and the level optimum P\* is the sum of the rooms' optima. This solver splits each level into its rooms, solves every room push-optimally on its own, and stitches the rooms together by walking the player between doors.

Per room it runs a push-space A\*: a state is the boxes' squares plus the player's reachable region (named by its lowest square). The heuristic is the cheapest box→goal assignment under the lone-box push distance (reverse pull search from each goal, other boxes ignored) — admissible and consistent, so the first goal state popped is push-optimal. Deadlock pruning keeps a box only on squares from which some goal is reachable by pulls, and rejects frozen 2×2 blocks of walls and boxes. Fallbacks per room: plain push BFS, then greedy best-first (any solution beats none). Budgets are counted in work (expansions), never in wall time, so two runs write the same bytes.

Practice set: **1,000,000** — all 5 instances solved in exactly P\* pushes (13, 16, 29, 47, 60). The house baseline (plain-astar 1.0.0) scores 200,000 on the same set.

## Run

Requires Node 22 or later, no dependencies, no network.

```sh
node main.ts
```

This writes `solution.txt` for every instance under `/output` (or `$MAKEFUN_OUTPUT_DIR` when set):

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=/tmp/out node main.ts
```

The output is deterministic: every run writes the same bytes.

## License

MIT — see [LICENSE](LICENSE).

---

Built for [make.fun](https://make.fun) — referral code `qgn48q27`.
