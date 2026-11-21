// Push-optimal Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Idea (from the level structure, not from any other solver): the generator
// lays a 1-wide hallway with 1-4 small rooms off it, 3-4 boxes per room, and
// a box pushed out of a room through its door can never come back. So every
// box stays in its room, the level optimum P* is the sum of the rooms'
// optima, and each room can be solved on its own, push-optimally, then the
// rooms stitched together by walking the player between doors.
//
// Per room this runs a push-space A*: a state is the boxes' squares plus the
// player's reachable region (named by its lowest square). The heuristic is
// the cheapest box->goal assignment under the lone-box push distance
// (reverse pull search from each goal, other boxes ignored), which is
// admissible and consistent, so the first goal state popped is
// push-optimal. Deadlock pruning: a box may only stand on a square from
// which some goal is reachable by pulls, and no frozen 2x2 block of walls
// and boxes may hold an off-goal box. Fallbacks per room: plain push BFS,
// then greedy best-first (any solution beats none: a solved-but-long
// answer still scores ~500*P*/pushes, an unsolved one scores 0).
//
// Budgets are counted in work (expansions / states), never in wall time, so
// two runs on one set write the same bytes. No clock, no randomness, no
// environment read except the input/output directories (as the starter).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order everywhere (up, down, left, right): determinism
// depends on it, so never reorder.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

// Work budgets (counts, never time). The whole set gets one shared pool so
// a hard early room cannot starve the rest, and every search also has its
// own cap. Rooms in the generated sets are kept only when the reference
// push search solves them within 250k states, and a room's whole state
// space is small (4 boxes on ~25 squares: at most ~50k states), so these
// caps sit far above anything a generated room needs — the practice set
// uses ~2,400 expansions total, a BFS-only run ~50k — while the shared
// pool bounds the worst case to roughly half the 60s wall even on a
// slower single CPU (~25k expansions/s measured here).
const TOTAL_EXPANSIONS = 800_000;
const ASTAR_EXPANSIONS = 200_000;
const BFS_EXPANSIONS = 200_000;
const GREEDY_EXPANSIONS = 100_000;
const MAX_STATES = 500_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * For each goal, the fewest pushes a lone box needs to reach it from every
 * square, found by pulling the box backwards from the goal (other boxes
 * ignored). A square is live when some goal reaches it; a box off a live
 * square can never be solved. The pull step: a box at x came from x - d
 * when the player could stand at x - 2d.
 */
function pushDistances(w: number, wall: Uint8Array, goals: number[]): { dist: Int32Array[]; live: Uint8Array } {
  const size = wall.length;
  const live = new Uint8Array(size);
  const dist: Int32Array[] = [];
  const hh = wall.length / w;
  for (const g of goals) {
    const d = new Int32Array(size).fill(-1);
    d[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      const xx = x % w;
      const xy = (x / w) | 0;
      for (let k = 0; k < 4; k++) {
        const px = xx - DX[k]!;
        const py = xy - DY[k]!;
        const sx = xx - 2 * DX[k]!;
        const sy = xy - 2 * DY[k]!;
        if (px < 0 || px >= w || py < 0 || py >= hh || sx < 0 || sx >= w || sy < 0 || sy >= hh) continue;
        const prev = py * w + px;
        const stand = sy * w + sx;
        if (wall[prev] || wall[stand] || d[prev] >= 0) continue;
        d[prev] = d[x]! + 1;
        queue.push(prev);
      }
    }
    for (let i = 0; i < size; i++) if (d[i] >= 0) live[i] = 1;
    dist.push(d);
  }
  return { dist, live };
}

/** Cheapest assignment of boxes to goals under the push distances, or Infinity. */
function matchingHeuristic(boxes: number[], goals: number[], dist: Int32Array[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  // Cost matrix, row per box, column per goal.
  const cost: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      const v = dist[j]![boxes[i]!]!;
      row.push(v);
    }
    cost.push(row);
  }
  if (n <= 5) {
    // Brute-force all assignments (n <= 5: at most 120).
    let best = Infinity;
    const used = new Array<boolean>(n).fill(false);
    const rec = (i: number, sum: number): void => {
      if (sum >= best) return;
      if (i === n) {
        best = sum;
        return;
      }
      for (let j = 0; j < n; j++) {
        if (used[j] || cost[i]![j]! < 0) continue;
        used[j] = true;
        rec(i + 1, sum + cost[i]![j]!);
        used[j] = false;
      }
    };
    rec(0, 0);
    return best;
  }
  // Many boxes: sum of each box's own cheapest goal (still a lower bound).
  let sum = 0;
  for (let i = 0; i < n; i++) {
    let m = Infinity;
    for (let j = 0; j < n; j++) if (cost[i]![j]! >= 0 && cost[i]![j]! < m) m = cost[i]![j]!;
    if (m === Infinity) return Infinity;
    sum += m;
  }
  return sum;
}

/**
 * The level's rooms: cut the dead (non-live) squares of the player's row
 * (the hallway: a box there reaches no goal) out of the floor; each
 * connected part left that holds a box is a room. Null when a box starts in
 * the hallway, a goal falls outside every room, or some room's boxes and
 * goals do not match.
 */
function roomsOf(level: Level): { squares: Uint8Array; boxes: number[]; goals: number[] }[] | null {
  const { w, h, wall, goals, boxes, player } = level;
  const { live } = pushDistances(w, wall, goals);
  const row = (player / w) | 0;
  const size = w * h;
  const cut = new Uint8Array(size);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: { squares: Uint8Array; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = new Uint8Array(size);
    const stack = [s];
    part[s] = rooms.length;
    squares[s] = 1;
    while (stack.length > 0) {
      const p = stack.pop()!;
      const px = p % w;
      const py = (p / w) | 0;
      for (let k = 0; k < 4; k++) {
        const qx = px + DX[k]!;
        const qy = py + DY[k]!;
        if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
        const q = qy * w + qx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares[q] = 1;
          stack.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  const withBoxes = rooms.filter((r) => r.boxes.length > 0);
  withBoxes.sort((a, b) => {
    const amin = a.boxes.reduce((m, x) => Math.min(m, x), Infinity);
    const bmin = b.boxes.reduce((m, x) => Math.min(m, x), Infinity);
    return amin - bmin;
  });
  return withBoxes;
}

interface SearchOpts {
  maxExpansions: number;
  useHeuristic: boolean; // false = plain push BFS (f = g)
  greedy: boolean; // true = best-first on h only (any solution, not optimal)
  budget: { n: number };
}

/**
 * Push-space search over one (sub-)level: {pushes, moves} or null when the
 * search gives up (budget spent) or the level has no solution. Boxes may
 * only be pushed onto `allowed` squares (null = anywhere on the floor).
 */
function pushSearch(
  w: number,
  h: number,
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  allowed: Uint8Array | null,
  opts: SearchOpts,
): { pushes: number; moves: string } | null {
  const size = w * h;
  const n = boxes0.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const solved = (bs: number[]): boolean => bs.every((b) => goalSet[b]);
  if (solved(boxes0)) return { pushes: 0, moves: '' };
  const { dist, live } = pushDistances(w, wall, goals);
  if (boxes0.some((b) => !live[b])) return null;
  const numericKey = (n + 1) * Math.log2(size) < 53;
  const keyOf = (bs: number[], low: number): number | string => {
    if (numericKey) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return bs.join(',') + '@' + low;
  };
  const hOf = (bs: number[]): number => (opts.useHeuristic ? matchingHeuristic(bs, goals, dist) : 0);

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // Flood the player's region from `from` with `occupied` in the way;
  // returns its lowest square, marking the region in `seen` with `stamp`.
  const floodLow = (from: number): number => {
    stamp++;
    const cur = stamp;
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    seen[from] = cur;
    while (sp > 0) {
      const p = stack[--sp]!;
      if (p < low) low = p;
      const px = p % w;
      const py = (p / w) | 0;
      for (let k = 0; k < 4; k++) {
        const qx = px + DX[k]!;
        const qy = py + DY[k]!;
        if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
        const q = qy * w + qx;
        if (!wall[q] && !occupied[q] && seen[q] !== cur) {
          seen[q] = cur;
          stack[sp++] = q;
        }
      }
    }
    return low;
  };

  // Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
  // and boxes holding a box off its goal.
  const frozenBlock = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (a < 0 || d >= size) continue;
      const ax = a % w;
      if (ax === w - 1) continue; // block would wrap a row
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };

  // States: boxes, the player's square after the push that made the state,
  // parent state, and the push (from square, direction) that made it.
  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [player0];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sDir: number[] = [-1];
  const sG: number[] = [0];
  const bestG = new Map<number | string, number>();
  for (const b of boxes0) occupied[b] = 1;
  bestG.set(keyOf(boxes0, floodLow(player0)), 0);
  for (const b of boxes0) occupied[b] = 0;

  // Binary heap on [f, g, seq, state]: lowest f first, deeper g first on a
  // tie, then earliest inserted. seq makes every order deterministic.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = hOf(boxes0);
  if (h0 === Infinity) return null;
  heap.push([h0, 0, seq++, 0]);
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const tmp = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  let expansions = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, g, , s] = heapPop();
    const boxes = sBoxes[s]!;
    for (const b of boxes) occupied[b] = 1;
    const low = floodLow(sPlayer[s]!);
    const known = bestG.get(keyOf(boxes, low));
    if (known !== undefined && known < g) {
      for (const x of boxes) occupied[x] = 0;
      continue;
    }
    if (solved(boxes)) {
      found = s;
      for (const x of boxes) occupied[x] = 0;
      break;
    }
    if (expansions >= opts.maxExpansions || opts.budget.n >= TOTAL_EXPANSIONS) {
      for (const x of boxes) occupied[x] = 0;
      return null;
    }
    expansions++;
    opts.budget.n++;
    const player = sPlayer[s]!;
    const here = stamp; // floodLow above already marked this state's region
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b / w) | 0;
      for (let k = 0; k < 4; k++) {
        const tx = bx + DX[k]!;
        const ty = by + DY[k]!;
        if (tx < 0 || tx >= w || ty < 0 || ty >= h) continue;
        const t = ty * w + tx;
        if (wall[t] || occupied[t] || !live[t] || (allowed !== null && !allowed[t])) continue;
        const sx = bx - DX[k]!;
        const sy = by - DY[k]!;
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
        const pushFrom = sy * w + sx;
        if (seen[pushFrom] !== here) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(t);
        let next: number[] | null = null;
        let key: number | string = -1;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          // The player ends on the box's old square; flood from there.
          stamp++;
          const cur = stamp;
          let low = b;
          let sp = 0;
          stack[sp++] = b;
          mark[b] = cur;
          while (sp > 0) {
            const p = stack[--sp]!;
            if (p < low) low = p;
            const px = p % w;
            const py = (p / w) | 0;
            for (let k2 = 0; k2 < 4; k2++) {
              const qx = px + DX[k2]!;
              const qy = py + DY[k2]!;
              if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
              const q = qy * w + qx;
              if (!wall[q] && !occupied[q] && mark[q] !== cur) {
                mark[q] = cur;
                stack[sp++] = q;
              }
            }
          }
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const g2 = g + 1;
        const prevBest = bestG.get(key);
        if (prevBest !== undefined && prevBest <= g2) continue;
        if (sBoxes.length >= MAX_STATES) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        const h2 = hOf(next);
        if (h2 === Infinity) continue;
        bestG.set(key, g2);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sDir.push(k);
        sG.push(g2);
        heapPush([opts.greedy ? h2 : g2 + h2, g2, seq++, sBoxes.length - 1]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Reconstruct: for each push on the path, walk the player to the push
  // square (BFS, fixed direction order), then the push itself.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0.slice();
  for (const s of path) {
    const k = sDir[s]!;
    const from = sFrom[s]!;
    const to = from + DY[k]! * w + DX[k]!;
    const stand = from - DY[k]! * w - DX[k]!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, h, wall, occupied, player, stand);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH[k]!;
    player = from;
    const at = boxes.indexOf(from);
    boxes[at] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves };
}

/** Shortest walk from `from` to `to` around walls and boxes, lower-case letters. */
function walkTo(w: number, h: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = w * h;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int32Array(size).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    const px = p % w;
    const py = (p / w) | 0;
    for (let k = 0; k < 4; k++) {
      const qx = px + DX[k]!;
      const qy = py + DY[k]!;
      if (qx < 0 || qx >= w || qy < 0 || qy >= h) continue;
      const q = qy * w + qx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = STEP[how[p]!]! + s;
  return s;
}

/** The square the player stands on after `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

/** Solve one level room by room; null when any room gives up. */
function solveByRooms(level: Level, budget: { n: number }): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  for (let r = 0; r < rooms.length; r++) {
    const room = rooms[r]!;
    const wall = level.wall.slice();
    for (let o = 0; o < rooms.length; o++) {
      if (o === r) continue;
      const other = rooms[o]!;
      for (const b of done[o] ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = { w: level.w, h: level.h, wall, goals: room.goals, boxes: room.boxes, player };
    let solved =
      pushSearch(sub.w, sub.h, sub.wall, sub.goals, sub.boxes, sub.player, room.squares, {
        maxExpansions: ASTAR_EXPANSIONS,
        useHeuristic: true,
        greedy: false,
        budget,
      }) ??
      pushSearch(sub.w, sub.h, sub.wall, sub.goals, sub.boxes, sub.player, room.squares, {
        maxExpansions: BFS_EXPANSIONS,
        useHeuristic: false,
        greedy: false,
        budget,
      }) ??
      pushSearch(sub.w, sub.h, sub.wall, sub.goals, sub.boxes, sub.player, room.squares, {
        maxExpansions: GREEDY_EXPANSIONS,
        useHeuristic: true,
        greedy: true,
        budget,
      });
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done[r] = true;
  }
  return { pushes, moves };
}

/** Whole-level fallback when the level is no set of rooms. */
function solveWhole(level: Level, budget: { n: number }): { pushes: number; moves: string } | null {
  return (
    pushSearch(level.w, level.h, level.wall, level.goals, level.boxes, level.player, null, {
      maxExpansions: ASTAR_EXPANSIONS,
      useHeuristic: true,
      greedy: false,
      budget,
    }) ??
    pushSearch(level.w, level.h, level.wall, level.goals, level.boxes, level.player, null, {
      maxExpansions: GREEDY_EXPANSIONS,
      useHeuristic: true,
      greedy: true,
      budget,
    })
  );
}

function solveLevel(level: Level, budget: { n: number }): string {
  const solved = solveByRooms(level, budget) ?? solveWhole(level, budget);
  if (solved === null) return '';
  return replayValid(level, solved.moves) ? solved.moves : '';
}

/** Replay moves on the level: true only when every step is legal and every box ends on a goal. */
function replayValid(level: Level, moves: string): boolean {
  const { w, wall, goals } = level;
  const goalSet = new Uint8Array(w * level.h);
  for (const g of goals) goalSet[g] = 1;
  const box = new Uint8Array(w * level.h);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    const k = c === 'u' ? 0 : c === 'd' ? 1 : c === 'l' ? 2 : 3;
    const d = DY[k]! * w + DX[k]!;
    const next = player + d;
    if (wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) if (box[i] && !goalSet[i]) return false;
  return true;
}

const budget = { n: 0 };
for (const id of readdirSync(INPUT_DIR).sort()) {
  const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  const moves = solveLevel(level, budget);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
