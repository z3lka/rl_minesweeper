import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type CSSProperties } from "react";
import { AGENTS, explain, think, type AgentId, type Thought, type Why } from "./agents";
import { Board, BoardPicture } from "./Board";
import { compact, duration, find, pct, type Result, type Results } from "./data";
import { Game, mulberry32, PRESETS } from "./game/board";
import { analyze } from "./game/solver";

const LEARNERS: AgentId[] = ["supervised", "dqn", "ppo"];
const agentInfo = (id: AgentId) => AGENTS.find((a) => a.id === id)!;

function useResults(): Results | null {
  const [results, setResults] = useState<Results | null>(null);
  useEffect(() => {
    fetch("data/results.json")
      .then((r) => r.json())
      .then(setResults)
      .catch(() => setResults({}));
  }, []);
  return results;
}

/** Fade and lift the marked parts of a section the first time it is seen. */
function useReveal(ready: boolean) {
  useEffect(() => {
    const nodes = document.querySelectorAll<HTMLElement>("[data-reveal]");
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries)
          if (e.isIntersecting) {
            e.target.classList.add("in");
            io.unobserve(e.target);
          }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    nodes.forEach((n) => io.observe(n));
    return () => io.disconnect();
  }, [ready]);
}

const reveal = (i = 0) => ({ "data-reveal": "", style: { "--d": `${i * 80}ms` } as CSSProperties });

/* ------------------------------------------------------------------ hero */

const EPS = 1e-6;
const risk = (x: number) => (x < EPS ? "0%" : x < 0.01 ? "under 1%" : `${Math.round(x * 100)}%`);

/* ------------------------------------------------------------ why this cell */
const share = (x: number) => (x > 0.9995 ? "100%" : x > 0.99 ? `${(x * 100).toFixed(1)}%` : `${Math.round(x * 100)}%`);
const figure = (id: AgentId, x: number) => (id === "dqn" ? x.toFixed(2) : share(x));

function whyText(id: AgentId, why: Why, game: Game): { claim: string; evidence: string } {
  const { w } = game.spec;
  const v = why.value;
  const next = why.second;
  let claim =
    id === "solver"
      ? v > 0.9995
        ? "Every mine layout that fits the numbers leaves this cell empty, so it is provably safe."
        : `This cell is empty in ${share(v)} of the mine layouts that fit the numbers, the best odds on the board.`
      : id === "supervised"
        ? `It rates this cell ${share(v)} likely to be safe${next !== null ? `, and its best other cell ${share(next)}` : ""}.`
        : id === "dqn"
          ? `It expects a reward of ${v.toFixed(2)} from this click${next !== null ? `, and ${next.toFixed(2)} from its best other one` : ""}.`
          : `This click holds ${share(v)} of its preference${next !== null ? `; its best other one holds ${share(next)}` : ""}.`;
  if (id !== "solver" && why.truth !== null) claim += ` The true chance of a mine here is ${risk(why.truth)}.`;

  let top = -1;
  let move = 0;
  why.without.forEach((x, k) => {
    const d = Math.abs(x - v);
    if (!Number.isNaN(d) && d > move) {
      move = d;
      top = k;
    }
  });
  let evidence: string;
  if (why.cells.length === 0) evidence = "No numbers are in view near this cell, so nothing on the board argues for or against it.";
  else if (top < 0 || move < (id === "dqn" ? 0.02 : 0.01))
    evidence = "No single nearby number decides it: cover any one of them and the rest still give almost the same answer.";
  else {
    const cell = why.cells[top];
    evidence = `It leans most on the ${game.counts[cell]} in row ${Math.floor(cell / w) + 1}, column ${(cell % w) + 1}: cover that number and the figure becomes ${figure(id, why.without[top])}. Thicker rings mark the numbers that matter more.`;
  }
  return { claim, evidence };
}

function whyWeights(why: Why, size: number, id: AgentId): Float32Array | null {
  const moves = why.without.map((x) => Math.abs(x - why.value));
  const most = Math.max(0, ...moves.filter((d) => !Number.isNaN(d)));
  if (most < (id === "dqn" ? 0.02 : 0.01)) return null;
  const out = new Float32Array(size);
  why.cells.forEach((cell, k) => (out[cell] = Number.isNaN(moves[k]) ? 0 : moves[k] / most));
  return out;
}

function Demo({ results }: { results: Results | null }) {
  const [boardName, setBoardName] = useState<"beginner" | "intermediate">("beginner");
  const [agent, setAgent] = useState<AgentId>("solver");
  const [game, setGame] = useState(() => Game.random(PRESETS.beginner));
  const [version, bump] = useReducer((x: number) => x + 1, 0);
  const [thought, setThought] = useState<(Thought & { key: string }) | null>(null);
  const [playing, setPlaying] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [tally, setTally] = useState({ won: 0, lost: 0 });
  const stage = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(true);

  const model = (id: AgentId) => find(results, boardName, id)?.model ?? find(results, "beginner", id)?.model;
  const available = (id: AgentId) => id === "solver" || id === "random" || !!model(id);
  const modelUrl = model(agent);
  const key = `${version}:${agent}:${boardName}`;

  // once the networks are known, let the best learner take the board
  const switched = useRef(false);
  useEffect(() => {
    if (!switched.current && results && find(results, "beginner", "supervised")?.model) {
      switched.current = true;
      setAgent("supervised");
    }
  }, [results]);

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    if (stage.current) io.observe(stage.current);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    let live = true;
    if (game.done || !available(agent)) {
      setThought(null);
      return;
    }
    think(agent, game, modelUrl)
      .then((t) => live && setThought({ ...t, key }))
      .catch(() => live && setThought(null));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, modelUrl, game]);

  const deal = useCallback(
    (name = boardName) => {
      setBoardName(name);
      setGame(Game.random(PRESETS[name]));
      bump();
    },
    [boardName],
  );

  const act = useCallback(
    (cell: number) => {
      game.click(cell);
      if (game.done) setTally((t) => (game.won ? { ...t, won: t.won + 1 } : { ...t, lost: t.lost + 1 }));
      bump();
    },
    [game],
  );

  const ready = thought?.key === key && !game.done;

  // explanation of the pending click, or of whichever hidden cell is pointed at
  const [explaining, setExplaining] = useState(false);
  const [hover, setHover] = useState<number | null>(null);
  const [why, setWhy] = useState<(Why & { key: string }) | null>(null);
  const target = ready ? (hover !== null && game.revealed[hover] === 0 ? hover : thought!.pick) : -1;
  const whyKey = `${key}:${target}`;
  useEffect(() => {
    if (!explaining || target < 0 || agent === "random") return;
    let live = true;
    explain(agent, game, target, modelUrl)
      .then((x) => live && setWhy(x ? { ...x, key: whyKey } : null))
      .catch(() => live && setWhy(null));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [explaining, whyKey, modelUrl]);
  const shownWhy = explaining && why?.key === whyKey ? why : null;
  const whyWords = shownWhy ? whyText(agent, shownWhy, game) : null;
  useEffect(() => {
    if (!playing || !visible || document.hidden) return;
    if (game.done) {
      const t = setTimeout(() => deal(), 1800);
      return () => clearTimeout(t);
    }
    if (!ready) return;
    const t = setTimeout(() => act(thought!.pick), boardName === "beginner" ? 520 : 260);
    return () => clearTimeout(t);
  }, [playing, visible, ready, game.done, thought, act, deal, boardName]);

  const safe = game.spec.h * game.spec.w - game.spec.mines;
  const status = game.done
    ? game.won
      ? `Cleared in ${game.moves} moves`
      : `Hit a mine on move ${game.moves}`
    : `Move ${game.moves + 1}, ${game.nRevealed} of ${safe} safe cells open`;
  const info = agentInfo(agent);
  const borrowed = boardName === "intermediate" && LEARNERS.includes(agent) && !find(results, "intermediate", agent)?.model;

  return (
    <div className="demo" data-ui ref={stage} id="demo">
      <div className="demo-bar">
        <label>
          <span>Player</span>
          <select value={agent} onChange={(e) => setAgent(e.target.value as AgentId)}>
            {AGENTS.map((a) => (
              <option key={a.id} value={a.id} disabled={!available(a.id)}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <div className="seg" role="group" aria-label="Board size">
          {(["beginner", "intermediate"] as const).map((name) => (
            <button key={name} type="button" aria-pressed={boardName === name} onClick={() => deal(name)}>
              {PRESETS[name].h} by {PRESETS[name].w}
            </button>
          ))}
        </div>
      </div>
      <dl className="demo-counters">
        <div>
          <dt>Mines</dt>
          <dd>{game.spec.mines}</dd>
        </div>
        <div>
          <dt>Moves</dt>
          <dd>{game.moves}</dd>
        </div>
        <div>
          <dt>Cells left</dt>
          <dd>{safe - game.nRevealed}</dd>
        </div>
      </dl>
      <Board
        game={game}
        heat={ready || game.done ? thought?.heat : null}
        pick={ready ? thought!.pick : undefined}
        onCell={(cell) => {
          setPlaying(false);
          act(cell);
        }}
        why={shownWhy ? whyWeights(shownWhy, game.spec.h * game.spec.w, agent) : null}
        target={explaining && target >= 0 ? target : undefined}
        onHover={explaining ? setHover : undefined}
        showMines
        label={`${game.spec.h} by ${game.spec.w} board with ${game.spec.mines} mines`}
      />
      <p className={`demo-status ${game.done ? (game.won ? "won" : "lost") : ""}`} aria-live="polite">
        {status}
      </p>
      <div className="demo-actions">
        <button
          type="button"
          className="ui-btn strong"
          onClick={() => {
            if (!playing) setExplaining(false);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" className="ui-btn" disabled={!ready || playing} onClick={() => ready && act(thought!.pick)}>
          Step
        </button>
        <button
          type="button"
          className="ui-btn"
          disabled={game.moves === 0}
          onClick={() => {
            setGame(new Game(game.spec, game.scores));
            bump();
          }}
        >
          Restart
        </button>
        <button type="button" className="ui-btn" onClick={() => deal()}>
          New board
        </button>
        <button
          type="button"
          className="ui-btn"
          aria-pressed={explaining}
          onClick={() => {
            if (!explaining) setPlaying(false);
            setHover(null);
            setExplaining((x) => !x);
          }}
        >
          Explain moves
        </button>
        <span className="demo-tally">
          {tally.won} won, {tally.lost} lost
        </span>
      </div>
      {explaining ? (
        <div className="demo-why" aria-live="polite">
          {agent === "random" ? (
            <p>The random player does not look at the board, so there is nothing to explain.</p>
          ) : game.done ? (
            <p>This board is over. Restart it or deal a new one to ask again.</p>
          ) : whyWords ? (
            <>
              <h3>
                {target === thought!.pick ? "Why it will click" : "How it sees"} row {Math.floor(target / game.spec.w) + 1}, column{" "}
                {(target % game.spec.w) + 1}
              </h3>
              <p>{whyWords.claim}</p>
              <p>{whyWords.evidence}</p>
              <p className="aside">
                Found by covering each nearby number in turn and asking again. Point at another hidden cell to ask about
                it, or press Step to play the move.
              </p>
            </>
          ) : (
            <p>Working it out.</p>
          )}
        </div>
      ) : (
        <p className="demo-note">
          {info.heat} The ringed cell is its next click; you can click any cell yourself.
          {borrowed ? " This network only ever trained on 9 by 9 boards." : ""}
        </p>
      )}
    </div>
  );
}

/* --------------------------------------------------------------- players */

function Sheet({ id, result }: { id: AgentId; result?: Result }) {
  const info = agentInfo(id);
  const rows: [string, string][] = [];
  if (result) {
    rows.push(["Games won", pct(result.win_rate)]);
    rows.push(["Safe cells opened per game", pct(result.cleared)]);
    rows.push(["Lost on a forced guess", pct(result.losses.unlucky)]);
    rows.push(["Lost on a worse guess than needed", pct(result.losses.bad_guess)]);
    rows.push(["Lost with a safe cell available", pct(result.losses.blunder)]);
    if (result.train_seconds) rows.push(["Training time", duration(result.train_seconds)]);
    if (result.env_steps) rows.push(["Moves seen in training", compact(result.env_steps)]);
    if (result.params) rows.push(["Network weights", compact(result.params)]);
  }
  const parts = result
    ? [
        { cls: "won", v: result.win_rate, name: "won" },
        { cls: "unlucky", v: result.losses.unlucky, name: "forced guess" },
        { cls: "bad", v: result.losses.bad_guess + result.losses.unknown, name: "worse guess" },
        { cls: "blunder", v: result.losses.blunder, name: "blunder" },
      ]
    : [];
  return (
    <article className="sheet" data-ui>
      <header>
        <h3>{info.name}</h3>
        <p>{info.kind}</p>
      </header>
      <p className="sheet-about">{info.about}</p>
      {result ? (
        <>
          <div className="split" role="img" aria-label={parts.map((p) => `${p.name} ${pct(p.v)}`).join(", ")}>
            {parts.map((p) => (
              <span key={p.cls} className={p.cls} style={{ flexGrow: Math.max(p.v, 0.0001) }} />
            ))}
          </div>
          <table>
            <caption>
              {result.games.toLocaleString("en")} beginner boards, the same for every player
            </caption>
            <tbody>
              {rows.map(([k, v]) => (
                <tr key={k}>
                  <th scope="row">{k}</th>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="sheet-about">This player has not finished training yet.</p>
      )}
    </article>
  );
}

function Players({ results }: { results: Results | null }) {
  const [selected, setSelected] = useState<AgentId>("dqn");
  return (
    <section className="players" id="players">
      <div className="players-index">
        <div>
          <h2 {...reveal(0)}>Five players, one test</h2>
          <div className="names" {...reveal(1)}>
            {AGENTS.map((a) => (
              <button key={a.id} type="button" aria-pressed={selected === a.id} onClick={() => setSelected(a.id)}>
                {a.name}
              </button>
            ))}
          </div>
        </div>
        <dl className="legend" {...reveal(2)}>
          <div>
            <dt>
              <i className="won" />
              Won
            </dt>
            <dd>Every safe cell opened.</dd>
          </div>
          <div>
            <dt>
              <i className="unlucky" />
              Forced guess
            </dt>
            <dd>No cell was provably safe. It took the best odds and lost.</dd>
          </div>
          <div>
            <dt>
              <i className="bad" />
              Worse guess
            </dt>
            <dd>It had to guess, and chose a cell with worse odds than the best.</dd>
          </div>
          <div>
            <dt>
              <i className="blunder" />
              Blunder
            </dt>
            <dd>A provably safe cell existed and it clicked a mine anyway.</dd>
          </div>
        </dl>
      </div>
      <div className="players-stage" {...reveal(2)}>
        <Sheet key={selected} id={selected} result={find(results, "beginner", selected)} />
      </div>
    </section>
  );
}

/* ------------------------------------------------- the same board, four ways */
const RACERS: AgentId[] = ["solver", "supervised", "dqn", "ppo"];
const RACER_NOTE: Partial<Record<AgentId, string>> = {
  solver: "Counts every possible mine layout. No learning.",
  supervised: "Was shown where the mines were while it trained.",
  dqn: "Learned from reward what each click is worth.",
  ppo: "Learned from reward which clicks to prefer.",
};

interface Plan extends Thought {
  risk: number; // true chance that the chosen cell is a mine
  best: number; // lowest such chance anywhere on the board
}


function verdict(p: Plan): string {
  if (Number.isNaN(p.risk)) return "Hit a mine.";
  if (p.best < EPS) return "Blunder: a provably safe cell was open to it.";
  if (p.risk > p.best + EPS) return `Worse guess: it took ${risk(p.risk)} risk when ${risk(p.best)} was on offer.`;
  return `Forced guess: ${risk(p.risk)} was the best odds on the board.`;
}

function Race({ results }: { results: Results | null }) {
  const spec = PRESETS.beginner;
  const center = Math.floor(spec.h / 2) * spec.w + Math.floor(spec.w / 2);
  const url = (id: AgentId) => find(results, "beginner", id)?.model;
  const ids = RACERS.filter((id) => id === "solver" || !!url(id));
  const idKey = ids.join();

  // every player gets the same layout and the same opening click
  const start = useCallback(
    (scores: ArrayLike<number>) =>
      Object.fromEntries(
        RACERS.map((id) => {
          const g = new Game(spec, scores);
          g.click(center);
          return [id, g];
        }),
      ) as Record<AgentId, Game>,
    [spec, center],
  );

  const [games, setGames] = useState(() => start(Game.random(spec).scores));
  const [round, bump] = useReducer((x: number) => x + 1, 0);
  const [plans, setPlans] = useState<{ round: number; by: Partial<Record<AgentId, Plan>> } | null>(null);
  const [verdicts, setVerdicts] = useState<Partial<Record<AgentId, string>>>({});
  const [tally, setTally] = useState<Partial<Record<AgentId, { won: number; played: number }>>>({});
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(false);
  const stage = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    if (stage.current) io.observe(stage.current);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    let live = true;
    (async () => {
      const by: Partial<Record<AgentId, Plan>> = {};
      for (const id of ids) {
        const g = games[id];
        if (g.done) continue;
        const t = await think(id, g, url(id));
        const p = analyze(g.obs(), spec);
        let best = NaN;
        if (p) {
          best = Infinity;
          p.forEach((v) => {
            if (!Number.isNaN(v) && v < best) best = v;
          });
        }
        by[id] = { ...t, risk: p ? p[t.pick] : NaN, best };
      }
      if (live) setPlans({ round, by });
    })().catch(() => {});
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [round, games, idKey]);

  const ready = plans?.round === round;
  const allDone = ids.every((id) => games[id].done);

  const step = useCallback(() => {
    if (!plans || plans.round !== round) return;
    const said: Partial<Record<AgentId, string>> = {};
    const ended: [AgentId, boolean][] = [];
    for (const id of idKey.split(",") as AgentId[]) {
      const g = games[id];
      const p = plans.by[id];
      if (!p || g.done) continue;
      g.click(p.pick);
      if (g.done) {
        ended.push([id, g.won]);
        if (!g.won) said[id] = verdict(p);
      }
    }
    if (ended.length) {
      setVerdicts((v) => ({ ...v, ...said }));
      setTally((t) => {
        const next = { ...t };
        for (const [id, won] of ended) {
          const old = t[id] ?? { won: 0, played: 0 };
          next[id] = { won: old.won + (won ? 1 : 0), played: old.played + 1 };
        }
        return next;
      });
    }
    bump();
  }, [plans, round, games, idKey]);

  const deal = useCallback(
    (scores: ArrayLike<number>) => {
      setGames(start(scores));
      setVerdicts({});
      bump();
    },
    [start],
  );

  // play boards out of sight until one is found where the players end differently
  const [searching, setSearching] = useState(false);
  const alive = useRef(true);
  useEffect(() => () => void (alive.current = false), []);
  const findSplit = useCallback(async () => {
    setPlaying(false);
    setSearching(true);
    const players = idKey.split(",") as AgentId[];
    let fallback: ArrayLike<number> | null = null;
    let found: ArrayLike<number> | null = null;
    try {
      for (let n = 0; n < 120 && !found && alive.current; n++) {
        const trial = start(Game.random(spec).scores);
        let wins = 0;
        for (const id of players) {
          const g = trial[id];
          while (!g.done) g.click((await think(id, g, url(id))).pick);
          if (g.won) wins++;
        }
        if (wins > 0 && wins < players.length) found = trial.solver.scores;
        else if (wins === 0 && !fallback) fallback = trial.solver.scores;
        await new Promise((r) => setTimeout(r));
      }
    } catch {
      /* a model failed to load: keep the current board */
    }
    if (!alive.current) return;
    setSearching(false);
    const scores = found ?? fallback;
    if (scores) {
      deal(scores);
      setPlaying(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idKey, start, deal]);

  useEffect(() => {
    if (!playing || !visible) return;
    if (allDone) {
      setPlaying(false);
      return;
    }
    if (!ready) return;
    const t = setTimeout(step, 650);
    return () => clearTimeout(t);
  }, [playing, visible, ready, allDone, step]);

  return (
    <section className="race" id="race">
      <div className="race-head">
        <h2 {...reveal(0)}>The same board, four ways</h2>
        <p {...reveal(1)}>
          Every player below gets the same mines and the same opening click. Blue shows what each one thinks of the
          hidden cells. Under each board is the true risk of the click it is about to make, so you can see who is
          reasoning and who is gambling. Most boards all four clear, so there is a button to find one they do not.
        </p>
      </div>
      <div className="race-stage" data-ui ref={stage} {...reveal(2)}>
        <div className="race-controls">
          <button
            type="button"
            className="ui-btn strong"
            onClick={() => {
              if (allDone && !playing) deal(Game.random(spec).scores);
              setPlaying((p) => !p);
            }}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" className="ui-btn" disabled={playing || !ready || allDone} onClick={step}>
            Step
          </button>
          <button type="button" className="ui-btn" onClick={() => deal(games.solver.scores)}>
            Restart
          </button>
          <button type="button" className="ui-btn" onClick={() => deal(Game.random(spec).scores)}>
            New board
          </button>
          <button type="button" className="ui-btn" disabled={searching || ids.length < 2} onClick={findSplit}>
            {searching ? "Searching" : "Find a board that splits them"}
          </button>
        </div>
        <div className="lanes">
          {ids.map((id) => {
            const g = games[id];
            const plan = ready ? plans!.by[id] : undefined;
            const worse = !!plan && plan.risk > plan.best + EPS;
            const t = tally[id];
            return (
              <div className="lane" key={id}>
                <h3>{agentInfo(id).name}</h3>
                <p className="lane-note">{RACER_NOTE[id]}</p>
                <Board
                  game={g}
                  heat={plan?.heat}
                  pick={plan?.pick}
                  showMines
                  label={`${agentInfo(id).name} playing a ${spec.h} by ${spec.w} board`}
                />
                <p className={`lane-status ${g.done ? (g.won ? "won" : "lost") : ""}`}>
                  {g.done ? (g.won ? `Cleared in ${g.moves} moves` : `Lost on move ${g.moves}`) : `Move ${g.moves + 1}`}
                </p>
                <p className={`lane-risk ${worse ? "worse" : ""}`} aria-live="polite">
                  {g.done
                    ? (verdicts[id] ?? "Every safe cell opened.")
                    : plan && !Number.isNaN(plan.risk)
                      ? `Next click: ${risk(plan.risk)} chance of a mine${worse ? `, though a ${risk(plan.best)} cell exists` : ""}.`
                      : "Thinking."}
                </p>
                <p className="lane-tally">{t ? `Won ${t.won} of ${t.played} here` : "No boards finished yet"}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ----------------------------------------------------------- four fields */

function Steps() {
  const stages = useMemo(() => {
    const spec = PRESETS.beginner;
    const scores = new Float32Array(spec.h * spec.w);
    const rng = mulberry32(11);
    for (let i = 0; i < scores.length; i++) scores[i] = rng();
    const replay = (moves: number) => {
      const g = new Game(spec, scores);
      let heat: Float32Array | null = null;
      let pick = 40;
      for (let m = 0; m <= moves; m++) {
        if (m > 0) g.click(pick);
        if (g.done) break;
        const p = analyze(g.obs(), spec);
        if (!p) break;
        heat = Float32Array.from(p, (v) => 1 - v);
        if (g.started) {
          pick = -1;
          p.forEach((v, i) => {
            if (!Number.isNaN(v) && (pick < 0 || v < p[pick])) pick = i;
          });
        }
      }
      return { game: g, heat, pick };
    };
    return [replay(0), replay(1), replay(1), replay(7)];
  }, []);
  const fields = [
    { text: "First, every cell is hidden.", heat: false },
    { text: "Then one click opens a region.", heat: false },
    { text: "Then each hidden cell is scored.", heat: true },
    { text: "Then the best one is opened, again.", heat: true },
  ];
  return (
    <section className="steps" id="method">
      {fields.map((f, i) => (
        <div className={`field f${i}`} key={i}>
          {i === 0 && <h2>How a move is made</h2>}
          <div className="field-board" data-ui {...reveal(i)}>
            <BoardPicture
              game={stages[i].game}
              heat={f.heat ? stages[i].heat : null}
              pick={f.heat ? stages[i].pick : undefined}
              label={`Example board: ${f.text}`}
            />
          </div>
          <p>{f.text}</p>
        </div>
      ))}
    </section>
  );
}

/* -------------------------------------------------------------- training */

type XMode = "steps" | "minutes";
type YMode = "win_rate" | "cleared";

function Chart({ results, shown, x, y }: { results: Results | null; shown: Record<string, boolean>; x: XMode; y: YMode }) {
  const W = 440;
  const H = 360;
  const pad = { l: 44, r: 84, t: 16, b: 40 };
  const runs = LEARNERS.map((id) => ({ id, r: find(results, "beginner", id) })).filter((s) => s.r?.curve?.length && shown[s.id]);
  const solver = find(results, "beginner", "solver");
  const y0 = y === "win_rate" ? 0 : 0.6;
  const xs = runs.flatMap((s) => s.r!.curve!.map((p) => (x === "steps" ? p.env_steps : p.wall / 60)));
  const xMax = Math.max(...xs, x === "steps" ? 1e6 : 1);
  const xMin = x === "steps" ? 2e4 : 0;
  const px = (v: number) =>
    pad.l + (W - pad.l - pad.r) * (x === "steps" ? Math.log(v / xMin) / Math.log(xMax / xMin) : v / xMax);
  const py = (v: number) => H - pad.b - (H - pad.t - pad.b) * ((v - y0) / (1 - y0));
  const yTicks = y === "win_rate" ? [0, 0.25, 0.5, 0.75, 1] : [0.6, 0.7, 0.8, 0.9, 1];
  const xTicks = x === "steps" ? [2e4, 2e5, 2e6, 2e7].filter((t) => t <= xMax * 1.01) : [0, 0.25, 0.5, 0.75, 1].map((f) => f * xMax);
  const ends: { id: string; yy: number; label: string }[] = [];
  return (
    <figure className="chart" data-ui>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training curves for the three learned players">
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.l} x2={W - pad.r} y1={py(t)} y2={py(t)} />
            <text className="tick" x={pad.l - 8} y={py(t) + 4} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} className="tick" x={px(Math.max(t, xMin))} y={H - pad.b + 20} textAnchor="middle">
            {x === "steps" ? (t < 1e6 ? `${t / 1e3}k` : `${t / 1e6}M`) : t.toFixed(xMax < 10 ? 1 : 0)}
          </text>
        ))}
        {solver && (
          <>
            <line className="ref" x1={pad.l} x2={W - pad.r} y1={py(solver[y])} y2={py(solver[y])} />
            <text className="tick" x={W - pad.r + 8} y={py(solver[y]) - 6}>
              solver
            </text>
          </>
        )}
        {runs.map(({ id, r }) => {
          const pts = r!.curve!
            .map((p) => ({ xv: x === "steps" ? p.env_steps : p.wall / 60, yv: p[y] }))
            .filter((p) => p.xv >= xMin);
          const d = pts.map((p, i) => `${i ? "L" : "M"}${px(p.xv).toFixed(1)},${py(Math.max(p.yv, y0)).toFixed(1)}`).join("");
          const last = pts[pts.length - 1];
          let yy = py(Math.max(last.yv, y0)) + 12;
          while (ends.some((e) => Math.abs(e.yy - yy) < 14)) yy += 14;
          ends.push({ id, yy, label: agentInfo(id).name });
          return (
            <g key={id} className={`line ${id}`}>
              <path d={d} />
              <text className="end" x={W - pad.r + 8} y={yy}>
                {agentInfo(id).name}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        {y === "win_rate" ? "Games won" : "Safe cells opened"} on 2,000 held-out boards, against{" "}
        {x === "steps" ? "moves seen in training (log scale)" : "minutes of training on one laptop"}.
      </figcaption>
    </figure>
  );
}

function Training({ results }: { results: Results | null }) {
  const [shown, setShown] = useState<Record<string, boolean>>({ supervised: true, dqn: true, ppo: true });
  const [x, setX] = useState<XMode>("steps");
  const [y, setY] = useState<YMode>("win_rate");
  const toggle = (id: AgentId) => (
    <label className="switch">
      <input type="checkbox" checked={shown[id]} onChange={(e) => setShown((s) => ({ ...s, [id]: e.target.checked }))} />
      <span>Show {agentInfo(id).name} on the chart</span>
    </label>
  );
  return (
    <section className="training" id="training">
      <h2 {...reveal(0)}>One network, three teachers</h2>
      <div className="training-stage">
        <Chart results={results} shown={shown} x={x} y={y} />
      </div>
      <div className="training-text">
        <div className="block" {...reveal(1)}>
          <h3>All three learners share one small network.</h3>
          <p>
            It reads the board as ten yes-or-no layers (hidden, or showing 0 to 8) and passes them through six
            convolutional blocks. Out comes one number per cell. Nothing in it depends on the board size, so a network
            trained on 9 by 9 can be dropped onto a larger board unchanged.
          </p>
          <div className="seg" role="group" aria-label="Chart measure">
            <button type="button" aria-pressed={y === "win_rate"} onClick={() => setY("win_rate")}>
              Games won
            </button>
            <button type="button" aria-pressed={y === "cleared"} onClick={() => setY("cleared")}>
              Cells opened
            </button>
          </div>
        </div>
        <div className="block" {...reveal(0)}>
          <h3>The supervised one is handed the answers.</h3>
          <p>
            The simulator knows where the mines are, so every board the network sees comes with a full grid of labels.
            It learns to predict them and plays by opening the cell it trusts most. No reward, no planning: a plain
            classifier, and a very strong one.
          </p>
          {toggle("supervised")}
        </div>
        <div className="block" {...reveal(0)}>
          <h3>DQN has to work it out from reward.</h3>
          <p>
            It plays 512 boards at once, keeps a million past moves in memory and repeatedly asks: was that click worth
            what I predicted? Revealed cells are masked out, a second slowly updated copy of the network supplies the
            targets, and one move in a hundred is random to keep it exploring.
          </p>
          {toggle("dqn")}
        </div>
        <div className="block" {...reveal(0)}>
          <h3>PPO adjusts its habits a little at a time.</h3>
          <p>
            It samples moves from its own probabilities across 1,024 boards, then shifts those probabilities toward
            moves that did better than its value estimate expected, and only ever by a small, clipped step. It learns from
            each batch of play once and throws it away.
          </p>
          {toggle("ppo")}
        </div>
        <div className="block" {...reveal(0)}>
          <h3>Count the cost in moves or in minutes.</h3>
          <p>
            Moves seen is the fair measure of how much experience a method needs. Minutes is what you actually wait on
            a laptop, and it is where a method that reuses old experience pays for its extra arithmetic.
          </p>
          <div className="seg" role="group" aria-label="Chart horizontal axis">
            <button type="button" aria-pressed={x === "steps"} onClick={() => setX("steps")}>
              Moves seen
            </button>
            <button type="button" aria-pressed={x === "minutes"} onClick={() => setX("minutes")}>
              Minutes
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ bars */

function Bars({ results }: { results: Results | null }) {
  const id: AgentId = find(results, "beginner", "dqn")?.curve ? "dqn" : "supervised";
  const run = find(results, "beginner", id);
  const curve = run?.curve ?? [];
  const n = Math.min(12, curve.length);
  const bars = Array.from({ length: n }, (_, i) => curve[Math.round((i * (curve.length - 1)) / Math.max(n - 1, 1))]);
  const max = Math.max(...bars.map((b) => b.win_rate), 0.01);
  const printed = new Set([1, Math.floor(n / 2), n - 1]);
  return (
    <section className="bars" id="result">
      <div className="bars-head">
        <h2 {...reveal(0)}>{run ? pct(run.win_rate) : "Training"}</h2>
        <p {...reveal(1)}>
          of boards won by {agentInfo(id).name}
          {id === "dqn" ? ", taught by reward alone" : ""}
        </p>
      </div>
      <div className="bars-chart" data-ui style={{ "--n": n } as CSSProperties}>
        {bars.map((b, i) => (
          <button
            type="button"
            key={i}
            className={`bar ${i === n - 1 ? "accent" : ""}`}
            style={{ "--v": b.win_rate / max, "--i": i } as CSSProperties}
            aria-label={`${pct(b.win_rate)} of games won after ${compact(b.env_steps)} moves`}
          >
            <span className={printed.has(i) ? "always" : ""}>{pct(b.win_rate, 0)}</span>
            <small>{b.env_steps ? `${compact(b.env_steps)} moves` : "untrained"}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ variations */

const ABLATIONS: [string, string][] = [
  ["dqn_beginner", "DQN as shown above"],
  ["dqn_gamma0", "No lookahead (discount 0 instead of 0.9)"],
  ["dqn_gamma099", "Long lookahead (discount 0.99)"],
  ["dqn_noshaping", "Reward for winning and losing only"],
  ["dqn_3blocks", "Half the network depth"],
];

const BIGGER: [string, string][] = [
  ["solver", "Exact solver"],
  ["supervised_zeroshot", "Supervised, 9 by 9 weights unchanged"],
  ["supervised_intermediate", "Supervised, trained further on 16 by 16"],
  ["dqn_zeroshot", "DQN, 9 by 9 weights unchanged"],
  ["dqn_intermediate", "DQN, trained further on 16 by 16"],
  ["ppo_zeroshot", "PPO, 9 by 9 weights unchanged"],
  ["ppo_intermediate", "PPO, trained further on 16 by 16"],
  ["random", "Random"],
];

function RunTable({ caption, rows, runs }: { caption: string; rows: [string, string][]; runs: Result[] }) {
  const found = rows.map(([run, label]) => ({ label, r: runs.find((x) => x.run === run) })).filter((x) => x.r);
  if (found.length < 2) return null;
  return (
    <table className="runs" {...reveal(1)}>
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Run</th>
          <th scope="col">Games won</th>
          <th scope="col">Blunders</th>
          <th scope="col">Training</th>
        </tr>
      </thead>
      <tbody>
        {found.map(({ label, r }) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            <td>{pct(r!.win_rate)}</td>
            <td>{pct(r!.losses.blunder)}</td>
            <td>{r!.train_seconds ? duration(r!.train_seconds) : "none"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Variations({ results }: { results: Results | null }) {
  const small = results?.beginner ?? [];
  const big = results?.intermediate ?? [];
  if (!small.some((r) => r.run === "dqn_gamma0") && !big.some((r) => r.run.endsWith("_zeroshot"))) return null;
  return (
    <section className="variations" id="variations">
      <div className="variations-text">
        <h2 {...reveal(0)}>Change one thing</h2>
        <p {...reveal(1)}>
          The same code, run again with a single setting altered, and the 9 by 9 networks moved to a 16 by 16 board
          with 40 mines. A blunder is a game lost while a provably safe cell was still on the board.
        </p>
        <p {...reveal(2)}>
          Looking ahead matters most: with no lookahead DQN loses about three and a half points, while the reward for
          each opened cell is worth less than one. On the bigger board the 9 by 9 networks blunder constantly until
          they are trained there, and the gap between learning from answers and learning from reward grows from one
          point to as many as nine.
        </p>
      </div>
      <div className="variations-tables">
        <RunTable caption="DQN on the 9 by 9 board, 8 million moves each (the first row had 15 million)" rows={ABLATIONS} runs={small} />
        <RunTable caption="The 16 by 16 board, 40 mines, 10,000 games" rows={BIGGER} runs={big} />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ cost */

function Cost({ results }: { results: Results | null }) {
  const items = LEARNERS.map((id) => ({ id, r: find(results, "beginner", id) }))
    .filter((s) => s.r?.train_seconds)
    .map((s) => `${duration(s.r!.train_seconds!)} for ${agentInfo(s.id).name}`);
  const total = [...(results?.beginner ?? []), ...(results?.intermediate ?? [])].reduce((t, r) => t + (r.train_seconds ?? 0), 0);
  return (
    <section className="cost" id="cost">
      <h2>What it cost to train</h2>
      <ul>
        <li>One M4 Pro laptop</li>
        {items.map((t) => (
          <li key={t}>{t}</li>
        ))}
        {total > 0 && <li>{duration(total)} for every run on this page</li>}
        <li>No cloud bill</li>
      </ul>
    </section>
  );
}

/* ------------------------------------------------------------------- app */

export function App() {
  const results = useResults();
  useReveal(results !== null);
  const best = LEARNERS.map((id) => find(results, "beginner", id)).filter(Boolean) as Result[];
  const games = find(results, "beginner", "solver")?.games ?? 10000;
  return (
    <>
      <header className="nav">
        <a className="mark" href="#top">
          Sweep
        </a>
        <nav aria-label="Sections">
          <a href="#players">Players</a>
          <a href="#race">Side by side</a>
          <a href="#method">Method</a>
          <a href="#training">Training</a>
          <a href="#cost">Cost</a>
        </nav>
      </header>
      <main id="top">
        <section className="hero">
          <div className="hero-text">
            <h1>Five ways to clear a minefield</h1>
            <p>
              A random clicker, an exact solver and three neural networks play the same {games.toLocaleString("en")}{" "}
              boards. The board on this page is live: watch any of them think, one cell at a time.
            </p>
            <div className="hero-actions">
              <a className="btn" href="#players">
                Compare the players
              </a>
              <a className="link" href="#method">
                How a move is made
              </a>
            </div>
            {best.length > 0 && (
              <dl className="hero-facts">
                {best.map((r) => (
                  <div key={r.run}>
                    <dt>{agentInfo(r.run.split("_")[0] as AgentId).name}</dt>
                    <dd>{pct(r.win_rate)}</dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
          <div className="hero-stage">
            <Demo results={results} />
          </div>
        </section>
        <Players results={results} />
        <Race results={results} />
        <Steps />
        <Training results={results} />
        <Bars results={results} />
        <Variations results={results} />
        <Cost results={results} />
      </main>
      <footer className="tiles">
        <div className="tile t-statement">
          <p>Sweep is a small study in reinforcement learning: one puzzle, five players, every number measured.</p>
          <small>Trained and evaluated on a single laptop.</small>
        </div>
        <nav className="tile t-nav" aria-label="Footer">
          <a href="#players">Players</a>
          <a href="#method">Method</a>
          <a href="#training">Training</a>
          <a href="#result">Result</a>
          <a href="#cost">Cost</a>
        </nav>
        <div className="tile t-board">
          <p>The test board</p>
          <small>9 by 9 cells, 10 mines, first click always opens a region</small>
        </div>
        <div className="tile t-stack">
          <p>PyTorch, NumPy, ONNX Runtime, React</p>
          <small>The networks on this page run in your browser.</small>
        </div>
        <div className="tile t-mark" aria-hidden="true">
          Sweep
        </div>
        <a className="tile t-go" href="#demo">
          <span>Play a board</span>
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <path d="M8 40 L40 8 M16 8 H40 V32" fill="none" stroke="currentColor" strokeWidth="4" />
          </svg>
        </a>
      </footer>
    </>
  );
}
