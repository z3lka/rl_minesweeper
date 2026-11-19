import type { CSSProperties } from "react";
import type { Game } from "./game/board";

interface Props {
  game: Game;
  heat?: Float32Array | null;
  pick?: number;
  onCell?: (cell: number) => void;
  showMines?: boolean;
  label: string;
  /** 0..1 per cell: how much each open number matters to the cell being explained. */
  why?: Float32Array | null;
  target?: number;
  onHover?: (cell: number | null) => void;
}

function MineMark() {
  return (
    <svg className="mine-mark" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2.5" />
      <circle cx="12" cy="12" r="4" fill="currentColor" />
    </svg>
  );
}

/** The board. Hidden cells are tinted by `heat`; `pick` is ringed. */
export function Board({ game, heat, pick, onCell, showMines, label, why, target, onHover }: Props) {
  const { h, w } = game.spec;
  const cells = [];
  for (let i = 0; i < h * w; i++) {
    const revealed = game.revealed[i] === 1;
    const mine = game.mines?.[i] === 1;
    const exposed = !revealed && mine && (game.hit === i || (showMines && game.done));
    const value = revealed ? game.counts[i] : 0;
    const weight = revealed && why && why[i] > 0.25 ? why[i] : 0;
    const cls = [
      "cell",
      revealed ? `open n${value}` : "shut",
      exposed ? "mine" : "",
      game.hit === i ? "hit" : "",
      pick === i && !game.done ? "pick" : "",
      target === i && pick !== i && !revealed && !game.done ? "target" : "",
      weight ? "why" : "",
    ]
      .filter(Boolean)
      .join(" ");
    const style = weight
      ? ({ "--why": `${(1 + weight * 3).toFixed(1)}px` } as CSSProperties)
      : !revealed && heat && !Number.isNaN(heat[i])
        ? ({ "--heat": heat[i].toFixed(3) } as CSSProperties)
        : undefined;
    const text = revealed && value > 0 ? value : exposed ? <MineMark /> : null;
    const name = `Row ${Math.floor(i / w) + 1}, column ${(i % w) + 1}, ${revealed ? (value ? `${value} mines nearby` : "empty") : exposed ? "mine" : "hidden"}`;
    cells.push(
      onCell ? (
        <button key={i} type="button" className={cls} style={style} aria-label={name} disabled={revealed || game.done} onClick={() => onCell(i)} onMouseEnter={onHover && !revealed ? () => onHover(i) : undefined} onFocus={onHover && !revealed ? () => onHover(i) : undefined}>
          {text}
        </button>
      ) : (
        <span key={i} className={cls} style={style}>
          {text}
        </span>
      ),
    );
  }
  return (
    <div
      className={`board ${w > 9 ? "dense" : ""}`}
      style={{ "--cols": w } as CSSProperties}
      role={onCell ? "group" : "img"}
      aria-label={label}
      onMouseLeave={onHover ? () => onHover(null) : undefined}
    >
      {cells}
    </div>
  );
}

/** A still picture of a board, drawn as SVG. Colours follow the surrounding field. */
export function BoardPicture({ game, heat, pick, label }: Omit<Props, "onCell" | "showMines">) {
  const { h, w } = game.spec;
  const S = 20;
  const cells = [];
  for (let i = 0; i < h * w; i++) {
    const x = (i % w) * S + 1;
    const y = Math.floor(i / w) * S + 1;
    const open = game.revealed[i] === 1;
    const hot = !open && heat && !Number.isNaN(heat[i]) ? heat[i] : 0;
    cells.push(
      <rect
        key={i}
        x={x}
        y={y}
        width={S - 2}
        height={S - 2}
        rx={1.5}
        style={{ fill: open ? "var(--open, var(--light))" : `color-mix(in srgb, var(--accent) ${(hot * 100).toFixed(0)}%, var(--cell))` }}
      />,
    );
    if (open && game.counts[i] > 0)
      cells.push(
        <text key={`t${i}`} className={`n${game.counts[i]}`} x={x + S / 2 - 1} y={y + S / 2 + 2.6} textAnchor="middle">
          {game.counts[i]}
        </text>,
      );
    if (pick === i && !open)
      cells.push(<rect key="pick" className="ring" x={x + 1.5} y={y + 1.5} width={S - 5} height={S - 5} rx={1} />);
  }
  return (
    <svg className="board-picture" viewBox={`0 0 ${w * S} ${h * S}`} role="img" aria-label={label}>
      {cells}
    </svg>
  );
}
