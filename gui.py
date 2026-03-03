#!/usr/bin/env python3
"""Tkinter GUI for watching trained poker models play.

Usage
-----
    python gui.py
"""

from __future__ import annotations

import math
import os
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

from poker_env.model import ActionMaskModel

# ── Card constants (must match poker_env/evaluator.py) ────────────
RANKS = "23456789TJQKA"
SUITS = "shdc"
SUIT_KEY = {"s": "spade", "h": "heart", "d": "diamond", "c": "club"}
SUIT_CLR = {"s": "#2c3e50", "h": "#c0392b", "d": "#c0392b", "c": "#2c3e50"}

ACTION_LABELS = ["Fold", "Check/Call", "Raise Min", "1/2 Pot", "Pot", "All-In"]

# ── Colour palette ────────────────────────────────────────────────
BG = "#0f1626"
FELT = "#0b6623"
FELT_RIM = "#064e14"
RAIL = "#5d4e37"
RAIL_DK = "#3e2f1c"
GOLD = "#f1c40f"
TXT = "#ecf0f1"
TXT_DIM = "#7f8c8d"
TXT_BLUE = "#3498db"
CARD_FG = "#fafafa"
CARD_BK = "#16537e"
WIN = "#2ecc71"
LOSE = "#e74c3c"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"
PANEL_BG = "#131b2e"


def _card_info(idx: int) -> tuple[str, str, str]:
    """Return (rank_char, suit_key, colour) for a card index 0-51."""
    r, s = idx // 4, idx % 4
    sk = SUITS[s]
    return RANKS[r], SUIT_KEY[sk], SUIT_CLR[sk]


class PokerGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Poker RL Viewer")
        root.geometry("1060x720")
        root.configure(bg=BG)
        root.minsize(820, 580)

        self.algo = None
        self.policy = None
        self.env = None
        self.env_config: dict = {}
        self.obs_dict: dict = {}
        self.hand_over = True
        self.auto_playing = False
        self.continuous_playing = False
        self.hand_count = 0
        self.action_probs: np.ndarray | None = None
        self.value_est: float | None = None
        self.last_action: int | None = None
        self.hand_rewards: dict[str, float] = {}
        self.action_log: list[tuple] = []
        self._resize_job = None

        cards_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards")
        self._suit_imgs: dict[str, tk.PhotoImage] = {}
        for name in ("spade", "heart", "diamond", "club"):
            self._suit_imgs[name] = tk.PhotoImage(
                file=os.path.join(cards_dir, f"{name}.png")
            )

        ray.init(ignore_reinit_error=True)
        ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(120, self._draw_empty)

    # ══════════════════════════════════════════════════════════════
    #  UI construction
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("D.TFrame", background=BG)
        sty.configure("D.TLabel", background=BG, foreground=TXT)
        sty.configure("Dim.TLabel", background=BG, foreground=TXT_DIM)

        # ── row 1: checkpoint ──
        r1 = ttk.Frame(self.root, style="D.TFrame")
        r1.pack(fill=tk.X, padx=10, pady=(8, 0))

        ttk.Label(r1, text="Checkpoint:", style="D.TLabel").pack(side=tk.LEFT)
        self.ckpt_var = tk.StringVar()
        ttk.Entry(r1, textvariable=self.ckpt_var, width=46).pack(side=tk.LEFT, padx=4)
        ttk.Button(r1, text="Browse...", command=self._browse).pack(side=tk.LEFT)

        self.load_btn = ttk.Button(r1, text="Load Model", command=self._load_threaded)
        self.load_btn.pack(side=tk.LEFT, padx=(14, 4))
        self.status_lbl = ttk.Label(r1, text="No model loaded", style="Dim.TLabel")
        self.status_lbl.pack(side=tk.LEFT, padx=6)

        # ── row 2: game settings ──
        r2 = ttk.Frame(self.root, style="D.TFrame")
        r2.pack(fill=tk.X, padx=10, pady=(4, 0))

        for label, var_name, default, lo, hi, w in [
            ("Players:", "n_var", 2, 2, 10, 3),
            ("Stack:", "stack_var", 1000, 50, 100000, 6),
            ("SB:", "sb_var", 5, 1, 500, 4),
            ("BB:", "bb_var", 10, 2, 1000, 4),
            ("Hidden:", "hid_var", 256, 64, 1024, 5),
        ]:
            ttk.Label(r2, text=label, style="D.TLabel").pack(side=tk.LEFT, padx=(6, 0))
            var = tk.IntVar(value=default)
            setattr(self, var_name, var)
            ttk.Spinbox(r2, from_=lo, to=hi, textvariable=var, width=w).pack(
                side=tk.LEFT, padx=2
            )

        # ── canvas ──
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas.bind("<Configure>", self._on_resize)

        # ── controls ──
        bot = ttk.Frame(self.root, style="D.TFrame")
        bot.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.btn_hand = ttk.Button(
            bot, text="New Hand", command=self.new_hand, state=tk.DISABLED
        )
        self.btn_hand.pack(side=tk.LEFT, padx=4)
        self.btn_step = ttk.Button(
            bot, text="> Step", command=self.step_action, state=tk.DISABLED
        )
        self.btn_step.pack(side=tk.LEFT, padx=4)
        self.btn_auto = ttk.Button(
            bot, text=">> Auto", command=self.toggle_auto, state=tk.DISABLED
        )
        self.btn_auto.pack(side=tk.LEFT, padx=4)
        self.btn_continuous = ttk.Button(
            bot, text="Continuous", command=self.toggle_continuous, state=tk.DISABLED
        )
        self.btn_continuous.pack(side=tk.LEFT, padx=4)

        self.determ_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bot, text="Deterministic", variable=self.determ_var
        ).pack(side=tk.LEFT, padx=(12, 2))
        self.cash_game_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            bot, text="Cash game", variable=self.cash_game_var
        ).pack(side=tk.LEFT, padx=(12, 2))

        ttk.Label(bot, text="Delay:", style="D.TLabel").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.IntVar(value=600)
        ttk.Scale(
            bot,
            from_=50,
            to=2000,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            length=120,
        ).pack(side=tk.LEFT)
        ttk.Label(bot, text="ms", style="D.TLabel").pack(side=tk.LEFT, padx=(0, 8))

        self.info_lbl = ttk.Label(bot, text="", style="Dim.TLabel")
        self.info_lbl.pack(side=tk.RIGHT, padx=8)

        self.root.bind("<space>", lambda _: self.step_action())
        self.root.bind("n", lambda _: self.new_hand())
        self.root.bind("a", lambda _: self.toggle_auto())
        self.root.bind("c", lambda _: self.toggle_continuous())

    # ══════════════════════════════════════════════════════════════
    #  Model loading
    # ══════════════════════════════════════════════════════════════

    def _browse(self):
        p = filedialog.askdirectory(title="Select RLlib Checkpoint")
        if p:
            self.ckpt_var.set(p)

    def _load_threaded(self):
        self.load_btn.configure(state=tk.DISABLED)
        self.status_lbl.configure(text="Loading...", foreground=GOLD)
        self.root.update_idletasks()

        try:
            ckpt = self.ckpt_var.get().strip()
            if not ckpt:
                raise ValueError("Select a checkpoint directory first")

            if self.algo:
                self.algo.stop()

            self.algo = PPO.from_checkpoint(os.path.abspath(ckpt))
            self.policy = self.algo.get_policy("shared_policy")

            train_cfg = self.algo.config.env_config
            self.env_config = {
                "num_players": train_cfg.get("num_players", self.n_var.get()),
                "initial_stack": self.stack_var.get(),
                "small_blind": self.sb_var.get(),
                "big_blind": self.bb_var.get(),
            }
            self.n_var.set(self.env_config["num_players"])
            self._on_loaded()

        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            self.status_lbl.configure(text="Failed", foreground=LOSE)
            self.load_btn.configure(state=tk.NORMAL)

    def _on_loaded(self):
        self.status_lbl.configure(text="OK - Model ready", foreground=WIN)
        self.load_btn.configure(state=tk.NORMAL)
        self.btn_hand.configure(state=tk.NORMAL)
        self.btn_continuous.configure(state=tk.NORMAL)
        self.hand_count = 0
        self._draw_empty()

    # ══════════════════════════════════════════════════════════════
    #  Game logic
    # ══════════════════════════════════════════════════════════════

    def new_hand(self):
        if not self.policy:
            return
        from poker_env.env import PokerEnv

        self.auto_playing = False
        self.btn_auto.configure(text=">> Auto")

        self.env_config = {
            "num_players": self.n_var.get(),
            "initial_stack": self.stack_var.get(),
            "small_blind": self.sb_var.get(),
            "big_blind": self.bb_var.get(),
            "cash_game": self.cash_game_var.get(),
            "max_hands_per_episode": 10_000,
        }
        cash_game = self.cash_game_var.get()
        if cash_game and self.env is not None and self.hand_over:
            self.hand_count += 1
            self.obs_dict, _ = self.env.start_next_hand()
        else:
            self.env = PokerEnv(self.env_config)
            self.hand_count += 1
            self.obs_dict, _ = self.env.reset(seed=self.hand_count)
        self.hand_over = not bool(self.obs_dict)
        self.hand_rewards = {}
        self.action_probs = None
        self.value_est = None
        self.last_action = None
        self.action_log = []

        if not self.hand_over:
            self.action_log.append(("street", "PREFLOP"))
            self._compute_probs()
            self.btn_step.configure(state=tk.NORMAL)
            self.btn_auto.configure(state=tk.NORMAL)
        self._redraw()

    def step_action(self):
        if self.hand_over or not self.obs_dict or not self.policy:
            return

        game = self.env._game
        pre_street = game.street
        acting_pid = game.current_player

        agent = next(iter(self.obs_dict))
        obs = self.obs_dict[agent]

        explore = not self.determ_var.get()
        action, _, _ = self.policy.compute_single_action(obs, explore=explore)
        action = int(action)
        self.last_action = action

        self.action_log.append(("act", acting_pid, ACTION_LABELS[action]))

        self.obs_dict, rewards, terms, _, _ = self.env.step({agent: action})
        self.hand_rewards.update(rewards)

        if not game.is_hand_over() and game.street != pre_street:
            self.action_log.append(("street", game.street.name))

        if terms.get("__all__", False):
            self.hand_over = True
            self.action_probs = None
            self.value_est = None
            self.btn_step.configure(state=tk.DISABLED)
            self.btn_auto.configure(state=tk.DISABLED)
            self.auto_playing = False
            self.btn_auto.configure(text=">> Auto")
            self.action_log.append(("end",))
            if self.continuous_playing:
                self.root.after(self.speed_var.get(), self._continue_next_hand)
        else:
            self._compute_probs()

        self._redraw()

    def _compute_probs(self):
        if not self.obs_dict or not self.policy:
            self.action_probs = self.value_est = None
            return
        agent = next(iter(self.obs_dict))
        obs = self.obs_dict[agent]
        with torch.no_grad():
            obs_t = {
                k: torch.from_numpy(v).unsqueeze(0).float() for k, v in obs.items()
            }
            logits, _ = self.policy.model({"obs": obs_t}, [], None)
            self.action_probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
            self.value_est = self.policy.model.value_function().item()

    def toggle_auto(self):
        if self.hand_over:
            return
        self.auto_playing = not self.auto_playing
        self.btn_auto.configure(
            text="|| Pause" if self.auto_playing else ">> Auto"
        )
        if self.auto_playing:
            self._tick()

    def toggle_continuous(self):
        self.continuous_playing = not self.continuous_playing
        self.btn_continuous.configure(
            text="Stop" if self.continuous_playing else "Continuous"
        )
        if self.continuous_playing:
            self.new_hand()
            if not self.hand_over:
                self.auto_playing = True
                self.btn_auto.configure(text="|| Pause", state=tk.NORMAL)
                self._tick()
        else:
            self.auto_playing = False
            self.btn_auto.configure(text=">> Auto")

    def _continue_next_hand(self):
        if not self.continuous_playing:
            return
        self.new_hand()
        if self.continuous_playing and not self.hand_over:
            self.auto_playing = True
            self.btn_auto.configure(text="|| Pause", state=tk.NORMAL)
            self._tick()

    def _tick(self):
        if not self.auto_playing or self.hand_over:
            return
        self.step_action()
        if self.auto_playing and not self.hand_over:
            self.root.after(self.speed_var.get(), self._tick)

    # ══════════════════════════════════════════════════════════════
    #  Drawing helpers
    # ══════════════════════════════════════════════════════════════

    def _on_resize(self, _event):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(
            80, lambda: self._redraw() if self.env else self._draw_empty()
        )

    def _wh(self):
        return self.canvas.winfo_width() or 1040, self.canvas.winfo_height() or 460

    # ── empty state ──

    def _draw_empty(self):
        c = self.canvas
        c.delete("all")
        w, h = self._wh()
        self._felt(w, h)
        c.create_text(
            w // 2, h // 2,
            text="Load a model and click  New Hand",
            fill=TXT_DIM, font=("Helvetica", 18),
        )

    # ── full redraw ──

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = self._wh()
        self._felt(w, h)

        if not self.env or not self.env._game:
            return

        g = self.env._game
        cx, cy = w // 2, h // 2 - 10
        rx = min(w * 0.35, 310)
        ry = min(h * 0.33, 180)

        self._draw_board(cx, cy - 30, g)

        pot_str = f"Pot  {g.pot:,}"
        c.create_text(cx, cy + 20, text=pot_str, fill=TXT, font=("Helvetica", 14, "bold"))

        acting = g.current_player if not g.is_hand_over() else -1
        for i in range(g.num_players):
            a = math.radians(90 + 360 * i / g.num_players)
            px = cx + rx * math.cos(a)
            py = cy + ry * math.sin(a)
            self._draw_seat(px, py, i, g, acting == i)

        if self.action_probs is not None:
            self._draw_probs(w)

        self._draw_log(h)

        if self.hand_over and self.hand_rewards:
            self._draw_results(w, h, g)

        street = g.street.name if not g.is_hand_over() else "SHOWDOWN"
        act_txt = f"Player {acting}" if acting >= 0 else "--"
        self.info_lbl.configure(
            text=f"Hand #{self.hand_count}  |  {street}  |  Acting: {act_txt}"
        )

    # ── table felt ──

    def _felt(self, w, h):
        c = self.canvas
        cx, cy = w // 2, h // 2 - 10
        rx, ry = min(w * 0.40, 370), min(h * 0.39, 215)
        c.create_oval(
            cx - rx + 5, cy - ry + 5, cx + rx + 5, cy + ry + 5,
            fill="#080e1c", outline="",
        )
        c.create_oval(
            cx - rx, cy - ry, cx + rx, cy + ry,
            fill=RAIL, outline=RAIL_DK, width=3,
        )
        p = 8
        c.create_oval(
            cx - rx + p, cy - ry + p, cx + rx - p, cy + ry - p,
            fill=FELT, outline=FELT_RIM, width=2,
        )

    # ── community cards ──

    def _draw_board(self, cx, cy, g):
        cw, ch = 42, 58
        gap = 7
        total = 5 * cw + 4 * gap
        x0 = cx - total // 2
        for i in range(5):
            x = x0 + i * (cw + gap)
            if i < len(g.board):
                self._card(x, cy - ch // 2, cw, ch, g.board[i])
            else:
                self.canvas.create_rectangle(
                    x, cy - ch // 2, x + cw, cy + ch // 2,
                    fill="#0a4a1a", outline="#1a6a2a", dash=(3, 3),
                )

    # ── single card ──

    def _card(self, x, y, w, h, idx):
        c = self.canvas
        rk, suit_key, clr = _card_info(idx)
        c.create_rectangle(x, y, x + w, y + h, fill=CARD_FG, outline="#bbb")
        c.create_text(
            x + w // 2, y + h * 0.30,
            text=rk, fill=clr, font=("Helvetica", 14, "bold"),
        )
        c.create_image(x + w // 2, y + h * 0.67, image=self._suit_imgs[suit_key])

    def _card_back(self, x, y, w, h):
        c = self.canvas
        c.create_rectangle(x, y, x + w, y + h, fill=CARD_BK, outline="#0d3b56")
        c.create_line(x + 4, y + 4, x + w - 4, y + h - 4, fill="#1a6a9e", width=1)
        c.create_line(x + w - 4, y + 4, x + 4, y + h - 4, fill="#1a6a9e", width=1)

    # ── player seat ──

    def _draw_seat(self, px, py, pid, g, acting):
        c = self.canvas
        cw, ch = 36, 50

        if acting:
            c.create_oval(
                px - 62, py - 56, px + 62, py + 62,
                outline=GOLD, width=2, dash=(5, 3),
            )

        for j, card in enumerate(g.hole_cards[pid]):
            cx = px - cw - 1 + j * (cw + 3)
            if g.folded[pid]:
                c.create_rectangle(
                    cx, py - ch // 2, cx + cw, py + ch // 2,
                    fill="#2a2a2a", outline="#444",
                )
                c.create_text(
                    cx + cw // 2, py, text="X", fill="#555",
                    font=("Helvetica", 13, "bold"),
                )
            else:
                self._card(cx, py - ch // 2, cw, ch, card)

        ly = py + ch // 2 + 6
        name = f"Player {pid}"
        badges = []
        if pid == g.dealer_idx:
            badges.append("D")
        if g.folded[pid]:
            badges.append("FOLD")
        if g.all_in[pid]:
            badges.append("ALL-IN")
        if badges:
            name += f"  [{', '.join(badges)}]"

        nc = GOLD if acting else (TXT_DIM if g.folded[pid] else TXT)
        c.create_text(px, ly, text=name, fill=nc, font=("Helvetica", 9, "bold"))
        c.create_text(
            px, ly + 14,
            text=f"Stack {g.stacks[pid]:,}", fill="#aaa", font=("Helvetica", 8),
        )
        if g.bets[pid]:
            c.create_text(
                px, ly + 26,
                text=f"Bet {g.bets[pid]:,}", fill=ORANGE, font=("Helvetica", 8, "bold"),
            )

    # ── action probabilities panel ──

    def _draw_probs(self, canvas_w):
        c = self.canvas
        px = canvas_w - 200
        py = 12
        pw = 185
        bh = 17
        gap = 4
        ph = len(ACTION_LABELS) * (bh + gap) + 48

        c.create_rectangle(
            px - 6, py - 6, px + pw + 6, py + ph,
            fill=PANEL_BG, outline="#34495e", width=1,
        )
        c.create_text(
            px + pw // 2, py + 10,
            text="Action Probabilities", fill=TXT_DIM, font=("Helvetica", 8, "bold"),
        )

        mask = np.ones(len(ACTION_LABELS))
        if self.obs_dict:
            agent = next(iter(self.obs_dict))
            mask = self.obs_dict[agent]["action_mask"]

        bar_max = pw - 62
        y = py + 26
        for i, lbl in enumerate(ACTION_LABELS):
            prob = float(self.action_probs[i])
            legal = mask[i] > 0
            lc = TXT if legal else "#555"

            c.create_text(
                px, y + bh // 2, text=lbl, fill=lc, font=("Helvetica", 8), anchor="w",
            )
            bx = px + 54
            c.create_rectangle(bx, y + 2, bx + bar_max, y + bh - 2, fill="#1c2636", outline="")

            if legal and prob > 0.005:
                chosen = self.last_action == i
                if chosen:
                    fill = LOSE
                elif prob > 0.3:
                    fill = WIN
                elif prob > 0.1:
                    fill = ORANGE
                else:
                    fill = TXT_BLUE
                c.create_rectangle(
                    bx, y + 2, bx + max(int(bar_max * prob), 2), y + bh - 2,
                    fill=fill, outline="",
                )

            pct = f"{prob:.0%}" if legal else "--"
            c.create_text(
                bx + bar_max + 4, y + bh // 2, text=pct, fill=lc,
                font=("Helvetica", 7), anchor="w",
            )
            y += bh + gap

        if self.value_est is not None:
            c.create_text(
                px + pw // 2, y + 8,
                text=f"V = {self.value_est:+.1f}", fill=PURPLE,
                font=("Helvetica", 10, "bold"),
            )

    # ── action log ──

    def _draw_log(self, canvas_h):
        c = self.canvas
        if not self.action_log:
            return
        px, py = 14, 14
        c.create_text(
            px, py, text="Action Log", fill=TXT_DIM,
            font=("Helvetica", 8, "bold"), anchor="nw",
        )
        y = py + 18
        visible = self.action_log[-16:]
        for entry in visible:
            kind = entry[0]
            if kind == "act":
                _, pid, name = entry
                c.create_text(
                    px + 4, y, text=f"P{pid}: {name}", fill=TXT,
                    font=("Helvetica", 8), anchor="nw",
                )
            elif kind == "street":
                _, sname = entry
                c.create_text(
                    px, y, text=f"-- {sname} --", fill=TXT_BLUE,
                    font=("Helvetica", 8, "bold"), anchor="nw",
                )
            elif kind == "end":
                c.create_text(
                    px, y, text="-- END --", fill=ORANGE,
                    font=("Helvetica", 8, "bold"), anchor="nw",
                )
            y += 15

    # ── results overlay ──

    def _draw_results(self, canvas_w, canvas_h, g):
        c = self.canvas
        from poker_env import evaluator

        lines: list[str] = []
        for agent in sorted(self.hand_rewards):
            r = self.hand_rewards[agent]
            pid = int(agent.split("_")[1])
            icon = "W" if r > 0 else " "
            lines.append(f"[{icon}] {agent}: {r:+.0f}")

        active = [i for i in range(g.num_players) if not g.folded[i]]
        if len(active) > 1 and len(g.board) == 5:
            lines.append("")
            for pid in active:
                rank = evaluator.hand_rank_string(g.hole_cards[pid], g.board)
                lines.append(f"  P{pid}: {rank}")

        txt = "\n".join(lines)
        box_h = max(18 * len(lines) + 28, 60)
        box_w = 190
        margin = 14
        x0, y0 = margin, canvas_h - margin - box_h

        c.create_rectangle(
            x0, y0, x0 + box_w, y0 + box_h,
            fill=BG, outline=GOLD, width=2,
        )
        c.create_text(
            x0 + box_w // 2, y0 + 12,
            text="Results", fill=GOLD, font=("Helvetica", 10, "bold"),
        )
        c.create_text(
            x0 + box_w // 2, y0 + box_h // 2 + 6,
            text=txt, fill=TXT, font=("Helvetica", 9), justify=tk.CENTER,
        )

    # ══════════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════════

    def _on_close(self):
        if self.algo:
            try:
                self.algo.stop()
            except Exception:
                pass
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        self.root.destroy()


def main():
    os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    root = tk.Tk()
    PokerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
