"""Faithful Theory-of-Mind rollout in a multi-agent segregated treasurehunt.

Three agents share one episode:
  * gem-lover:  IQN policy (loaded from train_preferences.py)
  * food-lover: IQN policy
  * observer:   frozen self-model + belief module (no IQN of its own).

Wider-than-FOV setup:
  The observer's model INPUT is a 15x15 patch centered on itself. Within this
  input, only the inner 9x9 (radius 4) is its ACTUAL perception -- the same FOV
  the IQN agents have. The outer ring (cells at chebyshev distance 5-7 from
  the observer) is BEYOND its sight and gets MASKED.

Architecture (faithful to the spec):
  1. FOV-GATED FIRST-PERSON ACCESS. When a watched agent is inside the
     observer's inner 9x9, the observer reads that agent's OWN first-person
     POV + action and appends them to a 5-frame window for that agent. The
     observer never accesses an agent's POV outside its FOV.
  2. BELIEF ENCODER produces g from each watched agent's window. g is trained
     (in train_belief.py) to encode the watched agent's PREFERENCE (gem-lover
     vs food-lover) via a classification head, with no action-prediction loss.
  3. TWO-PASS ACTION SELECTION. Pass 1 (VISION): frozen base + injected g
     reconstructs the observer's full 15x15 POV; we substitute the
     reconstructed cells into the masked outer ring to form a belief-completed
     observation. Pass 2 (ACTION): frozen base runs on the belief-completed
     POV WITHOUT g; the action head reflects the observer's OWN preference
     (the base is the gem-lover's self-model) applied to the belief-inferred
     world. The observer's action is its own choice, informed by belief.
"""

import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import MovingAgent
from sorrel.buffers import TransformerBuffer
from sorrel.examples.treasurehunt.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt.entities import EmptyEntity, Sand, Wall
from sorrel.examples.treasurehunt.env import TreasurehuntEnv
from sorrel.examples.treasurehunt.world import TreasurehuntWorld
from sorrel.models.pytorch import PyTorchIQN
from sorrel.models.pytorch.transformer import BeliefEncoder, ViTOneHot
from sorrel.observation.observation_spec import OneHotObservationSpec
from sorrel.utils.visualization import ImageRenderer

# ==========================================================================
# Configuration
# ==========================================================================

DATA_DIR = Path(__file__).parent / "../data"
SELF_MODEL_PATH = DATA_DIR / "checkpoints/self_model.pkl"
BELIEF_PATH = DATA_DIR / "checkpoints/belief.pkl"  # FOV-gated first-person belief encoder
GEM_IQN_PATH = DATA_DIR / "checkpoints/treasurehunt_model_gemlover.pkl"
FOOD_IQN_PATH = DATA_DIR / "checkpoints/treasurehunt_model_foodlover.pkl"

EPISODE_LEN = 50
# Which watched agents to spawn alongside the observer:
#   "gem_only"  -> just gem-lover
#   "food_only" -> just food-lover
#   "both"      -> gem-lover + food-lover
#   "none"      -> no watched agents (pure base baseline)
WATCHED_CONFIG = "both"
BELIEF_ON = True  # when False, observer ignores g (pure frozen base baseline)
GEM_PREF = {"Gem": 1.0, "Food": 0.2, "Bone": 1.0}
FOOD_PREF = {"Gem": 0.2, "Food": 1.0, "Bone": 1.0}

# (no separate OUT_GIF -- ImageRenderer.save_gif writes to DATA_DIR/gifs/)

# Frozen-base architecture (must match training)
#
# Channels:
#   0  EmptyEntity     (all-zero entity vector — same as a masked cell, so we
#                       added a dedicated mask channel below to distinguish)
#   1  Wall
#   2  Gem
#   3  Bone
#   4  Food
#   5  TreasurehuntAgent
#   6  Mask            (NEW): 1 where the observer cannot actually see the cell
#                       (outer ring beyond inner 9x9 FOV), 0 elsewhere.
# Entity channels 0..5 are zeroed wherever mask channel == 1.
ARCH = dict(
    state_size=(7, 15, 15),  # 6 entity channels + 1 mask channel
    action_space=4,
    layer_size=192,
    patch_size=3,
    num_frames=5,
    num_heads=3,
    batch_size=1,
    num_layers=2,
)
NUM_ENTITY_CHANNELS = 6      # underlying obs_spec channels (gem/food/etc buffers stay 6-ch)
MASK_CHANNEL = 6             # index of the dedicated mask channel
ACTUAL_FOV_RADIUS = 4        # observer's real perception radius (9x9 inner of the 15x15 input)
ENTITY_LIST = ["EmptyEntity", "Wall", "Gem", "Bone", "Food", "TreasurehuntAgent"]
ACTION_NAMES = ["up", "down", "left", "right"]
GEM_CH, FOOD_CH, AGENT_CH = 2, 4, 5

# Precomputed outer-ring 2D mask (1 in outer ring, 0 inside inner FOV).
def _outer_ring_2d_np(H: int, W: int, radius: int = ACTUAL_FOV_RADIUS) -> np.ndarray:
    """1 in outer ring (chebyshev > radius from center), 0 inside."""
    cy, cx = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return (np.maximum(np.abs(yy - cy), np.abs(xx - cx)) > radius).astype(np.float32)


_H, _W = ARCH["state_size"][1], ARCH["state_size"][2]
OUTER_RING_2D_NP = _outer_ring_2d_np(_H, _W, ACTUAL_FOV_RADIUS)         # (H, W) numpy
OUTER_RING_2D = torch.from_numpy(OUTER_RING_2D_NP)                       # (H, W) torch


def add_mask_channel(states_6ch: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
    """Zero out entity channels under mask_2d and append mask_2d as channel 6.

    Args:
        states_6ch: (..., 6, H, W) — entity channels only.
        mask_2d:    (H, W) — 1 where masked, 0 elsewhere.

    Returns:
        (..., 7, H, W) with entity channels zeroed inside the mask and channel
        6 set to mask_2d everywhere.
    """
    H, W = mask_2d.shape
    leading = states_6ch.shape[:-3]
    # Broadcastable mask: (1, ..., 1, 1, H, W)
    mask_b = mask_2d.view(*([1] * (states_6ch.dim() - 2)), H, W)
    masked = states_6ch * (1.0 - mask_b)
    # Mask channel with same leading dims as states_6ch but 1 channel
    mask_channel = mask_b.expand(*leading, 1, H, W).to(states_6ch.dtype)
    return torch.cat([masked, mask_channel], dim=-3)


def add_zero_mask_channel(states_6ch: torch.Tensor) -> torch.Tensor:
    """Append a zero-valued mask channel to a 6-channel state (nothing masked).

    Used for the belief encoder's input: the watched agent has no masked
    region, so its mask channel is all zeros. This keeps encoder and base
    operating on consistent 7-channel inputs.
    """
    leading = states_6ch.shape[:-3]
    H, W = states_6ch.shape[-2], states_6ch.shape[-1]
    zero_ch = torch.zeros(*leading, 1, H, W, dtype=states_6ch.dtype, device=states_6ch.device)
    return torch.cat([states_6ch, zero_ch], dim=-3)

def _make_config(watched_config: str):
    """Build the env config for a given watched-agent configuration."""
    if watched_config == "gem_only":
        prefs = [GEM_PREF]
    elif watched_config == "food_only":
        prefs = [FOOD_PREF]
    elif watched_config == "both":
        prefs = [GEM_PREF, FOOD_PREF]
    elif watched_config == "none":
        prefs = []
    else:
        raise ValueError(f"Unknown watched_config: {watched_config!r}")
    return OmegaConf.create(
        {
            "experiment": {
                "epochs": 1,
                "max_turns": EPISODE_LEN,
                "record_period": 50,
            },
            "model": {
                "agent_vision_radius": 7,  # 15x15 obs (matches train_preferences.py)
                "epsilon_decay": 0.0,
                "num_agents": len(prefs),  # IQN watched agents only
                "preferences": prefs,
                "save_weights": False,
            },
            "world": {
                "height": 20,
                "width": 20,
                "gem_value": 10,
                "food_value": 10,
                "bone_value": -10,
                "spawn_prob": 0.01,
                # Non-segregated: must match the world the pipeline was trained in
                # (see train_preferences.py).
                "segregate": False,
                "bone_fraction": 0.2,
            },
        }
    )


# ==========================================================================
# Observer agent
# ==========================================================================


class _ObserverModelStub:
    """Minimal model wrapper: holds the frozen self-model + belief encoder, plus
    a fresh TransformerBuffer that records the observer's own (state, action)
    history -- used to provide the self-model with the observer's last
    `num_frames` context at action time."""

    def __init__(self, self_model: ViTOneHot, belief_encoder: BeliefEncoder):
        self.self_model = self_model
        self.belief_encoder = belief_encoder
        self.memory = TransformerBuffer(
            capacity=EPISODE_LEN + 10,
            obs_shape=(int(np.prod(ARCH["state_size"])),),
            n_frames=ARCH["num_frames"],
        )
        self.epsilon = 0.0
        self.n_frames = ARCH["num_frames"]

    def reset(self):
        self.memory.idx = 0
        self.memory.size = 0

    def start_epoch_action(self, **kw):
        pass

    def end_epoch_action(self, **kw):
        pass

    def train_step(self):
        return 0.0

    def epsilon_decay(self, *a, **kw):
        pass

    def save(self, **kw):
        pass

    def eval(self):
        pass


class ObserverAgent(MovingAgent[TreasurehuntWorld]):
    """Belief-augmented observer. Holds references to the watched agents and
    maintains a FOV-gated 5-frame window of each one's first-person POV +
    action."""

    # Observer operating modes:
    #   "belief"       - encoded g modulates pass-1 reconstruction (current belief_on=True)
    #   "imagine"      - no g; base's unconditional reconstruction still fills the
    #                    outer ring (this is what belief_on=False actually did)
    #   "masked"       - no substitution at all; pass 2 acts on the outer-ring-masked
    #                    view (mask channel = 1) — the true "no belief" baseline,
    #                    in-distribution for the Phase-1 base
    #   "oracle_world" - pov() is NOT masked; the observer sees the true outer ring.
    #                    Ceiling on how much any belief could ever help.
    #   "oracle_g"     - like "belief" but injects a precomputed class-prototype g
    #                    (ground-truth preference) instead of the online-encoded g.
    #                    Isolates decode-path quality from inference-path quality.
    MODES = ("belief", "imagine", "masked", "oracle_world", "oracle_g")

    def __init__(
        self,
        observation_spec: OneHotObservationSpec,
        action_spec: ActionSpec,
        self_model: ViTOneHot,
        belief_encoder: BeliefEncoder,
        watched_agents: list[TreasurehuntAgent],
        belief_on: bool = True,
        mode: str | None = None,
        g_override: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__(observation_spec, action_spec, _ObserverModelStub(self_model, belief_encoder))
        self.mode = mode if mode is not None else ("belief" if belief_on else "imagine")
        if self.mode not in self.MODES:
            raise ValueError(f"Unknown observer mode: {self.mode!r}")
        if self.mode == "oracle_g" and not g_override:
            raise ValueError("mode='oracle_g' requires g_override={'gemlover': g, 'foodlover': g}")
        self.g_override = g_override or {}
        # Appear as a TreasurehuntAgent in the observation spec's entity_map so
        # the observer is rendered into observations with the agent one-hot.
        self.kind = "TreasurehuntAgent"
        # Observer's OWN preference. The frozen base was trained on a gem-lover,
        # so we mirror that here so `act()` reports preference-weighted reward
        # (gems are worth more than food to a gem-lover). This is what the
        # experiment measures: did the observer end up where IT wanted to be?
        self.preferences: dict[str, float] = {"Gem": 1.0, "Food": 0.2, "Bone": 1.0}
        self.self_model = self_model
        self.belief_encoder = belief_encoder
        # g is only gathered in the two g-driven modes.
        self.belief_on = self.mode in ("belief", "oracle_g")
        self.watched_agents = watched_agents
        # Per-watched-agent rolling windows of (pov, action) -- ONLY frames
        # when that agent was inside the observer's FOV.
        self._windows: dict[int, deque] = {
            id(a): deque(maxlen=ARCH["num_frames"]) for a in watched_agents
        }
        # Diagnostics for the visualization panel.
        self.inferences: dict[str, float] = {}
        self.last_action_name: str = "—"
        self.last_belief_recon: dict[str, float] | None = None
        # Per-episode entity-hit counters (reset by reset()).
        self.collected: dict[str, int] = {"Gem": 0, "Food": 0, "Bone": 0}
        # Cached internals for visualization (populated each get_action call).
        self.last_masked_pov: np.ndarray | None = None
        self.last_completed_pov: np.ndarray | None = None
        self.last_pos_probs: np.ndarray | None = None
        self.last_belief_was_used: bool = False

    # --- pov / FOV utilities --------------------------------------------------

    def pov(self, world: TreasurehuntWorld) -> np.ndarray:
        """7-channel POV: 6 entity channels with outer ring zeroed, plus a 7th
        mask channel set to 1 in the outer ring (so masked cells are
        distinguishable from EmptyEntity, which is also all-zero across the
        entity channels).

        In "oracle_world" mode nothing is masked: the observer sees the true
        outer ring (mask channel = 0 everywhere). This is the information
        ceiling for any belief mechanism.
        """
        full_6ch = self.observation_spec.observe(world, self.location)  # (6, 15, 15) np
        full_t = torch.from_numpy(full_6ch.astype(np.float32))           # to torch
        ring = torch.zeros_like(OUTER_RING_2D) if self.mode == "oracle_world" else OUTER_RING_2D
        seven = add_mask_channel(full_t, ring)                           # (7, 15, 15)
        return seven.numpy().reshape(1, -1)

    def _in_fov(self, other: TreasurehuntAgent) -> bool:
        """Is `other` inside the observer's ACTUAL FOV (inner 9x9 of the 15x15
        input)? Uses ACTUAL_FOV_RADIUS, not the input radius."""
        oy, ox, _ = self.location
        py, px, _ = other.location
        return abs(py - oy) <= ACTUAL_FOV_RADIUS and abs(px - ox) <= ACTUAL_FOV_RADIUS

    # --- belief inference -----------------------------------------------------

    def _record_visible_agents(self, world: TreasurehuntWorld) -> None:
        """FOV-GATED FIRST-PERSON: when a watched agent is in the observer's
        inner FOV, append (watched agent's OWN first-person POV, watched
        agent's action this turn) to that agent's window. The observer can
        only access the agent's perspective WHILE it can see them.

        The watched agent's POV is 6-channel (their full first-person view).
        We pad it with a zero mask channel here so encoder/base operate on
        consistent 7-channel inputs.
        """
        entity_shape = (NUM_ENTITY_CHANNELS, ARCH["state_size"][1], ARCH["state_size"][2])
        for other in self.watched_agents:
            if not self._in_fov(other):
                continue
            mem = other.model.memory
            if mem.size < 1:
                continue
            idx = (mem.idx - 1) % mem.capacity
            pov_6ch = mem.states[idx].reshape(entity_shape)  # (6, 15, 15)
            pov_7ch = add_zero_mask_channel(
                torch.from_numpy(pov_6ch.astype(np.float32))
            ).numpy()
            act = int(mem.actions[idx])
            self._windows[id(other)].append((pov_7ch, act))

    def _encode_g(self, window: deque) -> torch.Tensor | None:
        """Encode g from a full 5-frame window, or return None if not full."""
        if len(window) < ARCH["num_frames"]:
            return None
        povs = np.stack([x[0] for x in window], axis=0)  # (T, C, H, W)
        acts = np.array([x[1] for x in window], dtype=np.int64).reshape(-1, 1)
        s = torch.tensor(povs, dtype=torch.float32).unsqueeze(0)  # (1, T, C, H, W)
        a = torch.tensor(acts).unsqueeze(0)  # (1, T, 1)
        with torch.no_grad():
            return self.belief_encoder(s, a)  # (1, layer_size)

    # --- action selection -----------------------------------------------------

    def _self_history(self, current_state_flat: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the observer's own (states, actions) sequence of length
        num_frames ending in the current observation. Falls back to zero
        padding when not enough memory yet."""
        T = ARCH["num_frames"]
        C, H, W = ARCH["state_size"]
        mem = self.model.memory
        states = np.zeros((T, C, H, W), dtype=np.float32)
        actions = np.zeros((T, 1), dtype=np.int64)
        # Fill T-1 frames from memory, last frame from current pov
        avail = min(mem.size, T - 1)
        if avail > 0:
            start = (mem.idx - avail) % mem.capacity
            for k in range(avail):
                i = (start + k) % mem.capacity
                states[T - 1 - avail + k] = mem.states[i].reshape(C, H, W)
                actions[T - 1 - avail + k, 0] = mem.actions[i]
        states[-1] = current_state_flat.reshape(C, H, W)
        # last action: take the latest stored action (placeholder until we act)
        return (
            torch.tensor(states).unsqueeze(0),
            torch.tensor(actions).unsqueeze(0),
        )

    def get_action(self, state: np.ndarray) -> int:
        """Two-pass action selection with PER-AGENT prediction averaging.

        Pass 1 (VISION): for each watched agent with a full window, run the
          frozen base + that agent's g separately; average the per-channel
          positive-class probabilities across watched agents; argmax over
          entity channels -> reconstructed outer ring.

          This is mathematically distinct from "average the g embeddings then
          run once" -- if g_gem says 'gem in the outer ring' and g_food says
          'food in the outer ring', averaging the EMBEDDINGS lands in a
          neither-here-nor-there region of g-space and the reconstruction is
          ambiguous; averaging the PREDICTIONS produces a 50/50 gem-vs-food
          posterior, which is the correct belief.

        Pass 2 (ACTION): frozen base runs on the belief-completed POV WITHOUT
          g, mask channel = 0 -- the base's own preference (gem-lover) drives
          the action choice.
        """
        s_seq, a_seq = self._self_history(state)
        C, H, W = ARCH["state_size"]

        gs = []
        self.inferences = {}
        if self.belief_on:
            for other in self.watched_agents:
                g = self._encode_g(self._windows[id(other)])
                if g is not None:
                    label = (
                        "gemlover"
                        if other.preferences.get("Gem", 0) > 0.5
                        else "foodlover"
                    )
                    if self.mode == "oracle_g":
                        # Ground-truth preference: swap in the class prototype.
                        # Same FOV gating (only when a full window exists), so
                        # the only difference vs "belief" is g quality.
                        g = self.g_override[label]
                    gs.append(g)
                    self.inferences[f"g({label})_norm"] = float(g.norm().item())

        with torch.no_grad():
            if self.mode in ("masked", "oracle_world"):
                # No pass-1 substitution: act directly on the observed view
                # (masked view stays masked; oracle view is already complete).
                _, action_logits = self.self_model.forward(
                    s_seq, a_seq, belief_embedding=None
                )
                probs = F.softmax(action_logits[0, -1], dim=-1)
                action = int(probs.argmax().item())
                self.last_action_name = ACTION_NAMES[action]
                self.last_belief_recon = {
                    a: float(probs[i].item()) for i, a in enumerate(ACTION_NAMES)
                }
                self.last_masked_pov = s_seq[0, -1].cpu().numpy()
                self.last_completed_pov = s_seq[0, -1].cpu().numpy()
                self.last_pos_probs = None
                self.last_belief_was_used = False
                return action

            # ---- PASS 1: vision (per-agent reconstruction, then average) ----
            if gs:
                pos_per_agent = []
                for g in gs:
                    preds, _ = self.self_model.forward(s_seq, a_seq, belief_embedding=g)
                    # preds: (1, T, 2, H, W, C=7). Softmax over neg/pos dim ->
                    # take positive-class probability per channel.
                    pos_per_agent.append(F.softmax(preds, dim=2)[:, :, 1])
                pos = torch.stack(pos_per_agent, dim=0).mean(dim=0)  # (1, T, H, W, C)
            else:
                # No belief (belief_on=False or no full window): run with belief=None.
                preds, _ = self.self_model.forward(s_seq, a_seq, belief_embedding=None)
                pos = F.softmax(preds, dim=2)[:, :, 1]

            # Argmax over entity channels (0..5), not the mask channel.
            entity_pos = pos[..., :NUM_ENTITY_CHANNELS]       # (1, T, H, W, 6)
            ch = entity_pos.argmax(dim=-1)                    # (1, T, H, W)
            recon_6 = F.one_hot(ch, num_classes=NUM_ENTITY_CHANNELS).permute(0, 1, 4, 2, 3).float()
            recon_mask_ch = torch.zeros(1, ARCH["num_frames"], 1, H, W)
            recon = torch.cat([recon_6, recon_mask_ch], dim=2)  # (1, T, 7, H, W)

            inner = 1.0 - OUTER_RING_2D                                   # (H, W)
            visible = inner.view(1, 1, 1, H, W).expand(1, ARCH["num_frames"], C, H, W)
            completed = s_seq * visible + recon * (1 - visible)
            completed[:, :, MASK_CHANNEL] = 0.0  # belief-completed view is "known"

            # ---- PASS 2: action ---------------------------------------------
            _, action_logits = self.self_model.forward(
                completed, a_seq, belief_embedding=None
            )
            probs = F.softmax(action_logits[0, -1], dim=-1)
            action = int(probs.argmax().item())
            self.last_action_name = ACTION_NAMES[action]

            hidden_mask = (1 - inner).bool()
            last_pos = pos[0, -1]  # (H, W, C)
            self.last_belief_recon = {
                "gem_in_ring": float(last_pos[..., GEM_CH][hidden_mask].mean().item()),
                "food_in_ring": float(last_pos[..., FOOD_CH][hidden_mask].mean().item()),
                **{a: float(probs[i].item()) for i, a in enumerate(ACTION_NAMES)},
            }
            self.last_masked_pov = s_seq[0, -1].cpu().numpy()
            self.last_completed_pov = completed[0, -1].cpu().numpy()
            self.last_pos_probs = last_pos.cpu().numpy()
            self.last_belief_was_used = bool(gs)
        return action

    # --- standard Agent overrides --------------------------------------------

    def reset(self) -> None:
        self.model.reset()
        for w in self._windows.values():
            w.clear()
        self.collected = {"Gem": 0, "Food": 0, "Bone": 0}

    def is_done(self, world: TreasurehuntWorld) -> bool:
        return world.is_done

    def act(self, world: TreasurehuntWorld, action: int) -> float:
        """Move + collect; preference-weighted reward (mirrors TreasurehuntAgent).
        Also counts which entity types the observer hits per episode -- this is
        the experiment's behavioral DV."""
        new_location = self.movement(action)
        target = world.observe(new_location)
        if target.kind in self.collected and target.value != 0:
            self.collected[target.kind] += 1
        reward = self.preferences.get(target.kind, 1.0) * target.value
        world.move(self, new_location)
        return reward

    def transition(self, world: TreasurehuntWorld) -> None:
        # Standard transition + record visible watched agents BEFORE we pick
        # our own action, so this turn's inference uses up-to-date windows.
        self._record_visible_agents(world)
        super().transition(world)


# ==========================================================================
# Custom env with deterministic spawn positions
# ==========================================================================


class ToMRolloutEnv(TreasurehuntEnv):
    """Same as TreasurehuntEnv but places agents at chosen positions for a
    clean demo, and lets us inject the Observer alongside the IQN agents."""

    def __init__(
        self,
        world,
        config,
        observer_spawn: tuple,
        watched_spawns: list[tuple],
        observer: ObserverAgent,
    ):
        # Pre-store these because super().__init__ calls populate_environment.
        self._observer = observer
        self._observer_spawn = observer_spawn
        self._watched_spawns = watched_spawns
        super().__init__(world, config)

    def setup_agents(self):
        super().setup_agents()  # creates the two IQN agents
        # Append the observer at the end (kept distinct from IQN agents).
        self.agents = list(self.agents) + [self._observer]

    def populate_environment(self):
        # Fill walls + sand as in the parent, but spawn agents at chosen spots.
        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (
                y in [0, self.world.height - 1] or x in [0, self.world.width - 1]
            ) and z == 1:
                self.world.add(index, Wall())
            elif z == 0:
                self.world.add(index, Sand())
        # Watched agents at chosen positions, observer at its position.
        watched_agents = self.agents[:-1]  # all but the observer
        observer = self.agents[-1]
        for agent, loc in zip(watched_agents, self._watched_spawns):
            self.world.add(tuple(loc), agent)
        self.world.add(tuple(self._observer_spawn), observer)


# ==========================================================================
# Main
# ==========================================================================


def build_env(
    watched_config: str = WATCHED_CONFIG,
    belief_on: bool = BELIEF_ON,
    seed: int | None = None,
    mode: str | None = None,
    g_override: dict[str, torch.Tensor] | None = None,
) -> tuple["ToMRolloutEnv", "ObserverAgent"]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    config = _make_config(watched_config)
    # 1. Load frozen self-model + belief encoder.
    base = ViTOneHot(
        memory=TransformerBuffer(capacity=10, obs_shape=(int(np.prod(ARCH["state_size"])),), n_frames=ARCH["num_frames"]),
        LR=1e-3,
        device="cpu",
        seed=0,
        reward_loss_weight=1.0,
        **ARCH,
    )
    base.load(str(SELF_MODEL_PATH))
    # The belief encoder lives inside a BeliefModel state_dict; rebuild and load.
    belief_encoder = BeliefEncoder(
        state_size=ARCH["state_size"],
        action_space=ARCH["action_space"],
        layer_size=ARCH["layer_size"],
        patch_size=ARCH["patch_size"],
        num_frames=ARCH["num_frames"],
        num_heads=ARCH["num_heads"],
        num_layers=ARCH["num_layers"],
        device="cpu",
    )
    ckpt = torch.load(BELIEF_PATH, map_location="cpu")
    belief_encoder.load_state_dict(ckpt["belief_encoder"])
    belief_encoder.eval()

    # 2. Observation/action specs for the observer.
    obs_spec = OneHotObservationSpec(
        ENTITY_LIST,
        full_view=False,
        vision_radius=config.model.agent_vision_radius,
    )
    obs_spec.override_input_size((int(np.prod(obs_spec.input_size)),))
    act_spec = ActionSpec(ACTION_NAMES)

    # 3. Observer (watched_agents set after env is built).
    observer = ObserverAgent(
        observation_spec=obs_spec,
        action_spec=act_spec,
        self_model=base,
        belief_encoder=belief_encoder,
        watched_agents=[],
        belief_on=belief_on,
        mode=mode,
        g_override=g_override,
    )

    # 4. Deterministic spawn positions. For watched agents, choose positions
    # such that each watched agent must walk THROUGH the observer's FOV to
    # reach their preferred region -- so the observer can see them heading
    # toward a region beyond its own view.
    #   gem-pref watched -> spawn east of observer (x=13); walks west into the
    #                       hidden gem region beyond observer's FOV.
    #   food-pref watched -> spawn west of observer (x=7); walks east into the
    #                        hidden food region beyond observer's FOV.
    observer_spawn = (10, 10, 1)
    watched_spawns: list[tuple] = []
    watched_iqn_paths: list[Path] = []
    for pref in config.model.preferences:
        if pref.get("Gem", 0) > 0.5:
            watched_spawns.append((10, 13, 1))
            watched_iqn_paths.append(GEM_IQN_PATH)
        else:
            watched_spawns.append((10, 7, 1))
            watched_iqn_paths.append(FOOD_IQN_PATH)

    # 5. Build the real env with observer injected.
    world = TreasurehuntWorld(config=config, default_entity=EmptyEntity())
    env = ToMRolloutEnv(
        world=world,
        config=config,
        observer_spawn=observer_spawn,
        watched_spawns=watched_spawns,
        observer=observer,
    )
    # Load IQN policies into the watched agents (in preference order).
    n_watched = len(config.model.preferences)
    for i in range(n_watched):
        env.agents[i].model.load(file_path=str(watched_iqn_paths[i]))
        env.agents[i].model.epsilon = 0.0
    # Point the observer's watched_agents at the env's IQN agents.
    observer.watched_agents = env.agents[:n_watched]
    observer._windows = {
        id(a): deque(maxlen=ARCH["num_frames"]) for a in observer.watched_agents
    }
    return env, observer


def main():
    env, observer = build_env()
    renderer = ImageRenderer(
        experiment_name=f"TomRollout_{WATCHED_CONFIG}_{'belief' if BELIEF_ON else 'nobelief'}",
        record_period=1,
        num_turns=EPISODE_LEN,
    )
    gif_dir = DATA_DIR / "gifs"

    print(
        f"Rollout: watched={WATCHED_CONFIG}, belief_on={BELIEF_ON}, "
        f"episode_len={EPISODE_LEN}"
    )
    print(
        f"  Observer at {observer.location}, watched at "
        f"{[a.location for a in observer.watched_agents]}"
    )
    for turn in range(EPISODE_LEN):
        env.take_turn()
        renderer.add_image(env.world)
        in_fov = {
            ("gemlover" if a.preferences.get("Gem", 0) > 0.5 else "foodlover"): observer._in_fov(a)
            for a in observer.watched_agents
        }
        print(
            f"  turn {turn:>3d}: obs={observer.location[:2]}  action={observer.last_action_name:<5s} "
            f"in_FOV={in_fov}  "
            f"g_norm={ {k: round(v, 2) for k, v in observer.inferences.items()} }  "
            f"action_probs={ {k: round(v, 3) for k, v in (observer.last_belief_recon or {}).items()} }"
        )
        if env.world.is_done:
            break

    renderer.save_gif(epoch=0, folder=gif_dir)
    print(
        f"GIF saved under: {gif_dir}/TomRollout_{WATCHED_CONFIG}_"
        f"{'belief' if BELIEF_ON else 'nobelief'}_epoch0.gif"
    )


if __name__ == "__main__":
    main()
