"""The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports

from omegaconf import DictConfig, OmegaConf

from sorrel.worlds import Gridworld

# end imports


# begin treasurehunt
class TreasurehuntWorld(Gridworld):
    """Treasurehunt world."""

    def __init__(self, config: dict | DictConfig, default_entity):
        layers = 2
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        super().__init__(
            config.world.height, config.world.width, layers, default_entity
        )

        self.values = {
            "gem": config.world.gem_value,
            "food": config.world.food_value,
            "bone": config.world.bone_value,
        }
        self.spawn_prob = config.world.spawn_prob
        # Theory-of-Mind setup: when `segregate` is True, gems spawn only in the
        # left half of the world and food only in the right half, so an agent's
        # preference determines its direction of travel (making the goal inferable
        # from behavior). Bones spawn everywhere (goal-neutral). Default False
        # preserves the original uniform spawning.
        self.segregate = bool(config.world.get("segregate", False))
        self.bone_fraction = float(config.world.get("bone_fraction", 0.2))


# end treasurehunt


class TwoRoomsWorld(TreasurehuntWorld):
    """Two rooms + center corridor, with a per-episode hidden latent.

    Layout (default 20x20)::

        x:  0 12345 6......13 14 15..18 19
            # |room|  corridor  | |room| #     (# border, | partition wall)

    Two vertical partition walls (default x=5 and x=14) split the world into a
    left room, a center corridor, and a right room. Each partition has a
    2-cell door (default y=9,10).

    The Theory-of-Mind latent: each episode, ``gem_side`` is sampled uniformly
    from {"left", "right"}. All gems are placed in the gem-side room, all food
    in the other room, and an EQUAL number of bones in both rooms
    (goal-neutral). Items are persistent — placed once per episode with no
    respawn churn (set ``spawn_prob: 0.0``) — so a watched agent's direction
    of travel through a door carries the full information about the latent.

    Set ``forced_gem_side`` to pin the latent (for controlled experiments);
    ``None`` restores per-episode sampling.
    """

    def __init__(self, config: dict | DictConfig, default_entity):
        super().__init__(config, default_entity)
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        w = config.world
        # Walls 7 apart so that an observer with a 9x9 FOV standing at a
        # center-corridor cell (x in {9, 10}) sees BOTH partition walls and
        # their doors, while the room item columns stay outside its view.
        self.left_wall_x = int(w.get("left_wall_x", 6))
        self.right_wall_x = int(w.get("right_wall_x", 13))
        # 4-cell doors: narrow (1-2 cell) doors made random exploration from the
        # corridor too unlikely to ever discover the rooms (reward plateaued at 0).
        self.door_ys = tuple(w.get("door_ys", [8, 9, 10, 11]))
        # Wall contact penalty. The global Wall entity has value -1, but here the
        # doors are embedded in walls: a bump penalty teaches agents to avoid the
        # partitions and therefore the doors. Default 0 in this world.
        self.wall_value = float(w.get("wall_value", 0.0))
        # Where agents spawn each episode: "anywhere" (corridor + rooms; needed
        # during Phase 0 training so agents actually experience item rewards) or
        # "corridor" (deployment/ToM rollouts: nobody starts with room knowledge).
        self.spawn_region = str(w.get("spawn_region", "anywhere"))
        self.n_gems = int(w.get("n_gems", 8))
        self.n_food = int(w.get("n_food", 8))
        self.n_bones_per_room = int(w.get("n_bones_per_room", 2))
        # Cluster items in the door-adjacent part of each room (inner two
        # columns, vertical band around the doors). With items scattered over
        # the whole room, "which room has gems" is only partially observable
        # from the corridor and the IQNs plateaued at chance-level door choice;
        # clustered items make the room contents directly visible from
        # mid-corridor, so the correct door is a reactive perceptual decision.
        self.cluster_items = bool(w.get("cluster_items", True))
        # Corridor "preview" items of BOTH kinds near the observer spawn: a
        # watched agent picking through them reveals its DESIRE inside the
        # observer's FOV, before its door choice reveals the latent.
        self.n_preview_gems = int(w.get("n_preview_gems", 2))
        self.n_preview_food = int(w.get("n_preview_food", 2))
        # Per-episode latent; sampled in TwoRoomsEnv.populate_environment().
        self.gem_side: str | None = None
        self.forced_gem_side: str | None = None

    def room_cells(self, side: str) -> list[tuple[int, int, int]]:
        """Agent-layer cells strictly inside the left or right room."""
        if side == "left":
            xs = range(1, self.left_wall_x)
        else:
            xs = range(self.right_wall_x + 1, self.width - 1)
        return [(y, x, 1) for y in range(1, self.height - 1) for x in xs]

    def item_cells(self, side: str) -> list[tuple[int, int, int]]:
        """Cells where items may be placed in the given room.

        With ``cluster_items`` (default): the two room columns nearest the
        corridor, within a vertical band spanning the doors ± 4 rows —
        all simultaneously visible in a 15x15 view from mid-corridor.
        """
        if not self.cluster_items:
            return self.room_cells(side)
        y_lo = max(1, min(self.door_ys) - 4)
        y_hi = min(self.height - 2, max(self.door_ys) + 4)
        # One empty column between the wall and the item columns, so the items
        # sit just beyond the reach of a 9x9 view from the corridor center.
        if side == "left":
            xs = range(self.left_wall_x - 3, self.left_wall_x - 1)
        else:
            xs = range(self.right_wall_x + 2, self.right_wall_x + 4)
        return [(y, x, 1) for y in range(y_lo, y_hi + 1) for x in xs]

    def corridor_cells(self) -> list[tuple[int, int, int]]:
        """Agent-layer cells strictly inside the center corridor."""
        return [
            (y, x, 1)
            for y in range(1, self.height - 1)
            for x in range(self.left_wall_x + 1, self.right_wall_x)
        ]

    def observer_spawn_cells(self) -> list[tuple[int, int, int]]:
        """Center-corridor cells from which a 9x9 FOV covers both partition
        walls (and doors) but neither room's item columns."""
        return [(y, x, 1) for y in (9, 10) for x in (9, 10)]

    def preview_cells(self) -> list[tuple[int, int, int]]:
        """Corridor cells visible from every observer spawn cell (the FOV
        intersection), where the desire-revealing preview items go."""
        return [
            (y, x, 1)
            for y in range(6, 14)
            for x in range(self.left_wall_x + 1, self.right_wall_x)
            if (y, x, 1) not in self.observer_spawn_cells()
        ]


class QuadrantWorld(TreasurehuntWorld):
    """Center chamber + up to four edge rooms, with a K-way hidden latent.

    ToM v3 generalization of `TwoRoomsWorld`. Walls at x in {6, 13} and
    y in {6, 13} partition the 20x20 grid into a 3x3 pattern::

        corner |  north room | corner
        -------+-------------+-------
        west   |   center    | east
        room   |   chamber   | room
        -------+-------------+-------
        corner |  south room | corner

    The four corner regions are sealed (wall-filled). The first ``n_rooms``
    (K, 2..4) of ``ROOM_ORDER`` = (west, east, north, south) get a 2-cell
    door centered on their chamber-facing wall; unused rooms stay sealed.
    Every door is within chebyshev distance 4 of every observer spawn cell,
    so all door commits happen inside a 9x9 FOV from the chamber center.

    The latent: each episode a random permutation assigns the K item kinds
    (Gem, Food, Coin, Berry, truncated to K) to the K rooms; the label is
    ``gem_room`` (index into used rooms). Items are persistent and clustered
    in a 2-row/column band just beyond each door but OUTSIDE the observer's
    FOV, plus equal bones per room (goal-neutral). Set ``forced_gem_room``
    to pin the latent; ``None`` restores per-episode sampling.
    """

    ROOM_ORDER = ("west", "east", "north", "south")
    ITEM_KINDS = ("Gem", "Food", "Coin", "Berry")

    def __init__(self, config: dict | DictConfig, default_entity):
        super().__init__(config, default_entity)
        if type(config) != DictConfig:
            config = OmegaConf.create(config)
        w = config.world
        self.n_rooms = int(w.get("n_rooms", 4))
        assert 2 <= self.n_rooms <= 4
        # Wall coordinates (low/high apply to both axes by symmetry).
        self.wall_lo = 6
        self.wall_hi = 13
        self.n_main_items_per_room = int(w.get("n_main_items_per_room", 6))
        self.n_bones_per_room = int(w.get("n_bones_per_room", 2))
        self.n_preview_per_kind = int(w.get("n_preview_per_kind", 2))
        # Per-episode latent; sampled in QuadrantEnv.populate_environment().
        self.gem_room: int | None = None  # index into used_rooms()
        self.kind_by_room: dict[str, str] = {}
        self.forced_gem_room: int | None = None

    def used_rooms(self) -> list[str]:
        return list(self.ROOM_ORDER[: self.n_rooms])

    def used_kinds(self) -> list[str]:
        return list(self.ITEM_KINDS[: self.n_rooms])

    def _room_ranges(self, room: str) -> tuple[range, range]:
        """(y-range, x-range) of the cells strictly inside a room."""
        low, mid, high = range(1, 6), range(7, 13), range(14, 19)
        return {
            "west": (mid, low),
            "east": (mid, high),
            "north": (low, mid),
            "south": (high, mid),
        }[room]

    def room_cells(self, room: str) -> list[tuple[int, int, int]]:
        ys, xs = self._room_ranges(room)
        return [(y, x, 1) for y in ys for x in xs]

    def door_cells(self, room: str) -> list[tuple[int, int, int]]:
        """The 2-cell door in the room's chamber-facing wall."""
        return {
            "west": [(9, self.wall_lo, 1), (10, self.wall_lo, 1)],
            "east": [(9, self.wall_hi, 1), (10, self.wall_hi, 1)],
            "north": [(self.wall_lo, 9, 1), (self.wall_lo, 10, 1)],
            "south": [(self.wall_hi, 9, 1), (self.wall_hi, 10, 1)],
        }[room]

    def item_cells(self, room: str) -> list[tuple[int, int, int]]:
        """A 2-deep band across the room, just beyond the observer's FOV.

        Observer spawn cells are (9|10, 9|10) with a 9x9 FOV, so coordinates
        <= 4 or >= 15 are never visible; the bands sit exactly there, two to
        three cells behind each door.
        """
        band = {"west": (3, 4), "east": (15, 16), "north": (3, 4), "south": (15, 16)}[
            room
        ]
        if room in ("west", "east"):
            return [(y, x, 1) for y in range(7, 13) for x in band]
        return [(y, x, 1) for y in band for x in range(7, 13)]

    def chamber_cells(self) -> list[tuple[int, int, int]]:
        """Agent-layer cells strictly inside the center chamber (all of them
        visible from every observer spawn cell)."""
        return [(y, x, 1) for y in range(7, 13) for x in range(7, 13)]

    def observer_spawn_cells(self) -> list[tuple[int, int, int]]:
        return [(y, x, 1) for y in (9, 10) for x in (9, 10)]

    def preview_cells(self) -> list[tuple[int, int, int]]:
        spawns = set(self.observer_spawn_cells())
        return [c for c in self.chamber_cells() if c not in spawns]

    def corner_cells(self) -> list[tuple[int, int, int]]:
        """Sealed corner-region cells (wall-filled at populate time)."""
        low, high = range(1, 6), range(14, 19)
        return [
            (y, x, 1)
            for ys in (low, high)
            for xs in (low, high)
            for y in ys
            for x in xs
        ]

    def room_of(self, loc) -> str:
        """Which region a location is in: a room name or "chamber" (walls,
        doors, and corners count as chamber — they are never commits)."""
        y, x = loc[0], loc[1]
        for room in self.used_rooms():
            ys, xs = self._room_ranges(room)
            if y in ys and x in xs:
                return room
        return "chamber"
