# begin imports
# general imports
import numpy as np
import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment

# imports from our example
from sorrel.examples.treasurehunt.agents import TreasurehuntAgent
from sorrel.examples.treasurehunt.entities import (
    Berry,
    Bone,
    Coin,
    EmptyEntity,
    Food,
    Gem,
    Sand,
    Wall,
)
from sorrel.examples.treasurehunt.world import (
    QuadrantWorld,
    TreasurehuntWorld,
    TwoRoomsWorld,
)

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import (
    OneHotObservationSpec,
    RGBObservationSpec,
)

# end imports


# begin treasurehunt environment
class TreasurehuntEnv(Environment[TreasurehuntWorld]):
    """The experiment for treasurehunt."""

    def __init__(self, world: TreasurehuntWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        Reads agent count from config.model.num_agents (default 2).
        """
        agent_num = int(self.config.model.get("num_agents", 2))
        # Optional per-agent reward preferences. config.model.preferences, when set,
        # is a list of {entity_kind: multiplier} dicts, one per agent. Agents without
        # an entry (or when unset) fall back to uniform reward.
        preferences_list = self.config.model.get("preferences", None)
        agents = []
        for agent_idx in range(agent_num):
            preferences = None
            if preferences_list is not None and agent_idx < len(preferences_list):
                preferences = dict(preferences_list[agent_idx])
            # create the observation spec
            entity_list = [
                "EmptyEntity",
                "Wall",
                "Gem",
                "Bone",
                "Food",
                "TreasurehuntAgent",
            ]
            if hasattr(self.config.model, "observation_spec"):
                obs_spec_type = self.config.model.observation_spec
            else:
                obs_spec_type = "onehot"
            if obs_spec_type == "onehot":
                observation_spec = OneHotObservationSpec(
                    entity_list,
                    full_view=False,
                    # note that here we require self.config to have the entry model.agent_vision_radius
                    # don't forget to pass it in as part of config when creating this experiment!
                    vision_radius=self.config.model.agent_vision_radius,
                )
            elif obs_spec_type == "rgb":
                observation_spec = RGBObservationSpec(
                    entity_list,
                    full_view=False,
                    # note that here we require self.config to have the entry model.agent_vision_radius
                    # don't forget to pass it in as part of config when creating this experiment!
                    vision_radius=self.config.model.agent_vision_radius,
                )
            else:
                raise ValueError(f"Unknown observation spec type: {obs_spec_type}")

            observation_spec.override_input_size(
                (int(np.prod(observation_spec.input_size)),)
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right"])

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.6,
                device="cpu",
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )
            model.memory.extra_data["positions"] = np.zeros(
                (model.memory.capacity, 2), dtype=np.int64
            )

            agents.append(
                TreasurehuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    preferences=preferences,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the treasurehunt world by creating walls, then randomly spawning the
        agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (y in [0, self.world.height - 1] or x in [0, self.world.width - 1]) and (
                z == 1
            ):
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif (
                z == 1
            ):  # if location is on the top layer, indicate that it's possible for an agent to spawn there
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the agents
        # using np.random.choice, we choose indices in valid_spawn_locations
        agent_locations_indices = np.random.choice(
            len(valid_spawn_locations), size=len(self.agents), replace=False
        )
        agent_locations = [valid_spawn_locations[i] for i in agent_locations_indices]
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.world.add(loc, agent)


class TwoRoomsEnv(TreasurehuntEnv):
    """Treasurehunt in the two-room world (see `TwoRoomsWorld`).

    Each episode: sample the gem-side latent, build the partition walls +
    doors, place persistent items (gems in the gem room, food in the other,
    equal bones in both), and spawn all agents in the center corridor. With
    ``spawn_prob: 0.0`` nothing respawns, so the episode's item layout is
    fixed at reset time and the latent is only discoverable by entering a
    room — or by watching someone who knows head for one.
    """

    world: TwoRoomsWorld

    def populate_environment(self):
        world = self.world
        # Sample the per-episode latent (or honor a forced value).
        world.gem_side = world.forced_gem_side or (
            "left" if np.random.random() < 0.5 else "right"
        )

        # Terrain: sand layer, border walls, partition walls with doors.
        def make_wall() -> Wall:
            wall = Wall()
            wall.value = world.wall_value
            return wall

        for index in np.ndindex(world.map.shape):
            y, x, z = index
            if z == 0:
                world.add(index, Sand())
            elif z == 1 and (y in [0, world.height - 1] or x in [0, world.width - 1]):
                world.add(index, make_wall())
        for wall_x in (world.left_wall_x, world.right_wall_x):
            for y in range(1, world.height - 1):
                if y not in world.door_ys:
                    world.add((y, wall_x, 1), make_wall())

        # Persistent items. Equal bone counts per room keep bones goal-neutral.
        gem_room = world.gem_side
        food_room = "right" if gem_room == "left" else "left"

        def fill_room(side: str, n_main: int, make_main):
            cells = world.item_cells(side)
            chosen = np.random.choice(
                len(cells), size=n_main + world.n_bones_per_room, replace=False
            )
            for i in chosen[:n_main]:
                world.add(cells[i], make_main())
            for i in chosen[n_main:]:
                world.add(cells[i], Bone(world.values["bone"]))

        fill_room(gem_room, world.n_gems, lambda: Gem(world.values["gem"]))
        fill_room(food_room, world.n_food, lambda: Food(world.values["food"]))

        # Corridor preview items (both kinds, visible from the observer
        # spawn): what a watched agent picks among them reveals its desire.
        n_preview = world.n_preview_gems + world.n_preview_food
        if n_preview > 0:
            cells = world.preview_cells()
            chosen = np.random.choice(len(cells), size=n_preview, replace=False)
            for i in chosen[: world.n_preview_gems]:
                world.add(cells[i], Gem(world.values["gem"]))
            for i in chosen[world.n_preview_gems :]:
                world.add(cells[i], Food(world.values["food"]))

        # Agent spawns. "anywhere" (training): corridor + rooms, so agents
        # regularly start next to items and the value of gems/food is not
        # gated behind door discovery. "corridor" (ToM rollouts): nobody
        # starts with room knowledge.
        if world.spawn_region == "corridor":
            spawnable = world.corridor_cells()
        else:
            spawnable = (
                world.corridor_cells()
                + world.room_cells("left")
                + world.room_cells("right")
            )
        spawnable = [
            loc for loc in spawnable if world.observe(loc).kind == "EmptyEntity"
        ]
        chosen = np.random.choice(len(spawnable), size=len(self.agents), replace=False)
        for i, agent in zip(chosen, self.agents):
            world.add(spawnable[i], agent)


class QuadrantEnv(TreasurehuntEnv):
    """Treasurehunt in the K-room quadrant world (see `QuadrantWorld`).

    Each episode: sample a random assignment of the K item kinds to the K
    rooms (the `gem_room` latent), build the cross walls with doors for used
    rooms only, place persistent clustered items + equal bones per room, put
    previews of ALL K kinds in the center chamber, and spawn agents in the
    chamber. With ``spawn_prob: 0.0`` the layout is fixed at reset time.
    """

    world: QuadrantWorld

    _ITEM_CLASSES = {"Gem": Gem, "Food": Food, "Coin": Coin, "Berry": Berry}

    def _make_item(self, kind: str):
        value = self.world.values["food" if kind == "Food" else "gem"]
        return self._ITEM_CLASSES[kind](value)

    def populate_environment(self):
        world = self.world
        rooms = world.used_rooms()
        kinds = world.used_kinds()
        k = world.n_rooms

        # Sample the K-way latent: kinds[i] goes to rooms[room_idx[i]].
        # kinds[0] is always "Gem", so gem_room = room_idx[0].
        room_idx = [int(r) for r in np.random.permutation(k)]
        forced = world.forced_gem_room
        if forced is not None and room_idx[0] != forced:
            j = room_idx.index(forced)
            room_idx[0], room_idx[j] = room_idx[j], room_idx[0]
        world.gem_room = room_idx[0]
        world.kind_by_room = {rooms[room_idx[i]]: kinds[i] for i in range(k)}

        # Terrain: sand layer, border walls, cross walls with doors for the
        # used rooms, sealed corner regions.
        for index in np.ndindex(world.map.shape):
            y, x, z = index
            if z == 0:
                world.add(index, Sand())
            elif z == 1 and (y in (0, world.height - 1) or x in (0, world.width - 1)):
                world.add(index, Wall())
        doors = {c for room in rooms for c in world.door_cells(room)}
        lo, hi = world.wall_lo, world.wall_hi
        for wall_c in (lo, hi):
            for i in range(1, world.height - 1):
                if (i, wall_c, 1) not in doors:
                    world.add((i, wall_c, 1), Wall())
                if (wall_c, i, 1) not in doors:
                    world.add((wall_c, i, 1), Wall())
        for c in world.corner_cells():
            world.add(c, Wall())

        # Persistent clustered items: each used room gets its assigned kind
        # plus an equal number of bones (goal-neutral).
        n_main = world.n_main_items_per_room
        for i, kind in enumerate(kinds):
            cells = world.item_cells(rooms[room_idx[i]])
            chosen = np.random.choice(
                len(cells), size=n_main + world.n_bones_per_room, replace=False
            )
            for ci in chosen[:n_main]:
                world.add(cells[ci], self._make_item(kind))
            for ci in chosen[n_main:]:
                world.add(cells[ci], Bone(world.values["bone"]))

        # Chamber preview items of ALL K kinds: what a watched agent picks
        # among them reveals its desire, whatever that desire is.
        n_prev = world.n_preview_per_kind
        if n_prev > 0:
            cells = world.preview_cells()
            chosen = np.random.choice(len(cells), size=n_prev * k, replace=False)
            for i, kind in enumerate(kinds):
                for ci in chosen[i * n_prev : (i + 1) * n_prev]:
                    world.add(cells[ci], self._make_item(kind))

        # Agents spawn in the chamber: nobody starts with room knowledge.
        spawnable = [
            c for c in world.chamber_cells() if world.observe(c).kind == "EmptyEntity"
        ]
        chosen = np.random.choice(len(spawnable), size=len(self.agents), replace=False)
        for i, agent in zip(chosen, self.agents):
            world.add(spawnable[i], agent)
