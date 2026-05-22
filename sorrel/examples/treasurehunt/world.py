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
