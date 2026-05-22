"""The agent for treasurehunt, a simple example for the purpose of a tutorial."""

# begin imports
from pathlib import Path

import numpy as np

from sorrel.agents import Agent, MovingAgent
from sorrel.examples.treasurehunt.world import TreasurehuntWorld

# end imports


# begin treasurehunt agent
class TreasurehuntAgent(MovingAgent[TreasurehuntWorld]):
    """A treasurehunt agent that uses the iqn model."""

    def __init__(self, observation_spec, action_spec, model, preferences=None):
        super().__init__(observation_spec, action_spec, model)
        self.sprite = Path(__file__).parent / "./assets/hero.png"
        # Per-agent reward preferences over entity kinds. Maps an entity's `kind`
        # (its class name, e.g. "Gem"/"Food"/"Bone") to a multiplier applied to
        # that entity's value when this agent collects it. A missing kind defaults
        # to 1.0, so an empty/None preference reproduces the original uniform reward.
        # Heterogeneous preferences are what make an agent's goal inferable from
        # behavior (the Theory of Mind signal).
        self.preferences: dict[str, float] = preferences or {}

    # end constructor

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        self.model.reset()

    def pov(self, world: TreasurehuntWorld) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(world, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state()
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, world: TreasurehuntWorld, action: int) -> float:
        """Act on the environment, returning the reward."""

        # Translate the model output to an action string
        action_name = self.action_spec.get_readable_action(action)

        new_location = self.location
        if action_name == "up":
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action_name == "down":
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action_name == "left":
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action_name == "right":
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location, scaled by this agent's
        # preference for that entity kind (default 1.0 = original behavior).
        target_object = world.observe(new_location)
        reward = self.preferences.get(target_object.kind, 1.0) * target_object.value

        # try moving to new_location
        world.move(self, new_location)

        return reward

    def is_done(self, world: TreasurehuntWorld) -> bool:
        """Returns whether this Agent is done."""
        return world.is_done
