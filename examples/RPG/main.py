# ------------------------ #
# region: Imports          #
import sys
import os
from datetime import datetime

# ------------------------ #
# region: path nonsense    #
# Determine appropriate paths for imports and storage
current_path = os.path.abspath('.').split('/')
# If we are already in the base directory...
if current_path[-1] == 'transformers':
    root = os.path.abspath('.')
# From the RPG directory, we need to go up two folders
elif current_path[-1] == 'RPG':
    root = os.path.abspath('../..')
# Base case: specify the folder manually
else:
    root = os.path.abspath('/Users/nathanbaek/Documents/transformer') # Change the wd as needed.
    # '/Users/rgelpi/Documents/GitHub/transformers'

# Make sure the transformers directory is in PYTHONPATH
if root not in sys.path:
    sys.path.insert(0, root)
# endregion                #
# ------------------------ #

from examples.RPG.utils import (
    init_log, parse_args, load_config,
    create_models,
    create_agents,
    create_entities
)

from examples.trucks.utils import GameVars

from examples.RPG.env import RPG
from examples.RPG.agents import Agent
import random

# endregion                #
# ------------------------ #

def run(cfg, **kwargs):
    # Initialize the environment and get the agents
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = RPG(cfg, agents, entities)

    # Set up tensorboard logging
    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            log_dir=f'{root}/examples/RPG/runs/{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )

    # Container for game variables (epoch, turn, loss, reward)
    game_vars = GameVars(cfg.experiment.epochs)

    # If a path to a model is specified in the run, load those weights
    if 'load_weights' in kwargs:
        for agent in agents:
            agent.model.load(file_path = kwargs.get('load_weights'))

    for epoch in range(cfg.experiment.epochs):

        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.reset()
        random.shuffle(agents)

        done = 0 
        turn = 0
        losses = 0
        game_points = 0

        while not done:

            turn = turn + 1

            for agent in agents:
                agent.model.start_epoch_action(**locals())

            for agent in agents:
                agent.reward = 0

            entities = env.get_entities_for_transition()
            # Entity transition
            for entity in entities:
                entity.transition(env)

            # Agent transition
            for agent in agents:

                (state,
                action,
                reward,
                next_state,
                done_
                ) = agent.transition(env)

                if turn >= cfg.experiment.max_turns or done_:
                    done = 1

                exp = (1, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                game_points += reward

                agent.model.end_epoch_action(**locals())

        # At the end of each epoch, train as long as the batch size is large enough.
        for agent in agents:
            loss = agent.model.train_model()
            losses += loss
            
            # Add the game variables to the game object
            game_vars.record_turn(epoch, turn, loss.detach().numpy(), game_points)

        # Print the variables to the console
        game_vars.pretty_print()

        # Add scalars to Tensorboard
        if cfg.log:
            writer.add_scalar('Loss', loss, epoch)
            writer.add_scalar('Reward', game_points, epoch)
            writer.add_scalar('Epsilon', agent.model.epsilon, epoch)
            writer.add_scalars(
                'Encounters',
                {
                    'Gem': agents[0].encounters['Gem'],
                    'Coin': agents[0].encounters['Coin'],
                    'Food': agents[0].encounters['Food'],
                    'Bone': agents[0].encounters['Bone'],
                    'Wall': agents[0].encounters['Wall'],
                }, epoch)

        # Special action: update epsilon
        for agent in agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)

    
    
    # Close the tensorboard log
    
    if cfg.log:
        writer.close()
    
    # If a file path has been specified, save the weights to the specified path
    if 'save_weights' in kwargs:
        for agent in agents:
            agent.model.save(file_path = kwargs.get('save_weights'))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", default='./examples/RPG/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args)
    init_log(cfg)
    run(cfg, 
        load_weights=f'{cfg.root}/examples/RPG/models/checkpoints/iRainbowModel_20230916-14091694890560.pkl',
        save_weights=f'{cfg.root}/examples/RPG/models/checkpoints/{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}.pkl')

if __name__ == '__main__':
    main()