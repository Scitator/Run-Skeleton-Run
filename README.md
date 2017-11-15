# Run-Skeleton-Run
[Reason8.ai](https://reason8.ai) PyTorch solution for 3rd place [NIPS RL 2017 challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run/leaderboards?challenge_round_id=12).

Additional thanks to [Michail Pavlov](https://github.com/fgvbrt) for collaboration.

## Agent policies

### no-flip-state-action

![Alt Text](http://www.sheawong.com/wp-content/uploads/2013/08/keephatin.gif)

![alt text](https://github.com/Scitator/Run-Skeleton-Run/blob/master/gifs/noflip.gif)

### flip-state-action

![alt text](https://github.com/Scitator/Run-Skeleton-Run/blob/master/gifs/flip.gif)


## How to setup environment?

1. `sh setup_conda.sh`
2. `source activate opensim-rl`

Would like to test baselines? (Need MPI support)
3. `sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev`
3+. `sh setup_env_mpi.sh`

OR like DDPG agents?
3. `sh setup_env.sh`

4. Congrats! Now you are ready to check our agents.


## Run DDPG agent

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python ddpg/train.py \
    --logdir ./logs_ddpg \
    --num-threads 4 \
    --ddpg-wrapper \
    --skip-frames 5 \
    --fail-reward -0.2 \
    --reward-scale 10 \
    --flip-state-action \
    --actor-layers 64-64 --actor-layer-norm --actor-parameters-noise \
    --actor-lr 0.001 --actor-lr-end 0.00001 \
    --critic-layers 64-32 --critic-layer-norm \
    --critic-lr 0.002 --critic-lr-end 0.00001 \
    --initial-epsilon 0.5 --final-epsilon 0.001 \
    --tau 0.0001
```


## Evaluate DDPG agent

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=./ python ddpg/submit.py \
    --restore-actor-from ./logs_ddpg/actor_state_dict.pkl \
    --restore-critic-from ./logs_ddpg/critic_state_dict.pkl \
    --restore-args-from ./logs_ddpg/args.json \
    --num-episodes 10

```


## Run TRPO/PPO agent

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python ddpg/train.py \
    --agent ppo \
    --logdir ./logs_baseline \
    --baseline-wrapper \
    --skip-frames 5 \
    --fail-reward -0.2 \
    --reward-scale 10
```
