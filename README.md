# Risk Aware RL
This project is a risk aware implementation of D4PG. The loss function for the policy was modified to include a component penalizing the standard deviation of the value function. The goal of this modification is to encourage the policy to take actions that have a consistent value over actions with a high but inconsistent value. 

This project also contains a vanilla D4PG implementation. Feel free to use it.

## Results

I experimentally compared my modified D4PG to Vanilla D4PG, PPO, and CVAR-PPO. I compare both over the learning curve during training and by varying the environment at test time and measuring how well each policy adapts to distributional shift. 

I found my modified D4PG to have the same training performance as the base D4PG algorithm. 

I found my modified D4PG to be more robust to distributional shift in 1 of the 4 MDPs I tested on, but equal in the other 3. Therefore, this idea does not work as expected. 

See the results in the /img/ folder.


I will leave the following documentation below for future reference, if needed. 



## Usage
- The training code is in the folder '/src'.
- These methods, including baselines and our methods, are based on [Spinning Up](https://github.com/openai/spinningup) 
- You first should install Spinning Up by

```
cd src
pip install -e .
```

- Then you can run the training code like

```
python -m spinup.run vpg --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/vpg/vpg-seed0 --epochs 750 --seed 0
python -m spinup.run trpo --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/trpo/trpo-seed0 --epochs 750 --seed 0
python -m spinup.run ppo --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/ppo/ppo-seed0 --epochs 750 --seed 0
python -m spinup.run pg_cmdp --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/pg_cmdp/pg_cmdp-seed0 --epochs 750 --seed 0 
python -m spinup.run cppo --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/cppo/cppo-seed0 --epochs 750 --seed 0
python -m spinup.run d4pg --hid "[64,32]" --env Walker2d-v3 --exp_name Ant/d4pg/d4pg-seed0 --epochs 750 --seed 0
```

- For PG-CMDP, you can also adjust parameters like --delay, --nu_delay and so on.

- For CPPO, you can also adjust parameters like --beta, --nu_start, --nu_delay, --delay, --cvar_clip_ratio and so on.

- For D4PG, you can also adjust parameters like --v_max and --v_min.


- Evaluate the performance under transition disturbance (changing mass)
```
python test_mass.py --task Walker2d --algos "vpg trpo ppo cvarvpg cppo d4pg" --mass_lower_bound 1.0 --mass_upper_bound 7.0 --mass_number 100 --episodes 5
```

- Evaluate the performance under observation disturbance (random noises)
```
python test_state.py --task Walker2d --algos "vpg trpo ppo pg_cmdp cppo d4pg" --epsilon_low 0.0 --epsilon_upp 0.4 --epsilon_num 100 --episodes 5
```

- Evaluate the performance under observation disturbance (adversarial noises)
```
python test_state_adversary.py --task Walker2d --algos "vpg trpo ppo pg_cmdp cppo d4pg" --epsilon_low 0.0 --epsilon_upp 0.2 --epsilon_num 100 --episodes 5
```


## Citation

This work was bootstrapped from "Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk". The addition of D4PG was my own contribution.

```
@inproceedings{
    ying2022towards,
    title={Towards Safe Reinforcement Learning via Constraining Conditional Value-at-Risk},
    author={Ying, Chengyang and Zhou, Xinning and Su, Hang and Yan, Dong and Chen, Ning and Zhu, Jun},
    booktitle={International Joint Conference on Artificial Intelligence},
    year={2022},
    url={https://arxiv.org/abs/2206.04436}
}
```
