import os
epochs = 750
env = "Humanoid-v3"
algos = ["ra_d4pg", "d4pg", "ppo", "pg_cmdp", "cppo"]
algo_extra_args = ["--v_min -0 --v_max 7000 --n_step_lookahead 5",
                    "--v_min -0 --v_max 7000 --n_step_lookahead 5",
                    "",
                    "",
                    ""]
seeds = range(10)

assert len(algos) == len(algo_extra_args)

print("Running {} algorithms for {} seeds for {} epochs each, for a total of {} runs and {} epochs".format(len(algos), len(seeds), epochs, len(algos) * len(seeds), len(algos) * len(seeds) *  epochs ))
print("Algorithms: ", algos)
print("Environment: ", env)
print("\n\n\n")

# run commands for all algos and seeds
for seed in seeds:
    for alg, args in zip(algos, algo_extra_args):
        print("Running {} with arguments {} on seed {}".format(alg, args, seed))
        command = "python -m spinup.run {} --hid \"[64,32]\" --env {} --exp_name {}/{}/seed{} --epochs {} --seed {} {}".format(alg, env, env, alg, seed, epochs, seed, args)
        os.system(command)


