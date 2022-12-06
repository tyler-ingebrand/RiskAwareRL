import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(xs, ys, stds, smooth=1, labels=None):
    xs = pd.concat(xs, ignore_index=True, axis=1).to_numpy().mean(axis=1)
    ys = pd.concat(ys, ignore_index=True, axis=1).to_numpy()
    stds = pd.concat(stds, ignore_index=True, axis=1).to_numpy()
    for i in range(len(labels)):
        y = ys[:, i]
        s = stds[:, i]
        if smooth > 1:
            for j in range(len(y)):
                min_index = max(0, j-smooth)
                y[j] = np.mean(y[min_index:j+1])
        plt.plot(xs, y, label=labels[i]) # , labels=["1", "2", "3", "4", "5"])
        plt.fill_between(xs, y + s, y - s, alpha=0.7)

    plt.legend(loc='upper left').set_draggable(True)
    plt.xlabel("Environment Interactions")
    plt.ylabel("Cumulative Reward")
    if True in [x > 1e5 for x in xs]:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    times = None
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1
            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data[["Performance"]])
            if times is None:
                times = exp_data[["TotalEnvInteracts"]]
    datasets = pd.concat(datasets, axis=1)
    return logdir, times, datasets.mean(axis=1), datasets.std(axis=1)


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix == x])
    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)
    # such as:
    # Plotting from ...
    # == == == == == == == == == == == == == == == == == == == == == == == == ==
    #
    # data / CartPole - v0
    #
    # == == == == == == == == == == == == == == == == == == == == == == == == ==

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    names, means, stds, times = [], [], [], []
    if legend:
        for log, leg in zip(logdirs, legend):
            name, time, m, s = get_datasets(log, leg)
            # i only care about alg, so remove everything else from log
            name = name[name.rfind('/') + 1:]
            means.append(m)
            stds.append(s)
            names.append(name)
            times.append(time)
    else:
        for log in logdirs:
            name, time, m, s = get_datasets(log)
            # i only care about alg, so remove everything else from log
            name = name[name.rfind('/')+1:]
            means.append(m)
            stds.append(s)
            names.append(name)
            times.append(time)
    return names, times, means, stds


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,  
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean'):
    names, timesteps, means, stds = get_all_datasets(all_logdirs, legend, select, exclude)
    plt.figure()
    plot_data(timesteps, means, stds, smooth=smooth, labels=names)
    plt.savefig("./1.png")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count, 
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)

if __name__ == "__main__":
    main()