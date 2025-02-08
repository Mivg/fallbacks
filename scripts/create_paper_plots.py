import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json
import operator as op
from itertools import cycle

from scipy.stats import mannwhitneyu
from tqdm import tqdm

from scripts.plotting_utils import save_to_latex

FAMILIES_WHITE_LIST = ['Llama2', 'Pythia', 'OLMo-premature', 'OLMo', 'Dolly' , 'Llama3']
CORRECT_UPPER_BOUND = {
    'answerable_qampari': 14.52,
    'hq_qampari': 6.63
}

FILTER_BAD_QUESTIONS = True
BAD_QUESTIONS = defaultdict(list, {
    "hq_qampari": [70, 73],
    "answerable_qampari": [],
    "unanswerable_qampari": [],

})

colors = ['#90BE6D', '#F8961E', '#277DA1', '#F94144'] # https://coolors.co/palette/f94144-f3722c-f8961e-f9844a-f9c74f-90be6d-43aa8b-4d908e-577590-277da1
colors += ['#5C374C', '#FAA275'] # https://coolors.co/palette/faa275-ff8c61-ce6a85-985277-5c374c
patterns = ['/', '\\', '|', 'X', '=', '-', '.']  # Patterns for correct, hallucinations, repetitions

PLOT_TOPIC_CHANGE_SEPARATELY = True  # <------- Change this if you want (/don't want) to treat all missing answers as bad
if PLOT_TOPIC_CHANGE_SEPARATELY:
    COLUMNS = ['correct', 'hallucinations', 'repetitions', 'bad_format', 'topic_change', 'EOS']
else:
    COLUMNS = ['correct', 'hallucinations', 'repetitions', 'missing']
    colors = colors[:-2]
    patterns = patterns[:-2]


def get_params_and_model(file, model_types, params, tokens=None):
    model_name = None
    try:
        model_name = re.search('.*prompts_(.*)_\d+T.*', file).group(1).replace('+', '/')
    except:
        pass
    if model_name is None:
        try:
            model_name = re.search('.*results_(.*)_.*', file).group(1).replace('+', '/')
        except:
            raise ValueError(f'Could not find model name in file {file}')

    # Parameter extraction
    match = re.search(r'(\d+(\.\d+)?)(m|b)', file, re.IGNORECASE)
    if match:
        if match.group(3).lower() == 'm':
            param = float(match.group(1)) * 1e6
        else:  # 'b'
            param = float(match.group(1)) * 1e9
        params.append(param)
    else:
        params.append(None)
    # Model type extraction
    if 'llama2' in file.lower() or 'llama-2' in file.lower():
        model_types.append("Llama2")
    elif 'llama3' in file.lower() or 'llama-3' in file.lower():
        model_types.append("Llama3")
    elif 'pythia' in file.lower() and 'deduped' in file.lower():
        model_types.append("pythia-deduped")
    elif 'pythia' in file.lower():
        model_types.append("Pythia")
    elif 'olmo' in file.lower():
        if '1bstep738020' in file.lower() or '7bstep557000' in file.lower():
            model_types.append("OLMo")
        else:
            model_types.append("OLMo-premature")

    elif 'dolly' in file.lower():
        model_types.append("Dolly")
    else:
        raise ValueError(f'Unknown model type for file {file}')

    if tokens is not None:
        if 'olmo' in model_name.lower():
            if 'instruct' in model_name.lower() or 'sft' in model_name.lower():
                tokens.append(None)
            else:
                # search for a sequence like tokens\d+B
                match = re.search(r'tokens(\d+)B', model_name, re.IGNORECASE)
                if match:
                    tokens.append(int(match.group(1)))
                else:
                    raise ValueError(f'Could not find tokens in model name {model_name}')
        elif 'pythia' in model_name.lower():
            # All models were trained for the equivalent of 143000 steps at a batch size of 2,097,152 tokens.
            # Revision/branch step143000 corresponds exactly to the model checkpoint on the main branch of each model.
            match = re.search(r'step(\d+)', model_name, re.IGNORECASE)
            if match:
                tokens.append(int(match.group(1))*2097152 / 1e9)  # always in Billions
            else:
                tokens.append(143000 * 2097152/ 1e9)
        elif 'dolly' in model_name.lower():
            tokens.append(143000 * 2097152 / 1e9)  # same as the last checkpoint in the pythia model
            params[-1] = 2.8e9 if params[-1] == 3e9 else 6.9e9 if params[-1] == 7e9 else params[-1]
        else:
            tokens.append(None)   # though we know llama 3 used 15T tokens and we know how much llama 2 used as well
    return model_name

def prepare_results_dataframe(experiment_type, results_dir, whitelist=FAMILIES_WHITE_LIST, reps=False,
                              include_temp=False, augment_greedy_from=None, slice=None):
    """Prepare a DataFrame from the results directory."""
    # Collect results
    prefix = {
        "answerable_qampari": "qampari_results",
        "unanswerable_qampari": "qampari_results",
        "hq_qampari": "hq_qampari_results",
        "open_ended": "open_ended",
    }[experiment_type]
    suffix = {
        "answerable_qampari": "answerable",
        "unanswerable_qampari": "unanswerable",
        "hq_qampari": "answerable",
        "open_ended": "answerable",
    }[experiment_type]
    n_lines = {
        "answerable_qampari": 100,
        "unanswerable_qampari": 100,
        "hq_qampari": 97,
        "open_ended": 125,
    }[experiment_type]
    bad_questions = BAD_QUESTIONS[experiment_type]
    if not os.path.isdir(results_dir):
        return []
    results = [(f, df) for f in os.listdir(results_dir) if f.startswith(prefix)
               and f.endswith('_'+suffix+'.csv')
               and len(df := pd.read_csv(os.path.join(results_dir, f))) == n_lines
               and (('reps_' not in f and not reps) or ('reps_' in f and reps))
               and 'llama-2-13b' not in f.lower()]   # exclude llama 2 13b
    if len(results) == 0:
        return []
    if slice is not None:
        results = [(k, v.iloc[slice[0]:slice[1]]) for k, v in results]
    if len(bad_questions) > 0 and FILTER_BAD_QUESTIONS:
        assert slice is None, "Can't filter bad questions with a slice"
        print(f'Filtering bad questions for {experiment_type}')
        results = [(k, v[~v.index.isin(bad_questions)].reset_index(drop=True)) for k, v in results]

    if reps:
        # there might be cases where we have old results with reps but without temp= in the name, in which case we need to drop duplicates
        results = dict(results)
        # assume everything is 5 reps
        keys_to_del = {k for k in results.keys() if '_temp=' not in k and k.replace('_5reps', '_temp=1.0_5reps') in results}
        results = [(k, v) for k, v in results.items() if k not in keys_to_del]
        print(f'Removed a total of {len(keys_to_del)} files that found duplicates of with _temp=1.0 in the name')


    columns = ['Family', 'parameters', 'tokens', 'type'] + COLUMNS + ['order', 'filename']
    rows = []
    for filename, df in results:
        model_types, params, tokens = [], [], []
        get_params_and_model(filename, model_types, params, tokens)
        family, params, tokens = model_types[0], params[0], tokens[0]
        if family not in whitelist:
            continue
        params = (f'{params/1000000}M' if params < 1000000000 else f'{params/1000000000}B').replace('.0', '')
        model_type = 'chat' if '-chat-hf' in filename else 'SFT' if 'SFT' in filename else 'Instruct' if 'Instruct' in filename else 'Dolly' if family == 'Dolly' else 'base'
        if '_temp' in filename:
            temp = filename.split('_temp')[-1].split('_')[0][1:]
            if include_temp:
                model_type += f' temp={temp}'
            elif float(temp) != 1:
                continue

        family = family.replace('OLMo-premature', 'OLMo').replace('Dolly', 'Pythia')
        correct = df['correct'].mean()
        hallucinations = df['hallucinated'].mean()
        repetitions = df['repeated'].mean()
        if 'eos' in df:
            bad_format = df['bad_format'].mean()
            topic_change = df['topic_change'].mean()
            eos = df['eos'].mean()
        else:
            eos = topic_change = 0
            bad_format = 25 - correct - hallucinations - repetitions
            # raise ValueError(f'Backward compatibility is not supported anymore. please re-run check on {filename}')

        if 'order' in df and not pd.isna(df.order.iloc[0]):
            try:
                order = np.stack(df['order'].apply(eval))
            except:
                order = list(df['order'].apply(eval))
                max_len = max(map(len, order))
                order = np.array([o + [o[-1]]*(max_len-len(o)) for o in order])
        else:
            order = []
        if PLOT_TOPIC_CHANGE_SEPARATELY:
            rows.append([family, params, tokens, model_type, correct, hallucinations, repetitions, bad_format, topic_change, eos, order, os.path.join(results_dir, filename)])
        else:
            if len(order) > 0:
                order[order==5] = 4  # Change everything to bad_format
            rows.append([family, params, tokens, model_type, correct, hallucinations, repetitions, bad_format+topic_change+eos, order, os.path.join(results_dir, filename)])


    df = pd.DataFrame(rows, columns=columns).fillna(0)

    if reps and include_temp and augment_greedy_from is not None:
        greedy_df = prepare_results_dataframe(experiment_type, augment_greedy_from, whitelist)
        df['otype'] = df.type.apply(lambda x: x.split(' ')[0])
        for _,  (family, parameters, tokens, type)  in df.groupby(['Family', 'parameters', 'tokens', 'otype'])['correct'].max().reset_index(drop=False).drop('correct', axis=1).iterrows():
            filtered_df = greedy_df[(greedy_df["Family"] == family) &
                                    (greedy_df["parameters"] == parameters) &
                                    (greedy_df["tokens"] == tokens) &
                                    (greedy_df["type"] == type)]
            if len(filtered_df) == 1:
                filtered_df['type'] = filtered_df['type'].apply(lambda x: x + ' temp=0')
                df = df.append(filtered_df, ignore_index=True)
        df = df.reset_index(drop=True)
        df.drop('otype', axis=1, inplace=True)
    return df

def sort_by(key):
    def f(row):
        if key == 'parameters':
            multiplier = 1000000 if row[key].endswith("M") else 1000000000
            return float(row[key][:-1]) * multiplier
        elif key == 'tokens':
            return int(row[key]*1e9)  # it's always in tokens
        else:  # type
            multiplier = 1000000 if row['parameters'].endswith("M") else 1000000000
            parameters = float(row['parameters'][:-1]) * multiplier
            temp = 0
            k = row[key]
            if '=' in k:
                k, temp = k.split(' ')
                temp = float(temp.split('=')[1])
            return parameters + defaultdict(lambda: 3, {'base': 1, 'SFT': 2})[k] + temp
    return f

def plot_refined_stacked_barplot(df, key_column, out_path, key, skip_single_cols=True, show_perc=False,
                                 width=None, height=None, font_size=22, rotate=True, factor=1,
                                 linewidth=3, min_value=1, y_label='Average answer counts', bbox_y_shift=None,
                                 plot_family_name=True, angle=30, x_axis_label=None, title=None):
    df = df.copy()

    df['OFamily'] = df['Family'].copy()
    if key_column != 'type':
        df = df[df.type=='base']
        if len(df) == 0:
            return
    if key_column != 'parameters':
        df['Family'] =  df['Family'] + '-' + df['parameters']
    if key_column != 'tokens':
        # in each family-key_column group, keep the row with the highest tokens
        df['tokens'] = df['tokens'].astype(int)
        df = df.sort_values('tokens', ascending=False).drop_duplicates(['Family', key_column])
    df['CustomSortOrder'] = df.apply(sort_by(key_column), axis=1)
    # Sort by 'Family' first, then by 'CustomSortOrder'
    df.sort_values(by=['OFamily', 'CustomSortOrder'], inplace=True)

    if key_column == 'tokens':
        df['tokens'] = df['tokens'].apply(lambda x: f'{round(x)}B')


    # Find unique families and parameters for plotting
    families = df['Family'].unique()
    parameters = df[key_column].unique()

    n_bars = 0
    n_families_to_plot = 0
    for family in families:
        family_df = df[df['Family'] == family]
        if len(family_df) == 1 and skip_single_cols:
            continue
        n_bars += len(family_df)
        n_families_to_plot += 1

    # Setting up the plot
    height = height or 8
    width = width or max(2, 2*n_bars)
    height, width = height * factor, width * factor
    fig, ax = plt.subplots(figsize=(width, height))

    # Preparing the base of stacked bars
    bar_width = (width/(5*n_bars)) * factor
    gap = 0.1 * factor
    current_position = 0

    # Legends and labels
    legends = COLUMNS

    # Track positions for x-ticks
    x_ticks_positions = []
    x_ticks_labels = []

    for family in families:
        family_df = df[df['Family'] == family]
        if len(family_df) == 1 and skip_single_cols:
            continue
        for param in family_df[key_column].unique():
            sub_df = family_df[family_df[key_column] == param]

            # Stack each component
            bottom = 0
            for index, component in enumerate(legends):
                value = sub_df.iloc[0][component]
                bar = ax.bar(current_position, value, bottom=bottom, color=colors[index], edgecolor='black',
                             hatch=patterns[index], label=legends[index] if current_position == 0 else "",
                             width=bar_width)
                bottom += value

                # Adding value label on each stack if value > 1
                if value > min_value:
                    total = sum(sub_df.iloc[0][legends])
                    text = f'{value:.1f}' if not show_perc else f'{value:.1f}\n({value/total:.1%})'
                    y_pos = bottom - value / 2
                    if show_perc:
                        y_pos -= value / 4
                    plt.rcParams['text.usetex'] = False
                    ax.text(current_position, y_pos, text, ha='center', fontsize=font_size, color='white')

            x_ticks_positions.append(current_position)
            x_lab = sub_df.iloc[0][key_column]
            if 'base temp=' in x_lab:
                # x_lab = x_lab.replace('base temp=', '$\\tau=') + '$'
                x_lab = x_lab.replace('base temp=', '')
            if n_families_to_plot > 1 and plot_family_name:
                x_ticks_labels.append(f"{family}-{x_lab}")
            else:
                x_ticks_labels.append(f"{x_lab}")
            current_position += bar_width + gap

        # Add extra gap after each family
        current_position += gap * 4

    if key in CORRECT_UPPER_BOUND:
        ax.axhline(y=CORRECT_UPPER_BOUND[key], color=colors[0], linestyle='--', linewidth=linewidth)
    # TODO - add title?

    # Adjusting the plot
    ax.set_xticks(x_ticks_positions)
    if rotate:
        ax.set_xticklabels(x_ticks_labels, rotation=angle, ha="right")
        bbox_y_shift = bbox_y_shift or -0.2
    else:
        ax.set_xticklabels(x_ticks_labels)
        bbox_y_shift = -0.1
    ax.tick_params(axis='x', labelsize=font_size)  # Increase x-tick font size
    ax.tick_params(axis='y', labelsize=font_size)  # Increase y-tick font size
    ax.set_ylabel(y_label, fontsize=font_size)
    # ax.set_xlabel(f'Family-{key_column.capitalize()}')

    # Moving the legend under the plot
    ncol = 3 if PLOT_TOPIC_CHANGE_SEPARATELY else 4
    if width > 20:
        # font_size *= 1.25
        ncol = 6
    ax.legend(bbox_to_anchor=(0.5, bbox_y_shift), loc='upper center', ncol=ncol, fontsize=font_size)

    if title is not None:
        ax.set_title(title, fontsize=font_size)
    if x_axis_label is not None:
        ax.set_xlabel(x_axis_label, fontsize=font_size)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close('all')
    # plt.show()


def plot_answers_ordering(out_path, upper_bound, orders, order_by_gold=False, order_by_correct=False, transpose=True,
                          order_by_blocks=True, **kwargs):
    assert not (order_by_correct and order_by_gold), "Can't order by both correct and gold"
    assert not (order_by_blocks and order_by_gold), "Can't order by both block and gold"
    assert not (order_by_correct and order_by_blocks), "Can't order by both correct and blocks"

    # Prepare data and positions
    data = orders
    plot_upper_bound = upper_bound is not None
    y_positions = np.full(data.shape[0], fill_value=data.shape[1])
    y_positions = np.clip(y_positions, a_min=1, a_max=25)  # Adjust as needed

    if order_by_gold:
        indices = np.argsort(y_positions)
        if transpose:
            indices = indices[::-1]
        y_positions = y_positions[indices]
        data = data[indices]
    if order_by_correct:
        count_1s = np.sum(data == 1, axis=1)
        count_2s = np.sum(data == 2, axis=1)
        count_3s = np.sum(data == 3, axis=1)

        # Create a structured array with counts
        structured_array = np.core.records.fromarrays([count_1s, count_2s, count_3s],
                                                      names='count_1s, count_2s, count_3s')
        # Argsort based on the structured array
        indices = np.argsort(structured_array, order=['count_1s', 'count_2s', 'count_3s'])[::-1]

        if transpose:
            indices = indices[::-1]
        data = data[indices]
        y_positions = y_positions[indices]
    if order_by_blocks:
        # repetitions blocks
        kernel = np.array([1, 5, 1])
        convolved_array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=data)
        rep_sequences = np.sum(convolved_array == 21, axis=1)

        # kernel = np.array([1, 5, 1])
        # convolved_array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=data)
        # hal_sequences = np.sum(convolved_array == 14, axis=1)
        #
        # kernel = np.array([1, 1, 1])
        # convolved_array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=data)
        # cor_sequences = np.sum(convolved_array == 3, axis=1)

        count_1s = np.sum(data == 1, axis=1)
        count_2s = np.sum(data == 2, axis=1)

        # Create a structured array with counts
        # structured_array = np.core.records.fromarrays([rep_sequences, hal_sequences, cor_sequences],
        #                                               names='count_1s, count_2s, count_3s')
        structured_array = np.core.records.fromarrays([rep_sequences, count_1s, count_2s],
                                                      names='count_1s, count_2s, count_3s')

        # Argsort based on the structured array
        indices = np.argsort(structured_array, order=['count_1s', 'count_2s', 'count_3s'])[::-1]

        if transpose:
            indices = indices[::-1]
        data = data[indices]
        y_positions = y_positions[indices]

    # Transpose data if required
    if transpose:
        data = data.T
        x_positions = y_positions  # In transposed case, these are actually the x positions

    xticks = np.arange(1, len(data) + 1)
    yticks = np.arange(1, len(data[0]) + 1)

    if order_by_gold or order_by_correct:
        if not transpose:
            xticks = indices + 1
        else:
            yticks = indices + 1

    fig, ax = plt.subplots(figsize=(20, 10))
    cmap = mpl.colors.ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    if PLOT_TOPIC_CHANGE_SEPARATELY:# or PAD in data:
        bounds += [5.5, 6.5]
        # if PAD in data:
        #     bounds += [7.5]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ax.pcolormesh(data.T, cmap=cmap, norm=norm)

    if plot_upper_bound:
        # Plot lines
        if transpose:
            for i in range(len(x_positions) - 1):
                ax.plot([x_positions[i], x_positions[i]], [i, i + 1], color='black', linewidth=2)
                if x_positions[i] != x_positions[i + 1]:
                    ax.plot([x_positions[i], x_positions[i + 1]], [i + 1, i + 1], color='black', linewidth=2)
            ax.plot([x_positions[-1], x_positions[-1]], [len(x_positions) - 1, len(x_positions)], color='black', linewidth=2)
        else:
            for i in range(len(y_positions) - 1):
                ax.plot([i, i + 1], [y_positions[i], y_positions[i]], color='black', linewidth=2)
                if y_positions[i] != y_positions[i + 1]:
                    ax.plot([i + 1, i + 1], [y_positions[i], y_positions[i + 1]], color='black', linewidth=2)
            ax.plot([len(y_positions) - 1, len(y_positions)], [y_positions[-1], y_positions[-1]], color='black', linewidth=2)

    # Adjust ticks and labels
    # ax.set(xticks=np.arange(0.5, len(data) + 0.5), yticks=np.arange(0.5, len(data[0]) + 0.5),
    #        xticklabels=xticks, yticklabels=yticks,
    #        xlim=(0, len(data)), ylim=(0, len(data[0])))
    # plt.xticks(rotation=90)
    plt.xticks([])
    plt.yticks([])
    fontsize = kwargs.get('fontsize', 18)
    if 'x_label' in kwargs:
        ax.set_xlabel(kwargs['x_label'], fontsize=fontsize)
    if 'y_label' in kwargs:
        ax.set_ylabel(kwargs['y_label'], fontsize=fontsize)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_path)


def generate_plots(output_dir, dfs, family_whitelist=None, do_orders=False, keys=None, params_white_list=None,
                   plots=('parameters', 'tokens', 'type'), tokens_white_list=None, tokens_black_list=None,
                   ordering_kwargs=None, **kwargs):
    # for k, v in {'width': 6, 'height': 4, 'font_size': 18}.items():
    #     if k not in kwargs:
    #         kwargs[k] = v
    ordering_kwargs = ordering_kwargs or {}
    os.makedirs(output_dir, exist_ok=True)

    if keys is not None:
        dfs = {k: dfs[k] for k in keys}
    if (family_whitelist is not None or params_white_list is not None or
            tokens_white_list is not None or tokens_black_list is not None):
        new_dfs = {}
        for key, df in dfs.items():
            if len(df) == 0:
                continue
            sub_df = df
            if params_white_list is not None:
                sub_df = df[df.parameters.isin(params_white_list)]
            if family_whitelist is not None:
                sub_df = sub_df[sub_df.Family.isin(family_whitelist)]
            if tokens_white_list is not None:
                sub_df = sub_df[sub_df.tokens.isin(tokens_white_list)]
            if tokens_black_list is not None:
                sub_df = sub_df[~sub_df.tokens.isin(tokens_black_list)]
            if len(sub_df) > 0:
                new_dfs[key] = sub_df
        dfs = new_dfs
    if len(dfs) == 0:
        return


    for key, df in dfs.items():
        if len(df) == 0:
            continue
        show_perc = False
        clean_key = key.replace('_random', '').replace('_no25', '').replace('_temperature', '').replace('_instruction', '').replace('_rand_temp05', '').replace('_idk', '').replace('_icl', '').replace('_clean', '')
        if clean_key.startswith('open_ended'):
            clean_key = 'open_ended'  # remove the popularity suffix
            show_perc = True
        skip_single_cols = key == clean_key and key != 'open_ended'
        if 'skip_single_cols' in kwargs:
            skip_single_cols = kwargs.pop('skip_single_cols')
        if 'parameters' in plots:
            plot_refined_stacked_barplot(df, 'parameters', os.path.join(output_dir, f'{key}_parameters.pdf'), clean_key, skip_single_cols, show_perc=show_perc, **kwargs)
        if 'tokens' in plots:
           plot_refined_stacked_barplot(df, 'tokens', os.path.join(output_dir, f'{key}_tokens.pdf'), clean_key, skip_single_cols, **kwargs)
        if 'type' in plots:
            plot_refined_stacked_barplot(df, 'type', os.path.join(output_dir, f'{key}_type.pdf'), clean_key, skip_single_cols, **kwargs)


    if 'hq_qampari' in dfs and len(dfs['hq_qampari']) and do_orders:
        HQ_QAMPARI_ANSWERS_PATH = 'data/hq_manual_easy_qampary_questions.jsonl'
        with open(HQ_QAMPARI_ANSWERS_PATH, 'r') as f:
            hq_qampari_answers = {k: v for line in f.readlines() for k, v in json.loads(line.strip()).items()}
            hq_n_answers = np.array(list(map(len, hq_qampari_answers.values())))
        for _, df in dfs['hq_qampari'].iterrows():
            if len(df.order) == 0:
                continue
            os.makedirs(os.path.join(output_dir, 'orders'), exist_ok=True)
            out_path = os.path.join(output_dir, 'orders', f'{df.Family}_{df.type}_{df.parameters}_{round(df.tokens)}T.pdf')
            plot_answers_ordering(out_path, hq_n_answers, df.order, **ordering_kwargs)

    if 'open_ended' in dfs and len(dfs['open_ended']) and do_orders:
        for _, df in dfs['open_ended'].iterrows():
            if len(df.order) == 0:
                continue
            os.makedirs(os.path.join(output_dir, 'open_ended_orders'), exist_ok=True)
            out_path = os.path.join(output_dir, 'open_ended_orders', f'{df.Family}_{df.type}_{df.parameters}_{round(df.tokens)}T.pdf')
            plot_answers_ordering(out_path, None, df.order, **ordering_kwargs)


def generate_all_plots(output_dir, dfs):
    white_list = FAMILIES_WHITE_LIST

def add_shift_score(df):
    if len(df) == 0 or len(df.iloc[0].order) == 0:
        return None

    def format_shift_score(ss):
        return f'{np.mean(ss):.1%}\\pm{np.std(ss):.1%}'

    def shift_score(shuffle):
        def f(order):
            ss = []
            for o in order:
                o = [i for i in o if 1<=i<=3]
                if len(o) <= 5:
                    continue
                if shuffle:
                    random_permutations = np.array([np.random.permutation(o) for _ in range(1000)])
                    diffs = np.diff(random_permutations, axis=1)
                    shift_scores = np.mean(diffs >= 0, axis=1)
                    ss.append(np.mean(shift_scores))
                else:
                    ss.append(np.mean(np.diff(o)>=0))
            # return format_shift_score(ss)
            return ss
        return f

    def shift_score_p_value(order):
        ss = shift_score(False)(order)
        if len(ss) < 30:
            return np.nan, np.nan
        ss_shuffle = shift_score(True)(order)
        u_statistic, p_value = mannwhitneyu(ss, ss_shuffle)
        return u_statistic, p_value

    # df['shift'] = df['order'].apply(shift_score(False))
    # df['shuffle'] = df['order'].apply(shift_score(True))

    # drop rows where the shift score is nan
    # df = df[df['shift'] != 'nan%\\pmnan%'].reset_index()

    # df['shift_score'] = df.apply(lambda x: f'{x["shift"]} ({x["shuffle"]})', axis=1)

    df[['u_statistic', 'p_value']] = df['order'].apply(
        lambda x: pd.Series(shift_score_p_value(x))
    )
    # drop rows where p_value is nan
    df = df[~df.p_value.isna()].reset_index(drop=True)


    return df

KEY_TO_TASK_MAP = {
    'hq_qampari': '\\hq',
    'answerable_qampari': '\\QAMPARI',
    'hq_qampari_rand_temp05_clean': '\\hq ($\\tau=0.5$)',
}

def reformat_scientific_notation(df):
    def format_scientific_string(s):
        if pd.isna(s) or s == 'NaN':
            return '-'
        match = re.match(r'([-+]?[0-9]*\.?[0-9]+)e([-+]?[0-9]+)', s)
        if match:
            base, exponent = match.groups()
            return f"${base} \\times 10^{{{exponent}}}$"
        return s

    for col in df.columns:
        df[col] = df[col].apply(format_scientific_string)

    return df

def create_shift_score_tables(dfs, output_dir, with_std=True):
    new_dfs = []
    for key, df in dfs.items():
        if key not in KEY_TO_TASK_MAP or len(df) == 0:
            continue
        # df = df[~df.filename.str.contains('step')].reset_index(drop=True)
        # drop duplicates with respect to family, type and parameters, keeping the one with the most steps
        df = df.sort_values('tokens', ascending=False).drop_duplicates(['Family', 'type', 'parameters']).reset_index(drop=True)
        df = df[df.type=='base']
        df = add_shift_score(df.copy())
        if df is not None and len(df) > 0:
            df['Task'] = KEY_TO_TASK_MAP[key]
            df['Model'] = df.apply(lambda x: f'{x["Family"]}-{x["type"]+"-" if x["type"] != "base" else ""}{x["parameters"]}', axis=1)
            new_dfs.append(df)
    mini_models = ['Pythia-2.9', 'Dolly', 'OLMo', 'Llama2', 'Llama3']
    full_df = pd.concat(new_dfs).reset_index(drop=True)
    # full_df['shift_score'] = full_df.apply(lambda x: f'{x["u_statistic"]:.2f} ({x["p_value"]:.1e})', axis=1)
    full_df['shift_score'] = full_df.apply(lambda x: f'{x["p_value"]:.1e}', axis=1)
    full_df['params'] = full_df['parameters'].apply(lambda x: x[:-1] * (10**9 if x[-1] == 'B' else 10**6))
    full_df.sort_values(['Family', 'type', 'params'], inplace=True)
    full_df['SortBy'] = full_df.index.values

    mini_df = full_df[full_df.Model.isin(mini_models)]

    def save_pivot_table(df, filename, label, caption):
        # pivot the df so that the tasks are the columns, the models are the rows, and the shift scores are the values    mini_df = mini_df.pivot(index='Family', columns='Task', values='shift_score').reset_index()
        pivot_table = df.pivot(index='Model', columns='Task', values='shift_score').reset_index()
        pivot_table = pd.merge(pivot_table, full_df[['Model', 'SortBy']].drop_duplicates(), on='Model', how='left')
        pivot_table = pivot_table.sort_values(by='SortBy')
        pivot_table = pivot_table.drop(columns=['SortBy']).set_index('Model')
        print(pivot_table)
        pivot_table = reformat_scientific_notation(pivot_table).reset_index(drop=False)
        print(pivot_table)
        save_to_latex(pivot_table, os.path.join(output_dir, filename), label=label, caption=caption)

    save_pivot_table(full_df, 'full_ordering_table.tex', 'tab:full-ordering', 'Ordering of answers for the full dataset')
    save_pivot_table(mini_df, 'mini_ordering_table.tex', 'tab:mini-ordering', 'Ordering of answers for the mini dataset')



def get_topic_tokens(slice=None):
    wikipedia_tokens_path = 'data/topic_tokens.json'
    if not os.path.exists(wikipedia_tokens_path):
        import wikipedia
        from transformers import AutoTokenizer
        tokznier = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
        space_token = tokznier.convert_ids_to_tokens(tokznier.encode(' '))[0]
        topic_tokens = []
        with open('data/open_ended/topics.jsonl') as f:
            topics = [json.loads(line)['topic'] for line in f.readlines()]
        for topic in tqdm(topics):
            try:
                content = wikipedia.page(topic, auto_suggest=False).content
            except wikipedia.exceptions.DisambiguationError as e:
                print(f'Failed to get content for {topic}: {e}')  # Carlos Alfonso -> Carlos J. Alfonso  ?? William Post -> William B. Post ??
                topic_tokens.append([])
                continue
            tokens = [token.replace(space_token, ' ') for token in tokznier.tokenize(content, add_special_tokens=False)]
            topic_tokens.append(tokens)
        with open(wikipedia_tokens_path, 'w') as f:
            json.dump(topic_tokens, f)
    else:
        with open(wikipedia_tokens_path, 'r') as f:
            topic_tokens = json.load(f)
    if slice is not None:
        topic_tokens = topic_tokens[slice[0]:slice[1]]

    return topic_tokens

def get_div_score(tokens, max_tokens=512, normalize_digits=False):
    seen = set()
    div_score = []
    for j, token in enumerate(tokens, 1):
        if j > max_tokens:
            break
        if normalize_digits and token.isdigit():
            token = '0'
        seen.add(token)
        div_score.append(len(seen) / j)
    return div_score

def plot_div_score(scores_2d_arr, ax, label, plot_CI=True, style_cycler=None):
    if style_cycler is None:
        style_cycler = cycle(['-', '--', '-.', ':'])

    # Calculate the standard error of the mean (SEM)
    sem = np.nanstd(scores_2d_arr, axis=0) / np.sqrt(scores_2d_arr.shape[0])
    mean = np.nanmean(scores_2d_arr, axis=0)

    # Define confidence interval
    # confidence_interval = 1.96 * sem  # for 95% confidence
    confidence_interval = sem

    # Calculate lower and upper bounds of the confidence interval
    lower_bound = mean - confidence_interval
    upper_bound = mean + confidence_interval

    # Plot the mean scores
    line_style = next(style_cycler)
    line,  = ax.plot(mean, label=label, linestyle=line_style)
    color = line.get_color()

    # Add confidence sleeve
    if plot_CI:
        ax.fill_between(range(len(mean)), lower_bound, upper_bound, color=color, alpha=0.3)

def create_diversity_score_plots(df, output_path, slice=None, models_whitelist=None, y_label='\\texttt{DiversityScore}', font_size=18, params_whitelist=None):

    if models_whitelist is not None:
        df = df[df.Family.isin(models_whitelist)]
    if params_whitelist is not None:
        df = df[df.parameters.isin(params_whitelist)]


    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    style_cycler = cycle(['-', '--', '-.', ':', (0, (5, 10))])  # https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

    wikipedia_div_scores = [get_div_score(tokens) for tokens in get_topic_tokens(slice)]
    wikipedia_div_scores = [t + [np.nan]*(512-len(t)) for t in wikipedia_div_scores]  # padding with nan so that we won't take the mean over it
    wikipedia_div_scores = np.array(wikipedia_div_scores)
    plot_div_score(wikipedia_div_scores, ax, 'Baseline', style_cycler=style_cycler)

    for _, row in df.iterrows():
        logprobs_file = row.filename.replace('open_ended_results_', 'open_ended_logprobs_125_prompts_').replace('_answerable.csv', '_512T_temp=0_5.jsonl')
        with open(logprobs_file) as f:
            div_scores = []
            for i, line in enumerate(f.readlines()):
                if slice is not None and (i >= slice[1] or i < slice[0]):
                    continue
                logprobs = json.loads(line)['logprobs']
                top_token = [max(x.items(), key=op.itemgetter(1))[0] for x in logprobs]
                # TODO should we normalize digits tokens?
                # should we cut where we found topic change if any?
                div_score = get_div_score(top_token)
                div_scores.append(div_score + [np.nan] * (512 - len(div_score)))  # padding with nan so that we won't take the mean over it
            div_scores = np.array(div_scores)
            plot_div_score(div_scores, ax, f'{row.Family}-{row.parameters}', style_cycler=style_cycler)

    plt.grid(True)
    plt.rcParams['text.usetex'] = True
    plt.legend(fontsize=font_size, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.set_xlabel('Tokens', fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)  # Increase x-tick font size
    ax.tick_params(axis='y', labelsize=font_size)  # Increase y-tick font size
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close('fig')




if __name__ == '__main__':
    white_list = FAMILIES_WHITE_LIST
    results_dir = 'data/relaxed_eval'
    output_dir = 'paper_plots'


    dfs = {
        'hq_qampari': prepare_results_dataframe('hq_qampari', results_dir, white_list),
        'hq_qampari_random': prepare_results_dataframe('hq_qampari', results_dir, white_list, True),

        'answerable_qampari': prepare_results_dataframe('answerable_qampari', results_dir, white_list),
        'unanswerable_qampari': prepare_results_dataframe('unanswerable_qampari', results_dir, white_list),

        'hq_qampari_no25': prepare_results_dataframe('hq_qampari', os.path.join(results_dir, 'no25_exps'), white_list),
        'hq_qampari_instruction': prepare_results_dataframe('hq_qampari', os.path.join(results_dir, 'instruction_prefix'), white_list),
        'hq_qampari_temperature': prepare_results_dataframe('hq_qampari', os.path.join(results_dir, 'temp_exps'), white_list, reps=True, include_temp=True, augment_greedy_from=results_dir),
        'hq_qampari_rand_temp05': prepare_results_dataframe('hq_qampari', os.path.join(results_dir, 'temp_05_5reps'), white_list, reps=True, include_temp=True),

        'hq_qampari_icl': prepare_results_dataframe('hq_qampari',os.path.join(results_dir, 'hq_icl_no_colon'), white_list),
        'hq_qampari_idk': prepare_results_dataframe('hq_qampari',os.path.join(results_dir, 'instruction_idk'), white_list),

        'open_ended': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list),
        'open_ended_vr': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list, slice=(0, 25)),
        'open_ended_r': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list, slice=(25, 50)),
        'open_ended_m': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list, slice=(50, 75)),
        'open_ended_f': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list, slice=(75, 100)),
        'open_ended_vf': prepare_results_dataframe('open_ended', os.path.join(results_dir, 'open_ended'), white_list, slice=(100, 125)),
    }

    dfs_hq_with_colon = {
        'hq_qampari': prepare_results_dataframe('hq_qampari', os.path.join(results_dir, 'hq_with_colons'), white_list),
    }

    dfs['hq_qampari_rand_temp05_clean'] = dfs['hq_qampari_rand_temp05'].copy()
    dfs['hq_qampari_rand_temp05_clean']['type'] = dfs['hq_qampari_rand_temp05_clean']['type'].str.replace(' temp=0.5', '')

    # augment the hq_qampari_idk with the results without the instruction
    baseline_models = []
    dfs['hq_qampari_idk'] = dfs['hq_qampari_idk'][dfs['hq_qampari_idk'].type != 'base']
    for i, row in dfs['hq_qampari_idk'].iterrows():
        baseline_models.append(dfs['hq_qampari'].query(f'Family == "{row.Family}" and type == "{row.type}" and parameters == "{row.parameters}" and tokens == {row.tokens}').iloc[0])
    dfs['hq_qampari_idk']['type'] = dfs['hq_qampari_idk']['type'].apply(lambda x: x + ' (IDK)')
    dfs['hq_qampari_idk'] = pd.concat([dfs['hq_qampari_idk'], pd.DataFrame(baseline_models)]).reset_index(drop=True).sort_values(by=['Family', 'parameters', 'tokens', 'type'])


    create_diversity_score_plots(dfs['open_ended'], os.path.join(output_dir, 'diversity_scores', 'open_ended.pdf'), models_whitelist=['Pythia'])
    create_diversity_score_plots(dfs['open_ended_vr'], os.path.join(output_dir, 'diversity_scores', 'open_ended_vr.pdf'), slice=(0, 25), models_whitelist=['Pythia'])
    create_diversity_score_plots(dfs['open_ended_vf'], os.path.join(output_dir, 'diversity_scores', 'open_ended_vf.pdf'), slice=(100, 125), models_whitelist=['Pythia'])
    create_diversity_score_plots(dfs['open_ended_vf'], os.path.join(output_dir, 'diversity_scores', 'open_ended_vf_llama.pdf'), slice=(100, 125), models_whitelist=['Pythia', 'Llama3'], params_whitelist=['2.8B', '6.9B', '12B', '8B'])


    generate_plots(os.path.join(output_dir, 'fig2'), dfs, ['Pythia'], keys=['hq_qampari', 'hq_qampari_rand_temp05'], plots=[], do_orders=True, params_white_list=['12B'], tokens_white_list=[299.892736])
    generate_plots(os.path.join(output_dir, 'fig2'), dfs, ['Pythia', 'Dolly'], keys=['hq_qampari', 'hq_qampari_rand_temp05'], rotate=False, factor=1, font_size=32, linewidth=5,min_value=2,  plots=['parameters'])
    generate_plots(os.path.join(output_dir, 'fig2'), dfs, ['Pythia', 'Dolly'], keys=['hq_qampari', 'hq_qampari_rand_temp05'], rotate=False, factor=1, font_size=32, linewidth=5, min_value=1.5, params_white_list=['6.9B'], plots=['tokens'], tokens_white_list=[2.097152, 4.194304,  8.388608, 16.777216, 67.108864,33.554432, 134.217728, 299.892736])
    generate_plots(os.path.join(output_dir, 'fig2'), dfs, ['Pythia', 'Dolly'], keys=['hq_qampari', 'hq_qampari_rand_temp05'], plots=['type'])

    generate_plots(os.path.join(output_dir, 'additional_params'), dfs,
                   ['OLMo', 'Llama2', 'Llama3', 'Pythia'],
                   keys=['hq_qampari', 'answerable_qampari'], plots=['parameters'],
                   factor=1, font_size=28, linewidth=5, min_value=2, bbox_y_shift=-0.3 )
    generate_plots(os.path.join(output_dir, 'additional_params'), dfs,
                   ['OLMo', 'Llama2', 'Llama3', 'Pythia'], keys=['hq_qampari_rand_temp05_clean'],
                   plots=['parameters'], factor=1, font_size=28, linewidth=5, min_value=2, bbox_y_shift=-0.3 )
    generate_plots(os.path.join(output_dir, 'additional_params'), dfs, FAMILIES_WHITE_LIST,
                   keys=['hq_qampari', 'answerable_qampari'], plots=['type'],
                   skip_single_cols=True, bbox_y_shift=-0.4, factor=1.5, font_size=52, linewidth=7.5, min_value=2, height=13)
    generate_plots(os.path.join(output_dir, 'additional_params_no_pythia'), dfs, ['Llama2', 'OLMo-premature', 'OLMo', 'Llama3'], keys=['hq_qampari', 'hq_qampari_rand_temp05', 'answerable_qampari'], plots=['type'], skip_single_cols=True, bbox_y_shift=-0.4)

    generate_plots(os.path.join(output_dir, 'pythia_tokens'), dfs, ['Pythia'], keys=['hq_qampari'],
                   plots=['tokens'], skip_single_cols=True, rotate=False, factor=1, font_size=48, linewidth=5,
                   min_value=1.5, plot_family_name=False, height=12)
    generate_plots(os.path.join(output_dir, 'olmo_tokens'), dfs, ['OLMo'], keys=['hq_qampari'],
                   plots=['tokens'], skip_single_cols=True, rotate=False, factor=1.5, font_size=48, linewidth=5,
                   min_value=1.5, plot_family_name=False, tokens_black_list=[44, 66, 111])

    # generate_plots(os.path.join(output_dir, 'icl'), dfs, FAMILIES_WHITE_LIST, keys=['hq_qampari_icl'], rotate=False, factor=1, font_size=32, linewidth=5,min_value=2, plots=['parameters'], skip_single_cols=True)
    generate_plots(os.path.join(output_dir, 'hq_icl_no_colon'), dfs, FAMILIES_WHITE_LIST, keys=['hq_qampari_icl'],
                   rotate=True, factor=1, font_size=28, height=9, linewidth=5, min_value=2, plots=['parameters'],
                   skip_single_cols=True,  bbox_y_shift=-0.4)
    # dfs['hq_qampari_idk'] = pd.concat([dfs['hq_qampari_idk'], dfs['hq_qampari'][(dfs['hq_qampari'].Family == 'Pythia') & (dfs['hq_qampari'].parameters.isin(['2.8B', '6.9B', '12B'])) & (dfs['hq_qampari'].type == 'base')]])
    generate_plots(os.path.join(output_dir, 'idk'), dfs, FAMILIES_WHITE_LIST, keys=['hq_qampari_idk'],
                   plots=['type'], skip_single_cols=True,
                   bbox_y_shift=-0.6, factor=1.5, font_size=58, linewidth=7.5, min_value=2, height=14, angle=25)
    generate_plots(os.path.join(output_dir, 'no25'), dfs, FAMILIES_WHITE_LIST, keys=['hq_qampari_no25'],
                   plots=['parameters'], skip_single_cols=True,  linewidth=5, bbox_y_shift=-0.25)


    generate_plots(os.path.join(output_dir, 'fig_temperature'), dfs, ['Pythia'], keys=['hq_qampari_temperature'], rotate=False, factor=1, font_size=32, linewidth=5,min_value=2, width=18, plots=['type'], params_white_list=['12B'])


    generate_plots(os.path.join(output_dir, 'params'), dfs, ['Llama2', 'Llama3'], keys=['hq_qampari', 'hq_qampari_rand_temp05'], plots=['type'])
    og_kewargs = dict(rotate=False, factor=1, font_size=26, width=12, height=8,min_value=2,y_label='Atomic facts')
    open_ended_white_list = ['Pythia'] # ['Pythia','Llama3']
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list,  keys=['open_ended'], plots=['parameters'], **og_kewargs)
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list, keys=['open_ended_vr'], plots=['parameters'], **og_kewargs)
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list, keys=['open_ended_r'], plots=['parameters'],    **og_kewargs)
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list, keys=['open_ended_m'], plots=['parameters'],    **og_kewargs)
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list, keys=['open_ended_f'], plots=['parameters'],    **og_kewargs)
    generate_plots(os.path.join(output_dir, 'open_ended'), dfs, open_ended_white_list, keys=['open_ended_vf'], plots=['parameters'],    **og_kewargs)
    generate_plots(os.path.join(output_dir, 'greedy_params/pythia'), dfs, ['Pythia'], keys=['hq_qampari'], plots=['parameters'])
    generate_plots(os.path.join(output_dir, 'greedy_params/llama2'), dfs, ['Llama2'], keys=['hq_qampari'], plots=['parameters'])
    generate_plots(os.path.join(output_dir, 'greedy_params/llama3'), dfs, ['Llama3'], keys=['hq_qampari'], plots=['parameters'])
    generate_plots(os.path.join(output_dir, 'greedy_params/olmo'), dfs, ['OLMo'], keys=['hq_qampari'], plots=['parameters'])


    generate_plots(os.path.join(output_dir, 'fqampari'), dfs, ['Pythia'], keys=['unanswerable_qampari'], rotate=False, factor=1, font_size=32, linewidth=5,min_value=2, plots=['parameters'])

    generate_plots(os.path.join(output_dir, 'orders'), dfs, ['Pythia', 'OLMo', 'Llama2', 'Llama3'], keys=['hq_qampari'], plots=['parameters'], do_orders=True)

    generate_plots(os.path.join(output_dir, 'hq_with_colon'), dfs_hq_with_colon, FAMILIES_WHITE_LIST, keys=['hq_qampari'], plots=['parameters'], do_orders=False)

    print('Done creating paper plots, now creating tweet plots')
    generate_plots(os.path.join(output_dir, 'tweet'), dfs, ['Pythia'], keys=['hq_qampari'], rotate=False,
                   factor=1, font_size=26, linewidth=5,min_value=2,  plots=['parameters'], x_axis_label='Parameters',
                   bbox_y_shift = -0.5, height=9,
                   title='Larger Pythia models fallback to more hallucinations over repetitions')
    generate_plots(os.path.join(output_dir, 'tweet'), dfs, ['OLMo'], keys=['hq_qampari'],
                   params_white_list=['1B'], plots=['tokens'], skip_single_cols=True, rotate=False, factor=1,
                   linewidth=5, min_value=1.5, plot_family_name=False, tokens_black_list=[44, 66, 111],
                   x_axis_label='Pretraining tokens seen', bbox_y_shift=-0.4,
                   title='Better trained OLMo-1B fallbacks to more hallucinations over repetitions')
    generate_plots(os.path.join(output_dir, 'tweet'), dfs, ['Llama2', 'Llama3'],
                   keys=['hq_qampari'], plots=['type'], skip_single_cols=True, bbox_y_shift=-0.5, height=9,
                   title='Instruction-tuned models hallucinate more', min_value=2.5)
    generate_plots(os.path.join(output_dir, 'tweet'), dfs, ['Pythia'], keys=['hq_qampari_temperature'],
                   rotate=False, factor=1, font_size=26, linewidth=5,min_value=2, width=18, plots=['type'],
                   bbox_y_shift=-0.5, height=9,
                   params_white_list=['12B'], x_axis_label='Pythia 12B random sampling temperature',
                   title='Randomness in the decoding shifts fallback behavior')
    create_diversity_score_plots(dfs['open_ended_vf'],
                                 os.path.join(output_dir, 'tweet', 'open_ended_vf.pdf'), slice=(100, 125),
                                 models_whitelist=['Pythia'], y_label='Unique tokens proportion', font_size=12)
    generate_plots(os.path.join(output_dir, 'tweet'), dfs, ['Pythia'], keys=['hq_qampari'],
                   plots=[], do_orders=True, params_white_list=['12B'], tokens_white_list=[299.892736],
                   ordering_kwargs={'x_label': 'Generated facts', 'y_label': 'Questions',
                                    'title': 'Pythia-12B shifts fallback during generations',
                                    'fontsize': 32})


    generate_plots(os.path.join(output_dir, 'talk'), dfs, ['Llama2',  'OLMo-premature', 'OLMo', 'Llama3'], keys=['hq_qampari_idk'],
                   plots=['type'], skip_single_cols=True, rotate=True, factor=1, angle=25,
                   linewidth=5, min_value=1.5, plot_family_name=True,
                   x_axis_label='Pretraining tokens seen', bbox_y_shift=-0.4,
                   title='Instruction tuned models hallucinate even when told not to')

    generate_plots(os.path.join(output_dir, 'talk'), dfs, ['Pythia'], keys=['hq_qampari_icl'],
                   rotate=False,factor=1, font_size=26, linewidth=5,min_value=2,  plots=['parameters'],
                   x_axis_label='Parameters', bbox_y_shift = -0.5, height=9,
                   title='Pythia models hallucinate even when shown how to change topics')

    generate_plots(os.path.join(output_dir, 'talk'), dfs, ['Pythia'], keys=['unanswerable_qampari'],
                   rotate=False,factor=1, font_size=26, linewidth=5,min_value=2,  plots=['parameters'],
                   x_axis_label='Parameters', bbox_y_shift = -0.5, height=9,
                   title='Pythia models hallucinate even when asked on fictitious entities')

    generate_plots(os.path.join(output_dir, 'talk'), dfs, ['Pythia'], keys=['hq_qampari_no25'],
                   rotate=False,factor=1, font_size=26, linewidth=5,min_value=2,  plots=['parameters'],
                   x_axis_label='Parameters', bbox_y_shift = -0.5, height=9,
                   title='Pythia models hallucinate even when not told to produce 25 answers')

    generate_plots(os.path.join(output_dir, 'talk'), dfs, ['Pythia'], keys=['hq_qampari_rand_temp05_clean'],
                   rotate=False,factor=1, font_size=26, linewidth=5,min_value=2,  plots=['parameters'],
                   x_axis_label='Parameters', bbox_y_shift = -0.5, height=9,
                   title='Pythia models fallback behaviors with temperature sampling')


    print('Done creating plots, now moving to p-value table. this can take a while..')
    create_shift_score_tables(dfs, output_dir)