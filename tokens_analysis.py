import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken


PROMPT_DIR = 'palm2/prompt'
EFFECTIVE_PROMPT_DIR = 'palm2/effective_prompt'
CODE_DIR = 'palm2/code'
PROMPTS_SUMMARY_FILE = 'prompts.json'
OUTPUT_DATA_FILE = 'tokens_count.csv'
STATEFUL_OUTPUT_DATA_FILE = 'tokens_count_stateful.csv'
# Get these from the spreadsheet
PROBLEM_IDX_WITH_WRONG_OUTPUT = [1, 3, 7, 11, 20, 25, 30]

# https://colab.research.google.com/github/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
encoding = tiktoken.get_encoding('p50k_base')  # ('cl100k_base')


def edit_distance(s1: list[int], s2: list[int]) -> int:
    """
    Return the edit distance between two sequence of integers.

    :param s1: The first sequence
    :param s2: The second sequence
    :return The edit distance between the two sequences
    """

    m = len(s1)
    n = len(s2)

    distance = [[0 for j in range(n+1)] for i in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                distance[i][j] = i + j
            elif s1[i-1] == s2[j-1]:
                distance[i][j] = distance[i-1][j-1]
            else:
                distance[i][j] = min(distance[i-1][j], distance[i][j-1], distance[i-1][j-1]) + 1

    return distance[m][n]


def num_tokens_from_string(string: str) -> int:
    """
    Return the number of tokens in a text string.

    :param string: Input text
    :return: The count of tokens
    """

    global encoding

    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_prompt_code_tokens(prompt_id: int, trial_id: int = None) -> tuple[int, int, int]:
    """
    Return the number of tokens in a given prompt and the code generated in response.

    :param prompt_id: The prompt ID
    :param trial_id: The trial number: None, 1, or 2
    :return: The token counts for the prompt, the effective prompt, and the code
    """

    if trial_id:
        prompt_file_name = f'prompt_{prompt_id}-{trial_id}.txt'
        code_file_name = f'code_{prompt_id}-{trial_id}.txt'
    else:
        prompt_file_name = f'prompt_{prompt_id}.txt'
        code_file_name = f'code_{prompt_id}.txt'

    with open(
        f'{PROMPT_DIR}/{prompt_file_name}',
        'r'
    ) as in_file1, open(
        f'{CODE_DIR}/{code_file_name}',
        'r'
    ) as in_file2, open(
        f'{EFFECTIVE_PROMPT_DIR}/{prompt_file_name}',
        'r'
    ) as in_file3:
        prompt = in_file1.read().strip()
        code = in_file2.read().strip()
        effective_prompt = in_file3.read().strip()
        x, y, z = (
            num_tokens_from_string(prompt),
            num_tokens_from_string(effective_prompt),
            num_tokens_from_string(code)
        )

    return x, y, z


def generate_count_data():
    """
    Generate a CSV file containing the various tokens count for each prompt.
    """

    with open(PROMPTS_SUMMARY_FILE, 'r') as in_file:
        json_data = json.loads(in_file.read())

    with open(OUTPUT_DATA_FILE, 'w') as out_file:
        out_file.write('prompt,trial,p_count,ep_count,c_count\n')

        for prompt in json_data['data']:
            idx, max_trials = prompt['id'], prompt['max_trials']

            for j in range(max_trials):
                if j == 0:
                    p_count, ep_count, c_count = count_prompt_code_tokens(idx)
                else:
                    p_count, ep_count, c_count = count_prompt_code_tokens(idx, j)

                out_file.write(f'{idx},{j},{p_count},{ep_count},{c_count}\n')


def generate_stateful_count_data():
    """
    Generate a CSV file containing the various tokens count for each stateful prompt.
    Given max_trials of a prompt, compute the edit distances between the successive versions
    to get the total token count. For code, consider only the final version.
    Only the effective prompt texts are considered here.
    """

    with open(PROMPTS_SUMMARY_FILE, 'r') as in_file:
        json_data = json.loads(in_file.read())

    with open(STATEFUL_OUTPUT_DATA_FILE, 'w') as out_file:
        out_file.write('prompt,max_trials,ep_count,c_count,correct\n')

        for prompt in json_data['data']:
            idx, max_trials = prompt['id'], prompt['max_trials']
            file_names = [f'prompt_{idx}.txt']

            if idx in PROBLEM_IDX_WITH_WRONG_OUTPUT:
                is_correct = 0
            else:
                is_correct = 1

            if max_trials > 1:
                final_code_file_name = f'code_{idx}-{max_trials - 1}.txt'
            else:
                final_code_file_name = f'code_{idx}.txt'

            # Size of the original effective prompt and the final code
            with open(f'{EFFECTIVE_PROMPT_DIR}/{file_names[0]}', 'r') as in_file:
                ep_count = num_tokens_from_string(in_file.read().strip())

            with open(f'{CODE_DIR}/{final_code_file_name}', 'r') as in_file:
                c_count = num_tokens_from_string(in_file.read().strip())

            for j in range(1, max_trials):
                file_names.append(f'prompt_{idx}-{j}.txt')

            for file1, file2 in zip(file_names, file_names[1:]):
                with open(
                        f'{EFFECTIVE_PROMPT_DIR}/{file1}',
                        'r'
                ) as in_file1, open(
                    f'{EFFECTIVE_PROMPT_DIR}/{file2}',
                    'r'
                ) as in_file2:
                    s1 = encoding.encode(in_file1.read().strip())
                    s2 = encoding.encode(in_file2.read().strip())
                    delta = edit_distance(s1, s2)
                    ep_count += delta
                    print(idx, max_trials, j, file1, file2, delta)

            out_file.write(f'{idx},{max_trials},{ep_count},{c_count},{is_correct}\n')


def plot_graphs(df: pd.DataFrame):
    x = np.arange(df.shape[0])
    width = 0.3
    multiplier = 0
    offset = width * multiplier

    fig, ax = plt.subplots()

    ax.bar(x + offset, df['p_count'], width, label='Prompt')
    multiplier = 1
    offset = width * multiplier
    ax.bar(x + offset, df['ep_count'], width, label='Effective prompt')
    multiplier = 2
    offset = width * multiplier
    ax.bar(x + offset, df['c_count'], width, label='Code', color='cyan')
    ax.set_xlabel('Prompt index')
    ax.set_ylabel('Count of tokens')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    width = 0.35
    multiplier = 0
    offset = width * multiplier

    ax.bar(x + offset, df['efficiency'], width, label='Code vs. prompt')
    multiplier = 2
    offset = width * multiplier
    ax.bar(x + offset, df['ep_efficiency'], width, label='Code vs. effective prompt')
    ax.set_xlabel('Prompt index')
    ax.set_ylabel('Ratio of tokens count')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_graphs_for_stateful_prompts(df: pd.DataFrame):
    x = np.arange(df.shape[0])
    width = 0.3
    multiplier = 0
    offset = width * multiplier

    fig, ax = plt.subplots()

    ax.bar(x + offset, df['ep_count'], width, label='Prompt')
    multiplier = 1
    offset = width * multiplier
    ax.bar(x + offset, df['c_count'], width, label='Code', color='cyan')
    ax.set_xlabel('Prompt index')
    ax.set_ylabel('Count of tokens')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    width = 0.35
    multiplier = 0
    offset = width * multiplier

    ax.bar(x + offset, df['ep_efficiency'], width, label='Code vs. effective prompt')
    ax.set_xlabel('Prompt index')
    ax.set_ylabel('Ratio of tokens count')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def build_df(file_name: str) -> pd.DataFrame:
    """
    Create a dataframe holding the prompt and code size values.

    :param file_name: The input CSV file
    :return: The dataframe
    """

    df = pd.read_csv(file_name)

    if 'p_count' in df.columns:
        df['efficiency'] = df['c_count'] / df['p_count']

    df['ep_efficiency'] = df['c_count'] / df['ep_count']
    print(df)

    if 'efficiency' in df.columns:
        print(f'Avg. efficiency: {df["efficiency"].mean()}')

    print(f'Avg. ep_efficiency: {df["ep_efficiency"].mean()}')

    if 'max_trials' in df.columns:
        print(f'Sequential prompts: avg. no. of trials per problem: {df["max_trials"].mean()}')

    return df


def main():
    generate_count_data()
    generate_stateful_count_data()

    df = build_df(OUTPUT_DATA_FILE)
    df.to_csv('count.csv', index=True)

    df = build_df(STATEFUL_OUTPUT_DATA_FILE)
    df.to_csv('sequential_count.csv', index=True)

    plot_graphs_for_stateful_prompts(df)


if __name__ == '__main__':
    # prompt_1 =   [16447, 257, 13860, 18250, 598, 351, 257, 2420, 3091, 11, 257, 4936, 11, 290, 257, 6167, 13, 1649, 262, 4936, 318, 28384, 11, 262, 15598, 5174, 422, 262, 2420, 815, 307, 9066, 287, 262, 6167, 13, 5765,      257,   3303, 2746, 422,     12905, 2667,    15399, 14699, 351, 16332, 35491, 11812,     329, 428, 4007, 13, 383, 42253, 7824, 1994, 815, 307, 1100, 422, 281, 2858, 7885, 13, 198, 198, 26410, 25, 198, 15506, 63, 29412]
    # prompt_1_1 = [16447, 257, 13860, 18250, 598, 351, 257, 2420, 3091, 11, 257, 4936, 11, 290, 257, 6167, 13, 1649, 262, 4936, 318, 28384, 11, 262, 15598, 5174, 422, 262, 2420, 815, 307, 9066, 287, 262, 6167, 13, 5765,      16332, 35491, 290,          12905, 2667,    32388, 16066,                               329, 428, 4007, 13, 383, 42253, 7824, 1994, 815, 307, 1100, 422, 281, 2858, 7885, 13, 198, 198, 26410, 25, 198, 15506, 63, 29412]
    # print(edit_distance(prompt_1, prompt_1_1))

    main()
