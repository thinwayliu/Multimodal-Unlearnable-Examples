import numpy as np

# ANSI escape code
ansi_code_ids =  [15, 224, 217, 210, 203, 9, 160, 124, 88]
length = len(ansi_code_ids)

hexcolors = [
    "#f5f5ff",
    "#ffe0e0",
    "#ffb6b6",
    "#ff8d8d",
    "#ff6363",
    "#ff3939",
    "#ff1010",
    "#f20000",
    "#dd0000",
    "#c80000",
    "#b40000",
    "#9f0000",
    "#8a0000",
]
hex_lenght = len(hexcolors)

template = "\033[38;5;{value}m{string}\033[0m"
def print_color_text(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        vid = int(w * length)
        value = ansi_code_ids[vid]
        print(template.format(string=string, value=value), end="")
    print()

markdown_template = '<font color="{}">{}</font>'
def render_color_markdown(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    ss = []
    for string, w in zip(text, ws):
        i = int(w * hex_lenght)
        ss.append(markdown_template.format(hexcolors[i], string))
    return "".join(ss)

def print_match_subsequence(text1, text2, reverse_render=False):
    # 高亮两个文本的最长公共子序列（LCS）
    s1 = len(text1)
    s2 = len(text2)
    dp = np.zeros((s1 + 1, s2 + 1), dtype=np.int32)
    for i in range(1, s1 + 1):
        for j in range(1, s2 + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    maxlen = dp[-1][-1]

    i = s1 - 1
    j = s2 - 1
    locs = []
    while len(locs) < maxlen:
        if text1[i] == text2[j]:
            locs.append((i, j))
            i -= 1
            j -= 1
        elif dp[i + 1, j] > dp[i, j + 1]:
            j -= 1
        else:
            i -= 1
    locs[::-1] = locs

    sw1 = np.zeros(s1)
    sw2 = np.zeros(s2)
    ts1 = set([i for i, _ in locs])
    ts2 = set([j for _, j in locs])
    for i, j in enumerate(text1):
        if i in ts1:
            sw1[i] = 1
    for i, j in enumerate(text2):
        if i in ts2:
            sw2[i] = 1

    if reverse_render:
        sw1 = 1 - sw1
        sw2 = 1 - sw2

    print_color_text(text1, sw1)
    print_color_text(text2, sw2)