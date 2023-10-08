import os, re
from sklearn.decomposition import PCA
from typing import Counter
from sklearn import cluster
import json
import numpy
import matplotlib.pyplot as plt
# from random import sample
import random
import argparse
import tqdm
import stanza

TAG_DICT = "dataset/tag_dict.json"

with open(TAG_DICT) as f:
    tag_dict = json.load(f)


def serialize_tree_with_fo(tree:dict, pure_parse:list, parse:list):
        """Recursively traverse the tree in the first-order."""
        if tree is not None:
            if len(tree.children)!=0:
                pure_parse.append(f" ({tree.label}")
                parse.append(f" ({tree.label}")
            else:
                pure_parse.append(" ")
                parse.append(f" {tree.label}")
            for i in range(len(tree.children)):
                serialize_tree_with_fo(tree.children[i], pure_parse, parse)
            if len(tree.children)!=0:
                pure_parse.append(")")
                parse.append(")")
        return pure_parse, parse


def seq2dict(seq, max_level):
    root = {'value': seq.split(" ")[0]}
    if max_level == 1:
        return root
    child = []
    stack = 0
    begin = -1
    for i in range(0, len(seq)):
        if seq[i] == '(':
            if stack == 0:
                begin = i
            stack += 1
        elif seq[i] == ')':
            stack -= 1
            if stack == 0:
                child.append(seq2dict(seq[begin+1:i], max_level-1))
    if child:
        root['child'] = child
    return root


def dict2bag(root, bag):
    bag[tag_dict[root['value']]] += 1
    if 'child' in root:
        for child in root['child']:
            dict2bag(child, bag)



def preorderTraversal(root, nums):
    nums.append(tag_dict[root['value']])
    if 'child' in root:
        for child in root['child']:
            preorderTraversal(child, nums)


def levelorderTraversal(root, nums):
    queue = [root]
    while queue:
        cur = queue.pop(0)
        nums.append(tag_dict[cur['value']])
        if 'child' in cur:
            for child in cur['child']:
                queue.append(child)


def lcs_distance(s1, s2, weight_decay=1):
    tot_len = 0
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] == 1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] == 1

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = 0
                if dp[i-1][j-1] > 1:
                    tot_len += dp[i-1][j-1] * \
                        (weight_decay**i+weight_decay**j)/2
    if dp[len(s1)-1][len(s2)-1] > 1:
        tot_len += dp[len(s1)-1][len(s2)-1]
    return 1-tot_len/min(len(s1), len(s2))


def lcs_distance_qiji(s1, s2, weight_decay=1):
    tot_len = 0
    tot_m = 0
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] = 1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] = 1

    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = 0
                if dp[i-1][j-1] > 1:
                    tot_len += dp[i-1][j-1] * (weight_decay**tot_m)
                    tot_m += 1
    import ipdb
    ipdb.set_trace()
    if dp[len(s1)-1][len(s2)-1] > 1:
        tot_len += dp[len(s1)-1][len(s2)-1] * (weight_decay**tot_m)
    return 1-tot_len/min(len(s1), len(s2))


def lcs_distance_qiji_230403(s1, s2, weight_decay=0.9):
    tot_len = 0
    tot_m = 0
    i_maxes, j_maxes = [0]*len(s1), [0]*len(s2)
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] = 1
            i_maxes[0] =1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] = 1
            j_maxes[0] = 1


    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i] == s2[j]:
                if dp[i-1][j-1] >= i_maxes[i] and dp[i-1][j-1] >= j_maxes[j]:
                    dp[i][j] = dp[i-1][j-1]+1
                    i_maxes[i] = dp[i-1][j-1]+1
                    j_maxes[j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = 0
                if dp[i-1][j-1] > 1:
                    tot_len += dp[i-1][j-1] * (weight_decay**tot_m)
                    tot_m += 1
    if dp[len(s1)-1][len(s2)-1] >= 1:
        tot_len += dp[len(s1)-1][len(s2)-1] * (weight_decay**tot_m)
    return 1-tot_len/min(len(s1), len(s2))


def lcs_distance_qiji1(s1, s2, weight_decay=0.5):
    dp = [[0]*(len(s2)) for _ in range(len(s1))]
    # dp = np.zeros([len(s1), len(s2)], dtype=int)
    for i in range(len(s2)):
        if s1[0] == s2[i]:
            dp[0][i] = 1

    for i in range(len(s1)):
        if s1[i] == s2[0]:
            dp[i][0] = 1

    b_span = 0
    for i in range(1, len(s1)):
        for j in range(1, len(s2)):
            if s1[i]==s2[j]:
                if s1[i-1]==s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i-1][j-1] + 1
                    b_span = i
            else:
                dp[i][j] = max((dp[i-1][j-1], dp[i][j-1], dp[i-1][j]))
                if s1[i-1] == s2[j-1]:
                    m = len(re.findall(r".*?{}.*?".format(s1[b_span: i+1]), str(s1)))
                    dp[i][j] = min(m, max((dp[i-1][j-1], dp[i][j-1], dp[i-1][j])))

    tot_m = max(dp[len(s1)-1])
    # import ipdb
    # ipdb.set_trace()
    # tot_m = dp[len(s1)-1].max()
    tot_len = sum([weight_decay**m for m in range(tot_m)])
    return 1-tot_len/min(len(s1), len(s2))


def cal_hws_distance(parses_pairs, weight_decay=0.95, idx=0):
    """Calculate HWS-distance for all pairs of parses.
    """
    rt_dists = []
    if idx == 0:
        data = tqdm.tqdm(enumerate(parses_pairs), total=len(parses_pairs))
    else:
        data = enumerate(parses_pairs)
    for i, (parse1,parse2) in data:
        seqs = []
        for parse in (parse1, parse2):
            root = seq2dict(parse[1:-1], 4)
            seq = []
            levelorderTraversal(root, seq)
            seqs.append(seq)
        dist = lcs_distance_qiji_230403(seqs[0][1:], seqs[1][1:], weight_decay)
        rt_dists.append(dist)
    return rt_dists


def wrapped(sent1, sent2):
    # with open("./tag_dict.json") as f:
    #     tag_dict = json.load(f)

    stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
                                  use_gpu=False, download_method=None)
    
    parsed1 = stanford_parser(sent1).sentences[0].constituency.__str__()
    parsed2 = stanford_parser(sent2).sentences[0].constituency.__str__()
    return cal_hws_distance([(parsed1,parsed2)], 0.5, idx=0)[0]


if __name__ == "__main__":
    f = open("dataset/robust_sent.json")
    dataset = json.load(f)
    f.close()

    f = open("dataset/carb.json")
    carb = json.load(f)
    f.close()

    carb = [sq["ori_sent"] for sq in carb]
    sentences = [sq["sent"] for sq in dataset["sentList"]]
    clique_info = dataset["cliqueList"]

    with open("./tag_dict.json") as f:
        tag_dict = json.load(f)

    stanford_parser = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency',
                                  use_gpu=True, download_method=None)
    
    robust_parsed, carb_parsed = [], []
    for i, sent in tqdm.tqdm(enumerate(sentences), total=len(sentences)):
        robust_parsed.append(stanford_parser(sent).sentences[0].constituency.__str__())
        # print(f"Finished {i}-th sentence.")
    
    for i, sent in tqdm.tqdm(enumerate(carb), total=len(carb)):
        carb_parsed.append(stanford_parser(sent).sentences[0].constituency.__str__())
        # print(f"Finished {i}-th sentence.")
    
    # parsed = [stanford_parser(sent).sentences[0].constituency.__str__() for sent in sentences]s

    random.seed(2)
    for sample_size in [50, 100, 200, 500, 1272, 4932]:
        demoset = random.sample(range(len(sentences)), sample_size)

        distances = []
        for idx, sent in enumerate(carb_parsed):
            distance = []
            for demo_idx in demoset:
                if idx == clique_info[demo_idx]:
                    continue
                dist = cal_hws_distance([(sent,robust_parsed[demo_idx])], 0.5, idx=0)[0]
                distance.append((dist, demo_idx))
            distances.append(distance)

        f = open(f"mutual/mutual_{sample_size}.json", "w")
        json.dump(distances, f)
        f.close()
        print(f"finished sample size {sample_size}")


    # distances = []
    # for idx, clique in enumerate(dataset):
    #     dist_clq = []
    #     for i, sent1 in enumerate(clique):
    #         parse1 = stanford_parser(sent1).sentences[0].constituency.__str__()
    #         score = 0
    #         for j, rec_sent in enumerate(recomposed[idx]):
    #             if i == j:
    #                 continue
    #             sent2 = rec_sent
    #             parse2 = stanford_parser(sent2).sentences[0].constituency.__str__()

    #         #     # bags_algorithm(k)
    #             distance = cal_hws_distance([(parse1,parse2)], 0.5, idx=0)[0]
    #             # print(i,j)
    #             score += distance
    #         score = score/(len(recomposed[idx])-1)
    #         dist_clq.append(score)

    #     distances.append(dist_clq)
    #     print(f"Finished {idx+1}-th clique")

    # f = open("distances_liedown756.json", "w")
    # json.dump(distances, f, indent=4, ensure_ascii= False)
    # f.close()
