import json
import numpy as np

# 5 shot

def find_idx(shot=6):

    for sample in [50, 100, 200, 1272, 4932]:
        f = open(f"mutual/mutual_{sample}.json")
        distances = json.load(f)
        f.close()

        demo_set = []
        demo_dist = []

        for idx, sent in enumerate(distances):
            sent.sort(key=lambda a: a[0])
            sent = sent[:shot]
            demo_set.append([a[1] for a in sent])
            sent = np.array(sent)
            avg = np.mean(sent[:,0], axis=0)
            demo_dist.append(avg)
            
        f = open(f"demo/demo_{sample}.json", "w")
        json.dump({"demo": demo_set, "distance":demo_dist}, f)
        f.close()


def build_prompt():

    for sample in [50, 100, 200, 1272, 4932]:
        f = open(f"demo/demo_{sample}.json")
        info = json.load(f)
        f.close()

        f = open("dataset/carb.json")
        carb = json.load(f)
        f.close()

        f = open("dataset/robust_sent.json")
        robust = json.load(f)["sentList"]
        f.close()

        info = info["demo"]

        chatgpt = []

        for idx in range(1272):
            sentence = carb[idx]["ori_sent"]
            demo_sents = [robust[demo]["sent"] for demo in info[idx]]
            demo_args = []
            for demo in info[idx]:
                ori_args = ["(" + ", ".join(arg) + ")" for arg in robust[demo]["args"]]
                demo_args.append(";".join(ori_args))
                # print(demo_args)
                # exit()
            
            prompt5 = f"Open information extraction requires the extraction of all relations in the sentence, \
i.e., predicates, the subjects and objects corresponding to these relations, and the possible time and place. \
For example, given the sentence: {demo_sents[0]} From this sentence, the following tuple can be extracted: {demo_args[0]}.\
Given the sentence:{demo_sents[1]} From this sentence, the following tuple can be extracted: {demo_args[1]}. \
Given the sentence:{demo_sents[2]} From this sentence, the following tuple can be extracted: {demo_args[2]}. \
Now could you please extract the following sentence? "+sentence
            
            chatgpt.append(prompt5)
            
            
        f = open(f"prompt/prompt_{sample}.json", "w")
        json.dump(chatgpt, f)
        f.close()


# find_idx()
# build_prompt()

instruction = "Open information extraction requires the extraction of all relations in the sentence, \
i.e., predicates, the subjects and objects corresponding to these relations, and the possible time and place. \
The results should be display in the format of tuples. In these tuples, we always put the predicate first, \
the second is the subject corresponding to the predicate, the third is the object corresponding to the predicate \
(if there is none, it is not labeled), and the last two are time and place, which should be omitted if there is none. \
Please extract information tuples from the following sentences and show the results in one line."

def build_message(shot=3, sample=4932):
    f = open(f"demo_{sample}.json")
    info = json.load(f)
    f.close()

    f = open("carb.json")
    carb = json.load(f)
    f.close()

    f = open("robust_sent.json")
    robust = json.load(f)["sentList"]
    f.close()

    info = info["demo"]

    chatgpt = []

    for idx in range(1272):
        sentence = carb[idx]["ori_sent"]
        demo_sents = [robust[demo]["sent"] for demo in info[idx]]
        demo_args = []

        message = [{"role": "system", "content": "You are a helpful, pattern-following assistant."}]
        message.append({"role": "user","content": instruction})

        # build demo output
        for demo in info[idx]:
            ori_args = ["(" + ", ".join(arg) + ")" for arg in robust[demo]["args"]]
            demo_args.append(";".join(ori_args))

        for i in range(len(info[idx])):
            message.append({"role": "user", "content": "Text: " + demo_sents[i]})
            message.append({"role": "assistant","content": demo_args[i]})
        message.append({"role": "user", "content": "Text: " + sentence})
        
        chatgpt.append(message)
        
        
    f = open(f"prompt_msg_{sample}_{shot}.json", "w")
    json.dump(chatgpt, f)
    f.close()


find_idx(6)
build_prompt()
# for sample in [50, 200, 1272, 4932]:
#     build_message(5, sample)

# import numpy as np

# f = open("demo_4932.json")
# data = json.load(f)["distance"]
# f.close()
# dist = np.array(data)
# dist = np.mean(dist)
# print(dist)
