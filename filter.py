import json

def washtilde(fanswer):

    f = open(fanswer)
    lines = f.readlines()
    f.close()

    tuplelines = [""]

    for line in lines:
        if line.startswith("~~~~"):
            tuplelines.append("")
        else:
            line = line.strip() + ";"
            tuplelines[-1] += line

    tuplelines.pop(-1)

    f = open("no_tilde.txt", "w")
    for line in tuplelines:
        s = line.find("(")
        e = line.rfind(")")
        if s == -1 or e == -1:
            f.write("\n")
        else:
            f.write(line[s:e+1]+"\n")
    f.close()
    # print(len(tuplelines))

def wash():
    f = open("no_tilde.txt")
    lines = f.readlines()
    f.close()

    NULLSET = set(['NA','None','none','NULL','-','null','_','(',')','\\','[empty]','{','}','N/A','not labeled', '"'])

    result = []
    for idx, line in enumerate(lines):
        rawtup = line.strip().split(";")
        tups = []
        if line.strip() == "":
            result.append([])
            continue
        # print(idx, rawtup)
        for rtup in rawtup:
            rtup = rtup.strip()
            if rtup == "":
                continue
            if rtup[0] != '(' or rtup[-1] != ')':
                s = rtup.find("(")
                e = rtup.rfind(")")
                if s == -1 or e == -1:
                    continue
                else:
                    rtup = rtup[s:e+1]
    #         rtup = rtup.strip()[1:-1]
            tup = rtup[1:-1].split(",")
            tup = [ele.strip() for ele in tup]
            for j in range(len(tup)):
                if tup[j].lower() in NULLSET:
                    tup[j] = ""
            tup = [ele for ele in tup if ele != ""]
            if tup not in tups:
                if len(tup) > 1:
                    tups.append(tup)
        result.append(tups)
            
    # print(len(result), result)
    f = open("chat.json", "w")
    json.dump(result, f, indent=4)
    f.close()

def compose():
    f = open("chat.json")
    ext = json.load(f)
    f.close()

    f = open("carb.json")
    sents = json.load(f)
    f.close()

    gold = []
    extract = []

    for idx in range(len(ext)):
        gold.append(
            {
                "ori_sent": sents[idx]["ori_sent"],
                "ori_args": sents[idx]["ori_args"],
                "id": idx,
                "paraphrases": []
            }
        )
        extract.append(
            {
                "ori_sent": sents[idx]["ori_sent"],
                "ori_args": ext[idx],
                "id": idx,
                "paraphrases": []
            }
        )


    f = open("gold.json", "w")
    json.dump(gold[:300], f, indent=4)
    f.close()

    f = open("result.json", "w")
    json.dump(extract[:300], f, indent=4)
    f.close()

washtilde(fanswer="answer/llama_1272_40.txt")
wash()
compose()