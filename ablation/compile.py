import json

def washtilde():
    f = open("liedown2.txt")
    lines = f.readlines()
    f.close()

    paraphrases = [[]]

    for line in lines:
        if line.startswith("~~~~"):
            paraphrases.append([])
        else:
            line = line.strip()
            paraphrases[-1].append(line)

    paraphrases.pop(-1)
    print(len(paraphrases))

    f = open("liedown2.json", "w")
    json.dump(paraphrases, f, indent=4)
    f.close()


f = open("liedown2.json")
content = json.load(f)
f.close()

content = [x[0] for x in content]

f = open("liedown.json")
pairs = json.load(f)
f.close()

newcontent = []

for idx, sent in enumerate(pairs):
    newcontent.append(
        [content[idx]] + sent
    )

f = open("liedown756.json", "w")
json.dump(newcontent, f, indent=4)
f.close()


