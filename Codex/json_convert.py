import json

f = open("Names.json")

data = json.load(f)

l = []

for keys in data:
    l.append(data[keys])

print(l)

with open("Names.txt",'w', encoding='utf-8') as f:
    for items in l:
        f.write("%s\n" % items)
    print("Done")