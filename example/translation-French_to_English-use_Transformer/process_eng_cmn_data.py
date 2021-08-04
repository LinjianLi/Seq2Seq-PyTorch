with open("./eng-cmn.txt", "r", encoding='utf-8') as f:
    lines = []
    for line in f:
        line = line.split("\t")[:2]
        line[1] = " ".join(line[1])
        line = "\t".join(line) + "\n"
        lines.append(line)

with open("./eng-cmn.processed.txt", "w", encoding='utf-8') as o:
    o.writelines(lines)
