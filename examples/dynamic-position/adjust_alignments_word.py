import torch

def parse_alignment(line):
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(len(alignments), 2)
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[idx, 0] = int(src_idx)
        parsed_alignment[idx, 1] = int(tgt_idx)
    return parsed_alignment

def recombine_alignment(line):
    fixed_alignment = ""
    for pair in line:
        s = '-'.join([str(pair[0]), str(pair[1])])
        fixed_alignment += s + " "
    return fixed_alignment.strip()

def target_align(elem):
    return elem[1]

with open("train.align_word_adjust", "w") as out_file:
    with open('train.en') as src_file:
        with open('train.align') as align_file:
            for src_line, align_line in zip(src_file, align_file):
                if align_line in ['\n', '\r\n']:
                    align_line = align_line.strip()
                    out_file.write(align_line + '\n')
                else:
                    alignments = parse_alignment(align_line)
                    sentence = []
                    src_line = src_line.strip()
                    src_line = src_line.split(" ")
                    for idx, word in enumerate(src_line):
                        if idx in alignments[:,0]:
                            align_idx = (alignments[:,0] == idx).nonzero()
                            target = alignments[align_idx[0], 1].tolist()
                            source = alignments[align_idx[0],0].tolist()
                            pair = [source[0], target[0], word]
                            sentence.append(pair)
                    sentence.sort(key=target_align)
                    for idx, pair in enumerate(sentence):
                        pair[1] = idx
                    out = recombine_alignment(sentence)
                    out_file.write(out + '\n')


