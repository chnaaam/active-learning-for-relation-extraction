def decode(sentence:str, tokens:list, labels:list):
    sentence = " 무리한 원가절감도 부메랑이 돼 도요타를 어렵게 만들 수 있다."
    tokens = ['▁무리', '한', '▁원', '가', '절', '감', '도', '▁부', '메', '랑', '이', '▁돼', '▁', '도', '요', '타를', '▁어렵', '게', '▁만들', '▁수', '▁있다', '.']
    labels = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Organization', 'I-Organization', 'E-Organization', 'O', 'O', 'O', 'O', 'O', 'O']
    
    pos_list = decode_label2pos(sentence=sentence, tokens=tokens, labels=labels)
    return decode_pos2char(sentence=sentence, pos_list=pos_list)

def decode_label2pos(sentence:str, tokens:list, labels:list):
    data = []
    character_idx = 0
    start_idx = -1

    for token, label in zip(tokens, labels):
        if token.startswith("▁"):
            token = token.replace("▁", " ")

        len_token = len(token)

        if label.startswith("S-"):
            start_idx = character_idx
            end_idx = character_idx

            data.append({
                "label": label[2:],
                "start_idx": character_idx,
                "end_idx": character_idx
            })

        else:
            if label.startswith("B-"):
                start_idx = character_idx
            elif label.startswith("E-"):
                end_idx = character_idx + len_token

                data.append({
                    "label": label[2:],
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })
            else:
                pass

        character_idx += len_token
    
    return data

def decode_pos2char(sentence: str, pos_list: list):
    data = [{"char": c, "label": "O"} for c in sentence]

    for pos in pos_list:
        for idx in range(pos["start_idx"], pos["end_idx"]):
            data[idx]["label"] = pos["label"]

    return data

if __name__ == "__main__":
    print(decode("", "", ""))