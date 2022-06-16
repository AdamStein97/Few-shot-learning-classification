def create_text_pairs(hypothesis, texts, labels):
    filled_hypothesis = [
        hypothesis.replace("<TEXT>", text).replace("<LABEL>", label)
        for text, label in zip(texts, labels)
    ]

    text_pairs1 = []
    text_pairs2 = []
    for text in filled_hypothesis:
        split_text = text.split("<SEP>")
        text_pairs1.append(split_text[0])
        text_pairs2.append(split_text[1])

    return text_pairs1, text_pairs2
