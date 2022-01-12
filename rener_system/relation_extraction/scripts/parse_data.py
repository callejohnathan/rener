import json

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

import random 

msg = Printer()

SYMM_LABELS = ["hasAction", "hasRelatedConcept", "hasAttribute"]
MAP_LABELS = {
    "hasAction": "hasAction",
    "hasRelatedConcept": "hasRelatedConcept",
    "hasAttribute": "hasAttribute"
}


def main(json_loc: Path, train_file: Path, dev_file: Path, test_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    ids = {"train": set(), "dev": set(), "test": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}
    #xxx
    all_data = []
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                neg = 0
                pos = 0
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the GGP entities
                    spans = example["spans"]
                    entities = []
                    span_end_to_start = {}
                    for span in spans:
                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        span_end_to_start[span["token_end"]] = span["token_start"]
                        entities.append(entity)
                        span_starts.add(span["token_start"])
                    doc.ents = entities

                    # Parse the relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        # the 'head' and 'child' annotations refer to the end token in the span
                        # but we want the first token
                        start = span_end_to_start[relation["head"]]
                        end = span_end_to_start[relation["child"]]
                        label = relation["label"]
                        label = MAP_LABELS[label]
                        if label not in rels[(start, end)]:
                            rels[(start, end)][label] = 1.0
                            pos += 1
                        if label in SYMM_LABELS:
                            if label not in rels[(end, start)]:
                                rels[(end, start)][label] = 1.0
                                pos += 1

                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if label not in rels[(x1, x2)]:
                                    neg += 1
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels

                    all_data.append(doc)
                except KeyError as e:
                    #print("error--->", example["text"])
                    msg.fail(e)


    random.shuffle(all_data)
    shuffled_data = all_data[:]
    idx_list = [i for i in range(len(shuffled_data))]

    train_idx = random.sample(idx_list, int(len(idx_list)*0.7))
    not_train_idx = [idx for idx in idx_list if idx not in train_idx]
    dev_idx = random.sample(not_train_idx, int(len(not_train_idx)*0.8))
    eval_idx = [idx for idx in not_train_idx if idx not in dev_idx]

    for idx, doc in enumerate(shuffled_data):
        if(idx in train_idx): 
            docs["train"].append(doc)
        elif(idx in dev_idx):
            docs["test"].append(doc)
        else:
            docs["dev"].append(doc)

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences"
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences"
    )

    docbin = DocBin(docs=docs["test"], store_user_data=True)
    docbin.to_disk(test_file)
    msg.info(
        f"{len(docs['test'])} test sentences"
    )


if __name__ == "__main__":
    typer.run(main)
