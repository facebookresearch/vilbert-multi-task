""" Try to generate from BERT """
import ipdb as pdb
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BertForMaskedLM

MASK = "[MASK]"
MASK_ATOM = "[MASK]"

def preprocess(tokens, tokenizer):
    """ Preprocess the sentence by tokenizing and converting to tensor. """
    tok_ids = tokenizer.convert_tokens_to_ids(tokens)
    tok_tensor = torch.tensor([tok_ids]) # pylint: disable=not-callable
    return tok_tensor


def get_seed_sent(seed_sentence, tokenizer, masking, n_append_mask=0):
    """ Get initial sentence to decode from, possible with masks. """

    def get_mask_ids(masking):
        if masking == "none":
            mask_ids = []
        elif masking == "random":
            mask_ids = []
        else:
            mask_ids = [int(d) for d in masking.split(',')]
        return mask_ids

    # Get initial mask
    mask_ids = get_mask_ids(masking)

    # Tokenize, respecting [MASK]
    seed_sentence = seed_sentence.replace(MASK, MASK_ATOM)
    toks = tokenizer.tokenize(seed_sentence)
    for i, tok in enumerate(toks):
        if tok == MASK_ATOM:
            mask_ids.append(i)

    # Mask the input
    for mask_id in mask_ids:
        toks[mask_id] = MASK

    # Append MASKs
    for _ in range(n_append_mask):
        mask_ids.append(len(toks))
        toks.append(MASK)
    mask_ids = sorted(list(set(mask_ids)))

    seg = [0] * len(toks)
    seg_tensor = torch.tensor([seg]) # pylint: disable=not-callable

    return toks, seg_tensor, mask_ids

def load_model(version):
    """ Load model. """
    model = BertForMaskedLM.from_pretrained(version)
    model.eval()
    return model


def predict(model, tokenizer, tok_tensor, seg_tensor, how_select="argmax"):
    """ Get model predictions and convert back to tokens """
    preds = model(tok_tensor, seg_tensor)

    if how_select == "sample":
        dist = Categorical(logits=F.log_softmax(preds[0], dim=-1))
        pred_idxs = dist.sample().tolist()
    elif how_select == "sample_topk":
        raise NotImplementedError("I'm lazy!")
    elif how_select == "argmax":
        pred_idxs = preds.argmax(dim=-1).tolist()[0]
    else:
        raise NotImplementedError("Selection mechanism %s not found!" % how_select)

    pred_toks = tokenizer.convert_ids_to_tokens(pred_idxs)
    return pred_toks

def sequential_decoding(toks, seg_tensor, model, tokenizer, selection_strategy):
    """ Decode from model one token at a time """
    for step_n in range(len(toks)):
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks, tokenizer)
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor, selection_strategy)
        print("\tBERT prediction: %s" % (" ".join(pred_toks)))
        toks[step_n] = pred_toks[step_n]
    return toks

def masked_decoding(toks, seg_tensor, masks, model, tokenizer, selection_strategy):
    """ Decode from model by replacing masks """
    for step_n, mask_id in enumerate(masks):
        print("Iteration %d: %s" % (step_n, " ".join(toks)))
        tok_tensor = preprocess(toks, tokenizer)
        pred_toks = predict(model, tokenizer, tok_tensor, seg_tensor, selection_strategy)
        print("\tBERT prediction: %s\n" % (" ".join(pred_toks)))
        toks[mask_id] = pred_toks[mask_id]
    return toks

def interact(args, model, tokenizer):
    while True:
        raw_str = input(">>> ")
        if raw_str.startswith("CHANGE"):
            _, attr, val = raw_str.split()
            setattr(args, attr, val)
            continue

        toks, seg_tensor, mask_ids = get_seed_sent(raw_str, tokenizer,
                                                   masking=args.masking,
                                                   n_append_mask=args.n_append_mask)
        if args.decoding_strategy == "sequential":
            toks = sequential_decoding(toks, seg_tensor, model, tokenizer, args.token_strategy)
        elif args.decoding_strategy == "masked":
            toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer, args.token_strategy)
        else:
            raise NotImplementedError("Decoding strategy %s not found!" % args.decoding_strategy)

        print("Final: %s" % (" ".join(toks)))

def main(arguments):
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--interact", action="store_true")
    parser.add_argument("--bert_version", default="bert-base-cased",
                        choices=["bert-base-uncased", "bert-base-cased",
                                 "bert-large-uncased", "bert-large-uncased"])

    # How to choose text
    parser.add_argument("--seed_sentence", type=str, default="this is a sentence .")
    parser.add_argument("--masking", type=str,
                        help="Masking strategy: either 'none', 'random', or list of idxs",
                        default="none")
    parser.add_argument("--n_append_mask", type=int, default=0)

    # Decoding
    parser.add_argument("--decoding_strategy", type=str, default="sequential",
                        choices=["masked", "sequential"])
    parser.add_argument("--token_strategy", type=str, default="argmax",
                        choices=["argmax", "sample", "sample_topk"])

    args = parser.parse_args(arguments)

    pdb.set_trace()
    tokenizer = BertTokenizer.from_pretrained(args.bert_version)
    model = load_model(args.bert_version)

    print("Decoding strategy %s, %s at each step" % (args.decoding_strategy, args.token_strategy))
    if args.interact:
        sys.exit(interact(args, model, tokenizer))
    else:
        toks, seg_tensor, mask_ids = get_seed_sent(args.seed_sentence, tokenizer,
                                                   masking=args.masking,
                                                   n_append_mask=args.n_append_mask)

        if args.decoding_strategy == "sequential":
            toks = sequential_decoding(toks, seg_tensor, model, tokenizer, args.token_strategy)
        elif args.decoding_strategy == "masked":
            toks = masked_decoding(toks, seg_tensor, mask_ids, model, tokenizer, args.token_strategy)
        else:
            raise NotImplementedError("Decoding strategy %s not found!" % args.decoding_strategy)

        print("Final: %s" % (" ".join(toks)))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
