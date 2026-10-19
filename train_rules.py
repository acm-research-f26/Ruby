# train_rules.py: learn FOLD-SE rules over Zoobot's filter switches, then test them
#
# NeSyFOLD step 3 (Padalkar, Wang & Gupta, AAAI 2024). Reads the tables that
# nesy/extract.py wrote and learns rules like
#     label(X,'lens') :- k445(X,'1'), not k469(X,'1').
# meaning "a lens when filter 445 fires and filter 469 doesn't".
#
# Run it with the Python that has foldse_client (the ExoScope FOLD-SE environment),
# from ~/ruby:
#   FOLD=/Users/nafisur/ExoScope-main/ml/fold/.venv/bin/python
#   $FOLD nesy/train_rules.py --dry-run           # local only: checks the data, shows the upload size
#   $FOLD nesy/train_rules.py                     # rules that predict the TRUE answer
#   $FOLD nesy/train_rules.py --target zoobot     # rules that copy ZOOBOT's answers (explain the model)
#   $FOLD nesy/train_rules.py --top 100           # only the 100 filters that differ most between classes
#   $FOLD nesy/train_rules.py --load              # re-test saved rules without uploading again
#
# Only fit() uses the internet: it sends the training table to Gupta's FOLD-SE server.
# The client replaces column names and values with random codes and encrypts them first.
# Everything after that (testing, explanations) runs on this computer.
#
# Output (next to the tables, e.g. nesy/out/zoobot_lr0.0001_seed1_norm/):
#   rules_<target>.json    the learned rules ("_top100" is added to names when --top is used) (reload with foldse_client.load_model_from_file)
#   rules_<target>.txt     the rules as text, plus the scores below
#   predictions_<target>.csv   rules' answer and Zoobot's answer for every val/test image

import argparse
import csv
import json
import re
from pathlib import Path

from foldse_client import Foldse, get_scores, load_model_from_file, save_model_to_file

ROOT = Path(__file__).resolve().parent.parent   # ~/ruby

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="nesy/out/zoobot_lr0.0001_seed1_norm")   # written by extract.py
parser.add_argument("--target", choices=["truth", "zoobot"], default="truth")
parser.add_argument("--top", type=int, default=0)        # 0 = all filters (as in the paper)
parser.add_argument("--ratio", type=float, default=0.5)  # FOLD-SE settings, same as the ExoScope run
parser.add_argument("--tail", type=float, default=0.005)
parser.add_argument("--timeout", type=int, default=600)  # seconds to wait for the server
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--load", nargs="?", const="saved")  # reuse rules_<target>.json (or a given file)
args = parser.parse_args()

data_dir = ROOT / args.dir
suffix = "_zoobot" if args.target == "zoobot" else ""
tag = args.target + (f"_top{args.top}" if args.top else "")   # names the output files

# ---- 1. Which filters to use ------------------------------------------------------
with open(data_dir / "train_bin.csv") as f:
    all_filters = [c for c in next(csv.reader(f)) if c != "label"]
with open(data_dir / "filters.csv") as f:
    stats = {r["filter"]: r for r in csv.DictReader(f)}
if args.top:
    # half the most lens-typical filters, half the most non-lens-typical (training images only).
    # Ranking by size of difference alone would pick almost only lens-side filters,
    # because lenses switch on far more filters.
    by_diff = sorted(all_filters, key=lambda k: float(stats[k]["difference"]))
    picked = set(by_diff[-(args.top - args.top // 2):]) | set(by_diff[:args.top // 2])
    filters = sorted(picked, key=lambda k: int(k[1:]))
else:
    filters = all_filters

# ---- 2. Load the tables (local) ----------------------------------------------------
model = Foldse(str_attrs=filters, num_attrs=[], label="label", pos_val="lens", timeout=args.timeout)
train = model.load_data(data_dir / f"train_bin{suffix}.csv")
val = model.load_data(data_dir / f"val_bin{suffix}.csv")
test = model.load_data(data_dir / f"test_bin{suffix}.csv")
rules_file = data_dir / f"rules_{tag}.json"

# images.csv is in the same row order as the tables: train, then val, then test
with open(data_dir / "images.csv") as f:
    images = list(csv.DictReader(f))
info = {s: [r for r in images if r["split"] == s] for s in ("train", "val", "test")}
assert len(info["train"]) == len(train) and len(info["test"]) == len(test), "tables and images.csv disagree"

n_pos = sum(r["label"] for r in train)
print(f"training rows: {len(train)} ({n_pos} '{'lens' if args.target == 'truth' else 'Zoobot says lens'}', "
      f"{len(train) - n_pos} not) | filters: {len(filters)} | target: {args.target}")

if args.dry_run:
    # build exactly what fit() would send, without sending it
    from foldse_client.remote import _Obfuscator
    ob = _Obfuscator(filters, [], "label", "lens")
    body = json.dumps({"config": ob.obfuscate_config(),
                       "data": [ob.obfuscate_row(r) for r in train]}, separators=(",", ":"))
    mb = len(body) * 4 / 3 / 1e6          # encrypted and base64-encoded: about 4/3 larger
    print(f"dry run: fit() would upload about {mb:.0f} MB. Nothing was sent.")
    if mb > 20:
        print("  that is large for the server; if fit() fails, try --top 100")
    raise SystemExit

# ---- 3. Learn the rules (this step uses the internet) -------------------------------
if args.load:
    rules_file = rules_file if args.load == "saved" else Path(args.load)
    model = load_model_from_file(rules_file)
    print(f"Loaded rules from {rules_file} (nothing sent)")
else:
    print("Sending the encoded training table to the FOLD-SE server...")
    print("(the first run can take 30+ seconds while the server wakes up)\n")
    try:
        model.fit(train, ratio=args.ratio, tail=args.tail)
    except Exception as e:
        raise SystemExit(f"fit() failed: {e}\nIf the upload is too big or timed out, try --top 100 "
                         f"or a longer --timeout.")
    save_model_to_file(model, rules_file)
rules = model.asp()

# ---- 4. Test on this computer -------------------------------------------------------
def evaluate(split, rows):
    rules_pred = model.predict(rows)
    truth = [r["label"] == "1" for r in info[split]]
    zoobot = [r["zoobot_pred"] == "1" for r in info[split]]
    acc, prec, rec, f1 = get_scores(rules_pred, truth)
    zoobot_acc = sum(z == t for z, t in zip(zoobot, truth)) / len(truth)
    fidelity = sum(p == z for p, z in zip(rules_pred, zoobot)) / len(truth)   # agrees with Zoobot
    return rules_pred, acc, prec, rec, zoobot_acc, fidelity

used = sorted({k for line in rules for k in re.findall(r"\b(k\d+)\(", line)}, key=lambda k: int(k[1:]))
report = ["LEARNED RULES:"] + ["  " + r for r in rules]
report.append(f"\n{len(rules)} rules using {len(used)} filters: {', '.join(used)}")
for k in used:
    s = stats[k]
    report.append(f"  {k:>5s}  ON for {float(s['on_rate_lenses']):4.0%} of training lenses, "
                  f"{float(s['on_rate_nonlenses']):4.0%} of training non-lenses")

predictions = {}
for split, rows in (("val", val), ("test", test)):
    pred, acc, prec, rec, zoobot_acc, fidelity = evaluate(split, rows)
    predictions[split] = pred
    report.append(f"\n{split.upper()} ({len(rows)} images, scored against the TRUE answer):")
    report.append(f"  rules accuracy   {acc:6.1%}   (Zoobot at score > 0.5: {zoobot_acc:.1%})")
    report.append(f"  rules precision  {prec:6.1%}   of the ones the rules called lenses, how many were")
    report.append(f"  rules recall     {rec:6.1%}   of the real lenses, how many the rules found")
    report.append(f"  fidelity         {fidelity:6.1%}   how often the rules agree with Zoobot")
print("\n".join(report))
(data_dir / f"rules_{tag}.txt").write_text("\n".join(report) + "\n")

with open(data_dir / f"predictions_{tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "split", "label", "notes", "zoobot_score", "zoobot_pred", "rules_pred"])
    for split in ("val", "test"):
        for r, p in zip(info[split], predictions[split]):
            w.writerow([r["id"], split, r["label"], r["notes"], r["zoobot_score"], r["zoobot_pred"], int(p)])

# ---- 5. Why? The compound lenses, plus one normal lens and one non-lens -------------
print("\nWHY - explanations for test images (the 4 compound lenses first):")
examples = [i for i, r in enumerate(info["test"]) if r["notes"] == "dspl"]
examples.append(next(i for i, r in enumerate(info["test"]) if r["label"] == "1" and r["notes"] != "dspl"))
examples.append(next(i for i, r in enumerate(info["test"]) if r["label"] == "0"))
for i in examples:
    r = info["test"][i]
    kind = "COMPOUND LENS" if r["notes"] == "dspl" else ("LENS" if r["label"] == "1" else "NON-LENS")
    print(f"\n--- {kind} {r['id']}: Zoobot score {float(r['zoobot_score']):.3f}, "
          f"rules say {'lens' if predictions['test'][i] else 'not a lens'} ---")
    for line in model.proof_trees(test[i]):
        print(line)

print(f"\nSaved rules_{tag}.json, rules_{tag}.txt and predictions_{tag}.csv "
      f"in {args.dir}/")
