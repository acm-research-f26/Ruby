# final_test.py: Step 4. Score the round 2 models on the test set, once, against the pass marks in train_models.py.
#
# Usage (from ~/ruby, with lens_env311):
#   python -I sim/final_test.py
#   python -I sim/final_test.py --part val --pairs 20 --seeds 1 --out final_dryrun     (checks the script on validation)
#
# The pass marks, unchanged from train_models.py:
#   Real compound lenses  rank by P(compound) among the 79 real test lenses. The 4 compound lenses must
#                         average 23 or better over the 3 seeds (luck gets there about 5% of the time),
#                         and be better than guessing (40) on every seed.
#   Ordinary lens-finding AUC of real lens vs. real non-lens on the 379-image test, using
#                         1 - P(not a lens), must stay within 0.01 of the same seed's baseline Zoobot.
#   False alarms          at most 20% of the 75 ordinary real test lenses called compound (P > 0.5), every seed.
#   Detection map         double vs. single AUC on simulated test galaxies, for distance 1.1, 1.2, 1.3, 1.4,
#                         1.6, 1.8 by outer brightness 0.5, 1, 1.5, 2, 3, 4, 6. At least 5 more of the 42
#                         cells must reach AUC 0.8 than the frozen baseline, every seed.
#   Semi-real             real test lens + a simulated outer source vs. the same lens: AUC must beat the
#                         frozen baseline by at least 0.05, every seed.
# What they mean: zoobot3 passing all of them is the main result. If only polar3 passes, the polar idea
# comes back. If neither does, we report the detection map and the gap between simulated and real.
#
# Details the marks left open, fixed here before any test image was scored:
#   - Rank 1 is the highest P(compound). The lens-finding mark must hold on every seed. Beating the baseline is fine.
#   - Each map cell has 300 pairs. Every cell uses the same galaxies and sources; only distance and brightness change.
#   - Semi-real leaves out the 4 compound lenses, because they already have an outer source. Otherwise it is
#     the validation warning check: real lenses with a Euclid radius, 5 versions each, brightness 0.5-6.
#   - Only sim/out/models2 is scored. Round 1's models are not.
#
# Nothing is sent anywhere.
#
# Output (in sim/out/final/):
#   results.txt   the printed report
#   map.csv       every map cell, for every model and seed
#   ranks.csv     P(compound) and rank of all 79 real test lenses, for every model and seed

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate as sim
from simulate import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--part", default="test")
parser.add_argument("--pairs", type=int, default=300)
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--models", default="models2")
parser.add_argument("--data", default="sim/out/v2")
parser.add_argument("--out", default="final")
args = parser.parse_args()

DISTANCES = [1.1, 1.2, 1.3, 1.4, 1.6, 1.8]
BRIGHTNESS = [0.5, 1, 1.5, 2, 3, 4, 6]
NAMES = {"107505": "Galileo's Lens", "726389": "Cosmic Ammonite", "469177": "Cosmic Dartboard", "378903": "Teapot Lens"}
IMG = 224
N_R, N_A, R_MAX = 120, 180, 3.0
device = "mps" if torch.backends.mps.is_available() else "cpu"
out_dir = ROOT / "sim" / "out" / args.out
out_dir.mkdir(parents=True, exist_ok=True)

report = []
def say(line=""):
    print(line, flush=True)
    report.append(line)

start = time.time()
def stage(name):
    print(f"\n[{name}, {(time.time() - start) / 60:.0f} min so far]", flush=True)

# ---- 1. Same preprocessing and models as train_models.py ---------------------------------
def preprocess(pixels):
    x = np.nan_to_num(pixels.astype(np.float32))
    background = np.median(x)
    noise = 1.4826 * np.median(np.abs(x - background)) + 1e-12
    x = np.arcsinh((x - background) / noise)
    lo, hi = np.percentile(x, [1, 99.5])
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1).astype(np.float32)

def to_tensor(images):
    return torch.from_numpy(np.stack([preprocess(x) for x in images]))[:, None]

r = (torch.arange(N_R) + 0.5) * (R_MAX / N_R) / 0.1
t = torch.arange(N_A) * (2 * np.pi / N_A)
GRID = torch.stack([r[:, None] * torch.cos(t) * 2 / 100, r[:, None] * torch.sin(t) * 2 / 100], -1)

def unwrap(img):
    return F.grid_sample(img, GRID.to(img.device).expand(len(img), -1, -1, -1), mode="bilinear", align_corners=False)

class PolarBranch(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=(9, 7), padding=(4, 0)), nn.BatchNorm2d(cout),
                                 nn.GELU(), nn.AvgPool2d(kernel_size=(1, 2)))
        def radial(dilation):
            return nn.Sequential(nn.Conv1d(ch, ch, 9, padding=4 * dilation, dilation=dilation), nn.BatchNorm1d(ch), nn.GELU())
        self.blocks = nn.ModuleList([block(1, ch), block(ch, ch), block(ch, ch)])
        self.radial = nn.Sequential(radial(1), radial(2), radial(4))
        self.head = nn.Linear(2 * ch, 3)
    def forward(self, img):
        x = unwrap(img)
        for b in self.blocks:
            x = b(F.pad(x, (3, 3, 0, 0), mode="circular"))
        x = self.radial(x.mean(dim=3))
        return self.head(torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1))

class Zoobot(nn.Module):
    def __init__(self, num_classes, seed):
        super().__init__()
        self.net = timm.create_model("convnext_nano", pretrained=False, num_classes=num_classes, in_chans=1)
        state = torch.load(ROOT / f"runs/zoobot_lr0.0001_seed{seed}/best.pt", map_location="cpu")
        if num_classes != 2:
            state = {k: v for k, v in state.items() if not k.startswith("head.fc")}
        self.net.load_state_dict(state, strict=False)
    def forward(self, img):
        return self.net(F.interpolate(img, size=(IMG, IMG), mode="bilinear", align_corners=False))

@torch.no_grad()
def run(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 64].to(device)).cpu() for i in range(0, len(x), 64)]).numpy()

def probabilities(model, x):
    return torch.softmax(torch.from_numpy(run(model, x)), 1).numpy()

def auc(yes, no):
    return roc_auc_score(np.r_[np.ones(len(yes)), np.zeros(len(no))], np.r_[yes, no])

# ---- 2. Images, made once so every model and seed sees the same ones ----------------------
stage("Loading real images")
split, lens_models = sim.load_inputs()
part = split[split.split == args.part].reset_index(drop=True)
lenses, non_lenses = part[part.label == 1].reset_index(drop=True), part[part.label == 0].reset_index(drop=True)
load = lambda rows: to_tensor([np.load(ROOT / p) for p in rows.path])
x_lens, x_non = load(lenses), load(non_lenses)
compound = lenses.notes.eq("dspl").values
print(f"  {len(non_lenses)} non-lenses, {len(lenses)} lenses, {compound.sum()} of them compound", flush=True)

stage("Making the detection map images")
hosts = non_lenses
weight = sim.host_weights(split, hosts)
picks = np.random.default_rng(2026).choice(len(hosts), args.pairs, p=weight)
raw_hosts = [np.load(ROOT / hosts.path[i]) for i in picks]
map_single, map_double = None, {}
for b in BRIGHTNESS:
    for d in DISTANCES:
        pairs = [sim.make_pair(raw, np.random.default_rng([2026, i]), lens_models, d=d, b=b) for i, raw in enumerate(raw_hosts)]
        singles = np.stack([p[0] for p in pairs])
        if map_single is None:
            map_single = singles
        assert np.array_equal(singles, map_single)     # same galaxies and sources in every cell
        map_double[b, d] = to_tensor([p[1] for p in pairs])
x_map_single = to_tensor(map_single)

stage("Making the semi-real images")
table = pd.read_csv(ROOT / "data/zenodo_15003116/modeling_lens_mass.csv")
e = np.hypot(table.ell_comps_0_median_pdf.astype(float), table.ell_comps_1_median_pdf.astype(float))
table = pd.DataFrame(dict(id=table.id_str, theta_e=table.einstein_radius_effective_median_pdf.astype(float), q=(1 - e) / (1 + e)))
known = lenses[~compound].merge(table, on="id")
known = known[known.theta_e.between(0.2, 3)]
rng = np.random.default_rng(2027)
semi, plain, semi_b = [], [], []
for _, row in known.iterrows():
    raw = np.nan_to_num(np.load(ROOT / row.path).astype(np.float32))
    _, noise = sim.noise_level(raw)
    for _ in range(5):
        b = np.exp(rng.uniform(np.log(0.5), np.log(6)))
        system = sim.draw_system(rng, lens_models, angle=sim.host_angle(raw), theta_e=row.theta_e, q=row.q, b=b, companion=False)
        _, outer = sim.lensed_light(system)
        semi.append(raw + (b * noise / sim.peak_level(outer)) * outer)
        plain.append(raw)
        semi_b.append(b)
x_semi, x_plain, semi_b = to_tensor(semi), to_tensor(plain), np.array(semi_b)
say(f"Semi-real: {len(known)} real lenses with a Euclid radius, 5 versions each")

data = np.load(ROOT / args.data / f"{args.part}.npz")
x_sim_single, x_sim_double = to_tensor(data["single"]), to_tensor(data["double"])

# ---- 3. Score every model on every seed ----------------------------------------------------
rows_map, rows_rank, seeds = [], [], {}
for seed in args.seeds:
    stage(f"Scoring seed {seed}")
    folder = ROOT / "sim" / "out" / args.models / f"seed{seed}"
    baseline = Zoobot(2, seed).to(device)
    encoder = Zoobot(0, seed).to(device)
    zoobot3 = Zoobot(3, seed)
    zoobot3.load_state_dict(torch.load(folder / "zoobot3.pt", map_location="cpu"))
    zoobot3.to(device)
    polar3 = PolarBranch()
    polar3.load_state_dict(torch.load(folder / "polar3.pt", map_location="cpu"))
    polar3.to(device)
    with open(folder / "frozen.pkl", "rb") as f:
        frozen = pickle.load(f)

    second = {"zoobot3": lambda x: probabilities(zoobot3, x)[:, 2],
              "polar3": lambda x: probabilities(polar3, x)[:, 2],
              "frozen": lambda x: frozen.decision_function(run(encoder, x))}
    result = {}
    base_finding = auc(probabilities(baseline, x_lens)[:, 1], probabilities(baseline, x_non)[:, 1])
    for name, model in [("zoobot3", zoobot3), ("polar3", polar3)]:
        p_lens, p_non = probabilities(model, x_lens), probabilities(model, x_non)
        result[name] = dict(finding=auc(1 - p_lens[:, 0], 1 - p_non[:, 0]), base_finding=base_finding,
                            false_alarms=(p_lens[~compound, 2] > 0.5).mean(), non_called=(p_non[:, 2] > 0.5).mean())
    for name, score in second.items():
        scores = score(x_lens)
        ranks = pd.Series(-scores).rank(method="average").values
        rows_rank += [dict(seed=seed, model=name, id=i, compound=c, score=s, rank=k) for i, c, s, k in zip(lenses.id, compound, scores, ranks)]
        one = result.setdefault(name, {})
        one["ranks"] = ranks[compound]
        one["semi"] = auc(score(x_semi), score(x_plain))
        one["semi_faint"] = auc(score(x_semi)[semi_b < 2], score(x_plain)[semi_b < 2])
        one["semi_clear"] = auc(score(x_semi)[semi_b >= 2], score(x_plain)[semi_b >= 2])
        one["all_pairs"] = auc(score(x_sim_double), score(x_sim_single))
        singles = score(x_map_single)
        cells = {}
        for (b, d), images in map_double.items():
            cells[b, d] = auc(score(images), singles)
            rows_map.append(dict(seed=seed, model=name, b=b, d=d, auc=cells[b, d]))
        one["cells"] = cells
        one["cells_reached"] = sum(v >= 0.8 for v in cells.values())
        if name != "frozen":
            one["mean_p"] = dict(sim_single=score(x_sim_single).mean(), sim_double=score(x_sim_double).mean(),
                                 ordinary=scores[~compound].mean(), semi=score(x_semi).mean(),
                                 real_compound=scores[compound].mean() if compound.any() else np.nan)
    seeds[seed] = result

pd.DataFrame(rows_map).to_csv(out_dir / "map.csv", index=False)
pd.DataFrame(rows_rank).to_csv(out_dir / "ranks.csv", index=False)

# ---- 4. The verdict ------------------------------------------------------------------------
verdict = lambda ok: "PASS" if ok else "FAIL"
marks_passed = {}
for name in ["zoobot3", "polar3"]:
    say(f"\n===== {name} =====")
    passed = {}

    finding = [seeds[k][name]["finding"] - seeds[k][name]["base_finding"] for k in args.seeds]
    say("Ordinary lens-finding, compared with the same seed's baseline: " + ", ".join(
        f"{seeds[k][name]['finding']:.4f} vs. {seeds[k][name]['base_finding']:.4f} ({v:+.4f})" for k, v in zip(args.seeds, finding)))
    passed["Ordinary lens-finding"] = all(v >= -0.01 for v in finding)

    alarms = [seeds[k][name]["false_alarms"] for k in args.seeds]
    say("False alarms, ordinary real lenses called compound: " + ", ".join(f"{v:.0%}" for v in alarms)
        + "; non-lenses called compound: " + ", ".join(f"{seeds[k][name]['non_called']:.0%}" for k in args.seeds))
    passed["False alarms"] = all(v <= 0.2 for v in alarms)

    gain = [seeds[k][name]["cells_reached"] - seeds[k]["frozen"]["cells_reached"] for k in args.seeds]
    say("Detection map, cells of 42 at AUC 0.8 or more: " + ", ".join(
        f"{seeds[k][name]['cells_reached']} vs. frozen {seeds[k]['frozen']['cells_reached']}" for k in args.seeds))
    passed["Detection map"] = all(v >= 5 for v in gain)

    semi_gain = [seeds[k][name]["semi"] - seeds[k]["frozen"]["semi"] for k in args.seeds]
    say("Semi-real AUC: " + ", ".join(f"{seeds[k][name]['semi']:.3f} vs. frozen {seeds[k]['frozen']['semi']:.3f} ({v:+.3f})" for k, v in zip(args.seeds, semi_gain))
        + "; faint " + ", ".join(f"{seeds[k][name]['semi_faint']:.3f}" for k in args.seeds)
        + "; clear " + ", ".join(f"{seeds[k][name]['semi_clear']:.3f}" for k in args.seeds))
    passed["Semi-real"] = all(v >= 0.05 for v in semi_gain)

    if compound.any():
        means = [seeds[k][name]["ranks"].mean() for k in args.seeds]
        say(f"Real compound lenses, mean rank of {len(lenses)}: " + ", ".join(f"{v:.1f}" for v in means) + f", average {np.mean(means):.1f}")
        passed["Real compound lenses"] = np.mean(means) <= 23 and all(v < 40 for v in means)

    for k in args.seeds:
        p = seeds[k][name]["mean_p"]
        say(f"  seed {k} mean P(compound): simulated single {p['sim_single']:.2f}, simulated double {p['sim_double']:.2f}, ordinary real lens {p['ordinary']:.2f}, "
            f"real lens + outer source {p['semi']:.2f}, real compound {p['real_compound']:.2f}; double vs. single, all simulated test pairs {seeds[k][name]['all_pairs']:.3f}")
    say("Verdict:")
    for mark, ok in passed.items():
        say(f"  {verdict(ok)}  {mark}")
    marks_passed[name] = passed

say("\n===== frozen, for reference =====")
if compound.any():
    say("Real compound lenses, mean rank: " + ", ".join(f"{seeds[k]['frozen']['ranks'].mean():.1f}" for k in args.seeds))
say("Double vs. single, all simulated test pairs: " + ", ".join(f"{seeds[k]['frozen']['all_pairs']:.3f}" for k in args.seeds))

if compound.any():
    say("\nEach compound lens, rank per seed:")
    frame = pd.DataFrame(rows_rank)
    frame = frame[frame.compound]
    frame["lens"] = frame.id.str[-6:].map(NAMES)
    say(frame.pivot_table(index="lens", columns=["model", "seed"], values="rank").round(1).to_string())

say("\nDetection map, mean over seeds (rows = outer brightness, columns = distance):")
frame = pd.DataFrame(rows_map)
for name in ["frozen", "zoobot3", "polar3"]:
    say(f"  {name}:\n" + frame[frame.model == name].pivot_table(index="b", columns="d", values="auc").round(3).to_string())

(out_dir / "results.txt").write_text("\n".join(report) + "\n")
print(f"\nDone ({(time.time() - start) / 60:.0f} min). Saved to sim/out/{args.out}/", flush=True)
