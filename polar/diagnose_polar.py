# One-off: is the polar branch's failure the idea, or the way the branch was built?
# Reuses quick_test.py's set-up, so the train/val images are identical to the seed 1 run.
import sys, time
from pathlib import Path
sys.argv = ["quick_test.py", "--seed", "1"]
src = open(Path(__file__).with_name("quick_test.py")).read().split("# ---- 5. Train all three")[0]
src = src.replace('Path(__file__).resolve().parent.parent', 'Path.home() / "ruby"')
exec(src)

class WiderPolar(nn.Module):
    """Same strip, but sees 0.6 arcsec of radius at once and finds bumps wherever they sit."""
    def __init__(self, ch=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=(9, 7), padding=(4, 0)), nn.BatchNorm2d(cout),
                                 nn.GELU(), nn.AvgPool2d(kernel_size=(1, 2)))
        self.blocks = nn.ModuleList([block(1, ch), block(ch, ch), block(ch, ch)])
        self.radial = nn.Sequential(nn.Conv1d(ch, ch, 9, padding=4), nn.BatchNorm1d(ch), nn.GELU())
        self.out = nn.Linear(ch, 2)
    def forward(self, img):
        x = unwrap(img)
        for b in self.blocks:
            x = b(F.pad(x, (3, 3, 0, 0), mode="circular"))
        x = self.radial(x.mean(dim=3))                # (batch, ch, 80 radius)
        return self.out(x.amax(dim=2))                # strongest response at any radius

t0 = time.time()
polar, a = train(PolarClassifier(), x_train, y_train, x_val, y_val, lr=1e-3, batch=64, patience=5)
print("original polar val", round(a, 3), "%.0fs" % (time.time() - t0), flush=True)
t0 = time.time()
wide, a = train(WiderPolar(), x_train, y_train, x_val, y_val, lr=1e-3, batch=64, patience=5)
print("wider polar val", round(a, 3), "%.0fs" % (time.time() - t0), "params", sum(p.numel() for p in wide.parameters()), flush=True)

# no CNN at all: average the strip around the ring, simple classifier on the 80 numbers
prof = lambda x: unwrap(x).mean(dim=3).flatten(1).numpy()
sc = StandardScaler().fit(prof(x_train))
clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(prof(x_train)), y_train.numpy())
print("profile val", round(roc_auc_score(y_val.numpy(), clf.decision_function(sc.transform(prof(x_val)))), 3), flush=True)

models = {"polar original": lambda x: scores(polar, x), "polar wider": lambda x: scores(wide, x),
          "profile only": lambda x: clf.decision_function(sc.transform(prof(x)))}
def test_cell(single_kind, **fixed):
    settings = [random_settings(**fixed) for _ in raws["test"]]
    d = to_batch([add_light(g, rings("double", s)) for g, s in zip(raws["test"], settings)])
    s_ = to_batch([add_light(g, rings(single_kind, s)) for g, s in zip(raws["test"], settings)])
    y = np.r_[np.ones(len(d)), np.zeros(len(s_))]
    both = torch.cat([d, s_])
    return {n: roc_auc_score(y, f(both)) for n, f in models.items()}
rows = []
for w in TEST_WIDTH:
    for d in TEST_DISTANCE:
        for b in TEST_BRIGHTNESS:
            fx = dict(r1=R, inner=INNER, d=d, b=b, width=w)
            for shape, extra in (("rings", {}), ("arcs", {}), ("round centred", dict(q=1.0, cx=0.0, cy=0.0))):
                for n, auc in test_cell("matched", shape="rings" if shape != "arcs" else "arcs", **extra, **fx).items():
                    rows.append(dict(model=n, width=w, distance=d, brightness=b, shape=shape, auc=auc))
r_ = pd.DataFrame(rows)
for shape in ("rings", "arcs", "round centred"):
    print("\n" + shape)
    print(r_[r_["shape"] == shape].pivot_table(index="distance", columns="model", values="auc").round(3).to_string())
print("\nrings at 1.2-1.3x by thickness")
print(r_[(r_["shape"] == "rings") & (r_.distance > 1.15)].pivot_table(index="width", columns="model", values="auc").round(3).to_string())
