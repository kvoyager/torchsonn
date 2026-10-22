"""Otto baselines: a one hidden layer MLP and XGBoost on the tutorial's data.

Both models see what the SONN of otto_classification.py sees: the same
stratified 70/15/15 split drawn by the seed, log1p and standardized
features, no class weights, and the same score, the unweighted multi-class
log loss on the test split, then accuracy. The seed draws the split and the
model, so a seed's result pairs with the SONN run of the same train.seed.

Run from any directory:
    python tutorials/otto/baselines.py mlp
    python tutorials/otto/baselines.py mlp --hidden 256 --activation silu --weight-decay 1e-4 --dropout 0
    python tutorials/otto/baselines.py xgboost --seeds 42 43
    python tutorials/otto/baselines.py xgboost --max-params 15356 --max-depth 3 --learning-rate 0.3

The MLP defaults are the tuned one: 149 gelu units, AdamW with weight
decay 0.5, dropout 0.2 after the activation; it reads the rows after the
same LayerNorm without parameters as the SONN. It trains with batches of
512, halves the learning rate when the dev log loss stalls for 5 epochs,
stops after 15 epochs without a gain and keeps its best dev epoch.
XGBoost grows trees of depth 8 with learning rate 0.05 (--max-depth,
--learning-rate), stops 50 rounds after the best dev log loss and keeps the
best round. --max-params N keeps the best dev round among those whose trees
hold at most N parameters, a feature and a threshold per split and a value
per leaf. XGBoost is not a torchsonn dependency: `pip install xgboost`.

Each seed logs one line, and the mean over the seeds closes the run.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from otto_classification import load_dataset, split_and_scale  # noqa: E402

logger = logging.getLogger(__name__)

NUM_CLASSES = 9
ACTIVATIONS = {"gelu": torch.nn.GELU, "silu": torch.nn.SiLU, "relu": torch.nn.ReLU,
               "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}


def score(proba, y):
    """Log loss and accuracy of predicted probabilities."""
    return log_loss(y, proba, labels=list(range(NUM_CLASSES))), float((proba.argmax(1) == y).mean())


def fit_mlp(args, seed, x_train, y_train, x_dev, y_dev, x_test, device):
    """Train the MLP with an early stop on the dev log loss.

    Returns the dev and test probabilities of the best dev epoch and the
    parameter count.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = x_train.shape[1]
    layers = [torch.nn.LayerNorm(d, elementwise_affine=False),
              torch.nn.Linear(d, args.hidden), ACTIVATIONS[args.activation]()]
    if args.dropout > 0:
        layers.append(torch.nn.Dropout(args.dropout))
    layers.append(torch.nn.Linear(args.hidden, NUM_CLASSES))
    model = torch.nn.Sequential(*layers).to(device)

    x_tr, y_tr = torch.tensor(x_train, device=device), torch.tensor(y_train, device=device).long()
    x_dv, y_dv = torch.tensor(x_dev, device=device), torch.tensor(y_dev, device=device).long()
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device=device, generator=gen)
        for i in range(0, len(x_tr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dev_loss = loss_fn(model(x_dv), y_dv).item()
        sched.step(dev_loss)
        if dev_loss < best - 1e-5:
            best, bad = dev_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = [torch.softmax(model(torch.tensor(x, device=device)), 1).cpu().numpy() for x in (x_dev, x_test)]
    return proba[0], proba[1], sum(p.numel() for p in model.parameters())


def fit_xgboost(args, seed, x_train, y_train, x_dev, y_dev, x_test, device):
    """Train XGBoost with an early stop on the dev log loss.

    Returns the dev and test probabilities of the best dev round and the
    parameter count: a feature and a threshold per split, a value per leaf,
    over the trees up to that round. With --max-params the best round is
    taken among the rounds whose trees stay within that count.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("XGBoost is not installed: pip install xgboost")
    # Every tree holds at least one value, so a budget bounds the rounds.
    rounds_cap = 5000 if args.max_params is None else min(5000, args.max_params // NUM_CLASSES)
    clf = xgb.XGBClassifier(
        objective="multi:softprob", n_estimators=rounds_cap, learning_rate=args.learning_rate,
        max_depth=args.max_depth, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device=device, early_stopping_rounds=50, eval_metric="mlogloss", random_state=seed)
    clf.fit(x_train, y_train, eval_set=[(x_dev, y_dev)], verbose=False)

    # Parameters of the model cut after each round, NUM_CLASSES trees a round.
    trees = clf.get_booster().trees_to_dataframe()
    per_node = np.where(trees["Feature"] == "Leaf", 1, 2)
    per_round = np.bincount(trees["Tree"].to_numpy() // NUM_CLASSES, weights=per_node)
    params = np.cumsum(per_round).astype(int)
    dev_loss = np.asarray(clf.evals_result()["validation_0"]["mlogloss"])
    allowed = np.ones(len(dev_loss), bool) if args.max_params is None else params[:len(dev_loss)] <= args.max_params
    if not allowed.any():
        raise SystemExit(f"one round of {NUM_CLASSES} trees already exceeds --max-params {args.max_params}")
    rounds = int(np.argmin(np.where(allowed, dev_loss, np.inf))) + 1

    # The arrays are on the CPU; predicting there avoids a copy to the device.
    clf.set_params(device="cpu")
    proba = [clf.predict_proba(x, iteration_range=(0, rounds)) for x in (x_dev, x_test)]
    return proba[0], proba[1], int(params[rounds - 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", choices=("mlp", "xgboost"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    mlp = parser.add_argument_group("mlp")
    mlp.add_argument("--hidden", type=int, default=149)
    mlp.add_argument("--activation", choices=sorted(ACTIVATIONS), default="gelu")
    mlp.add_argument("--weight-decay", type=float, default=0.5)
    mlp.add_argument("--dropout", type=float, default=0.2)
    mlp.add_argument("--lr", type=float, default=1e-3)
    mlp.add_argument("--batch-size", type=int, default=512)
    mlp.add_argument("--epochs", type=int, default=300)
    mlp.add_argument("--patience", type=int, default=15)
    xgboost = parser.add_argument_group("xgboost")
    xgboost.add_argument("--max-depth", type=int, default=8)
    xgboost.add_argument("--learning-rate", type=float, default=0.05)
    xgboost.add_argument("--max-params", type=int, default=None,
                         help="keep the best dev round whose trees hold at most this many parameters")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = fit_mlp if args.model == "mlp" else fit_xgboost
    if args.model == "mlp":
        logger.info("MLP, %d %s units, weight decay %g, dropout %g",
                    args.hidden, args.activation, args.weight_decay, args.dropout)
    else:
        logger.info("XGBoost, depth %d, learning rate %g%s", args.max_depth, args.learning_rate,
                    "" if args.max_params is None else f", at most {args.max_params:,} parameters")

    X, y, feature_names = load_dataset()
    rows = []
    for seed in args.seeds:
        x_train, y_train, x_dev, y_dev, x_test, y_test = split_and_scale(X, y, feature_names, seed)
        dev_proba, test_proba, params = fit(args, seed, x_train, y_train, x_dev, y_dev, x_test, device)
        row = (*score(dev_proba, y_dev), *score(test_proba, y_test), params)
        rows.append(row)
        logger.info("seed %d   dev %.4f / %.4f   test %.4f / %.4f   parameters %s",
                    seed, *row[:4], f"{params:,}")
    mean = np.mean(rows, axis=0)
    logger.info("mean of %d seeds   dev %.4f / %.4f   test %.4f / %.4f",
                len(rows), *mean[:4])


if __name__ == "__main__":
    main()
