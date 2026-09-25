"""Fine-tune Laya on NBA possession decisions. Single GPU.

Adapted from Convai's laya_finetune_typed_decisions_2xT4_kaggle.ipynb:
same model build, same soft-CE + policy-gradient loss, same temperature fit.
Changes: reads finetune/data/{train,calib,test}.jsonl (split by game), one GPU,
per-question class weights so the two rare-"yes" questions are not learned as "no",
and a test scoreboard in the same format as `nba-laya scoreboard`.

    python train.py --data finetune/data --out laya_nba --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options

MODEL_ID = "convaiinnovations/laya"
SUBFOLDER = "multilingual"
# Questions whose positive class is rare. Weight = neg/pos, capped.
# Empty for the current battery: the rarest yes rate is run_continues at 33%.
RARE_YES: set[str] = set()
WEIGHT_CAP = 6.0


def load_rows(path):
    """Read a .jsonl, or the .jsonl.gz next to it if that is what got uploaded."""
    import gzip

    path = str(path)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def build_items(rows, tok, cfg, limit=0, seed=0, log=print):
    """Tokenize rows into per-question items. With a limit, rows are sampled first
    so a 100k-item run does not tokenize the whole season."""
    if limit:
        rows = list(rows)
        random.Random(seed).shuffle(rows)
    items = []
    dropped = 0
    used = 0
    for row in rows:
        if limit and len(items) >= limit:
            break
        used += 1
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                continue
            t = q["type"]
            crit = q.get("criteria", {})
            g = gold[qid]
            if t == "choice":
                keys = list(crit)
                target = [g["probabilities"].get(k, 0.0) for k in keys]
            elif t == "noul":
                target = [g["probabilities"].get("false", 0.5), g["probabilities"].get("true", 0.5)]
            else:
                target = [g["probabilities"].get(str(i), 0.0) for i in range(len(crit))]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            k = len(render_options({"t": t, "crit": crit}))
            seq, markers = build_sequence(
                tok, state, {"t": t, "ins": q["instructions"], "crit": crit}, cfg["max_len"], cfg["head_max_len"]
            )
            if len(markers) != k:
                dropped += 1
                continue
            items.append({
                "ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                "label": int(np.argmax(target)), "qid": qid, "game_id": row["game_id"],
            })
    log(f"built {len(items)} items from {used} of {len(rows)} rows (dropped {dropped})")
    return items


def class_weights(items):
    """Per-item loss weight. Rare-yes questions upweight the positive class."""
    counts = defaultdict(Counter)
    for it in items:
        counts[it["qid"]][it["label"]] += 1
    weights = {}
    for qid, c in counts.items():
        if qid in RARE_YES and c[1] > 0:
            w = min(WEIGHT_CAP, c[0] / c[1])
            weights[qid] = {0: 1.0, 1: w}
        else:
            weights[qid] = {label: 1.0 for label in c}
    return weights


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def forward(model, batch, device):
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits, act = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
        )
    return logits.float(), act


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def predict_logits(model, items, tok, device, batch_size=16):
    model.eval()
    out = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        logits, _ = forward(model, collate(chunk, tok.pad_token_id), device)
        logits = logits.cpu().numpy()
        for r, it in enumerate(chunk):
            out.append(logits[r, : len(it["markers"])])
    return out


def scoreboard(items, logits, temps, log=print):
    """Accuracy, Brier, ECE per question, matching nba-laya scoreboard.

    Regular season and playoffs are scored apart: team ratings mean something
    different once every matchup is two good teams.
    """
    results = {}
    for phase, prefix in (("regular season", "002"), ("playoffs", "004")):
        sel = [(it, z) for it, z in zip(items, logits) if it["game_id"].startswith(prefix)]
        if not sel:
            continue
        log(f"[{phase}]")
        results[phase] = _scoreboard_rows(sel, temps, log)
    return results


def _scoreboard_rows(pairs_in, temps, log):
    per_q = defaultdict(list)
    for it, z in pairs_in:
        z = np.asarray(z, dtype=np.float64) / temps[it["qtype"]]
        p = np.exp(z - z.max())
        p /= p.sum()
        truth = it["label"]
        per_q[it["qid"]].append((float(p[truth]), int(np.argmax(p)) == truth))
    log(f"{'question':<16}{'n':>7} {'acc':>6} {'brier':>6} {'ece':>6}")
    results = {}
    for qid, pairs in sorted(per_q.items()):
        n = len(pairs)
        acc = sum(hit for _, hit in pairs) / n
        brier = sum((1 - p) ** 2 for p, _ in pairs) / n
        ece = 0.0
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            grp = [hit for p, hit in pairs if lo <= p < hi or (b == 9 and p == 1.0)]
            if grp:
                ece += len(grp) / n * abs(sum(grp) / len(grp) - (lo + hi) / 2)
        results[qid] = {"n": n, "accuracy": acc, "brier": brier, "ece": ece}
        log(f"{qid:<16}{n:>7} {acc:6.3f} {brier:6.3f} {ece:6.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="finetune/data")
    ap.add_argument("--out", default="laya_nba")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train items (0 = all)")
    ap.add_argument("--max-eval", type=int, default=20000,
                    help="cap calib and test items; both are still held-out games (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL_ID, allow_patterns=[f"{SUBFOLDER}/*"])
    model_dir = os.path.join(model_dir, SUBFOLDER)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_len"] = 1024
    cfg["head_max_len"] = 256

    train_items = build_items(load_rows(f"{args.data}/train.jsonl"), tok, cfg, limit=args.max_train, seed=args.seed)
    calib_items = build_items(load_rows(f"{args.data}/calib.jsonl"), tok, cfg, limit=args.max_eval, seed=args.seed + 1)
    test_items = build_items(load_rows(f"{args.data}/test.jsonl"), tok, cfg, limit=args.max_eval, seed=args.seed + 2)
    weights = class_weights(train_items)
    print("class weights:", {q: w for q, w in weights.items() if any(v != 1.0 for v in w.values())})

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)

    print("\n== zero-shot on test (same weights laya-serve uses)")
    base_temps = cfg.get("temperature", [1.0, 1.0, 1.0])
    scoreboard(test_items, predict_logits(model, test_items, tok, device), base_temps)

    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW(
        [{"params": enc, "lr": args.lr_encoder}, {"params": head, "lr": args.lr_head}], weight_decay=0.01
    )
    total_updates = max(1, (len(train_items) // (args.micro_batch * args.grad_accum)) * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    GROUP, SIG0, SIG1 = 4, 0.4, 0.1

    print(f"\n== training {len(train_items)} items, {args.epochs} epochs, {total_updates} updates")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_items)
        sigma = SIG0 + (SIG1 - SIG0) * (epoch / max(1, args.epochs - 1))
        epoch_loss, n_batches, step = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(train_items), args.micro_batch):
            chunk = train_items[b : b + args.micro_batch]
            batch = collate(chunk, tok.pad_token_id)
            logits, act = forward(model, batch, device)
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            w = torch.tensor([weights[it["qid"]].get(it["label"], 1.0) for it in chunk], device=device)

            eps = torch.randn((GROUP,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp * w).mean()
            ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
            loss_ce = (ce * w).sum() / w.sum()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            step += 1
            if step % args.grad_accum == 0 or b + args.micro_batch >= len(train_items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            if n_batches % 100 == 0:
                print(f"  epoch {epoch+1} step {n_batches} loss {loss.item()*args.grad_accum:.4f} "
                      f"reward {r.mean().item():.3f} {time.time()-t0:.0f}s")
        print(f"== epoch {epoch+1}/{args.epochs} avg loss {epoch_loss/max(1,n_batches):.4f} ({time.time()-t0:.0f}s)")
        ck = os.path.join(args.out, "checkpoint_latest")
        os.makedirs(ck, exist_ok=True)
        save_file({k_: v.half().contiguous().cpu() for k_, v in model.state_dict().items()},
                  os.path.join(ck, "model.safetensors"))

    print("\n== fitting temperatures on calib split")
    calib_logits = predict_logits(model, calib_items, tok, device)
    temps = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, it["target"]) for it, z in zip(calib_items, calib_logits) if it["qtype"] == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])

    print("\n== fine-tuned on test")
    results = scoreboard(test_items, predict_logits(model, test_items, tok, device), temps)

    os.makedirs(args.out, exist_ok=True)
    save_file({k_: v.half().contiguous().cpu() for k_, v in model.state_dict().items()},
              os.path.join(args.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-nba"
    cfg["temperature"] = temps
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.out, "test_scoreboard.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved to {args.out}/  (point laya-serve at it or laya.load('{args.out}'))")


if __name__ == "__main__":
    main()
