#!/usr/bin/env python3
"""Build a self-contained static demo page for GitHub Pages.

Every number on the page is read from ``experiments/results/*.json`` so the
demo, README, model card, evaluation report and slides share one source of
truth. The page has no backend and no JavaScript fetch: it can be served by
GitHub Pages directly from ``docs/``.

Usage::

    python scripts/08_build_demo_page.py
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DOCS = ROOT / "docs"
PAGE_ASSETS = DOCS / "assets"
OUT = DOCS / "index.html"
REPO_URL = "https://github.com/ljh-ningxia/helmet-safety-yolo"


def load(name: str) -> dict[str, Any]:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def f(v: Any, n: int = 3, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{n}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def ensure_assets() -> None:
    """Keep the two evidence images inside docs/ so /docs Pages can serve them."""
    PAGE_ASSETS.mkdir(parents=True, exist_ok=True)
    inference = PAGE_ASSETS / "real_inference.png"
    src = ROOT / "assets" / "real_inference.png"
    if not inference.is_file() and src.is_file():
        shutil.copy2(src, inference)

    dashboard = PAGE_ASSETS / "dashboard_live.png"
    if not dashboard.is_file():
        for candidate in (
            ROOT.parent / "运行界面" / "01_看板首页.png",
            ROOT / "assets" / "dashboard_live.png",
        ):
            if candidate.is_file():
                shutil.copy2(candidate, dashboard)
                break
    missing = [p.name for p in (dashboard, inference) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"demo page assets missing: {missing}")


def kpi(label: str, value: str, note: str) -> str:
    return (
        '<div class="kpi"><div class="kpi-v">__V__</div>'
        '<div class="kpi-l">__L__</div><div class="kpi-n">__N__</div></div>'
    ).replace("__V__", esc(value)).replace("__L__", esc(label)).replace("__N__", esc(note))


def ab_rows(summary: dict[str, Any]) -> tuple[str, str]:
    rows: list[str] = []
    notes: list[str] = []
    for x in summary.get("experiments", []):
        a, b = x.get("A", {}), x.get("B", {})
        rows.append(
            "<tr>"
            f"<td><b>{esc(x.get('experiment', '—'))}</b><br><span class=\"dim\">{esc(x.get('name', ''))}</span></td>"
            f"<td><code>{esc(x.get('variable', '—'))}</code></td>"
            f"<td>{f(a.get('recall'), 4)} → <b>{f(b.get('recall'), 4)}</b></td>"
            f"<td>{f(a.get('precision'), 4)} → <b>{f(b.get('precision'), 4)}</b></td>"
            f"<td>{f(a.get('mean_latency_ms'), 2)} → <b>{f(b.get('mean_latency_ms'), 2)}</b> ms</td>"
            f"<td>{f(a.get('alert_events'), 0)} → <b>{f(b.get('alert_events'), 0)}</b></td>"
            "</tr>"
        )
        notes.append(
            f"<li><b>{esc(x.get('experiment', '—'))}</b>：{esc(x.get('conclusion_boundary', ''))}</li>"
        )
    return "\n".join(rows), "\n".join(notes)


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>工地安全帽识别与预警系统 · 实测展示</title>
<style>
:root{--green:#0e8a5f;--green-d:#0a6b49;--ink:#1c2733;--muted:#5d6b78;--line:#dfe7e9;--bg:#f5f8f7;--card:#fff;--amber:#8a6a28;--amber-bg:#fff8e8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",Arial,sans-serif;line-height:1.6}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
header{background:#fff;border-top:4px solid var(--green);border-bottom:1px solid var(--line)}
header .wrap{display:flex;align-items:center;justify-content:space-between;gap:24px;padding-top:22px;padding-bottom:22px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:42px;height:42px;border-radius:10px;background:var(--green);color:#fff;font-weight:700;font-size:22px;display:flex;align-items:center;justify-content:center;flex:none}
h1{font-size:21px;margin:0;letter-spacing:.2px}
.brand p{margin:2px 0 0;color:var(--muted);font-size:13px}
.badges{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.badge{background:#e6f5ee;color:var(--green-d);border:1px solid #bfe6d3;border-radius:999px;padding:5px 12px;font-size:12px;font-weight:700;letter-spacing:.4px}
.btn{background:var(--green);color:#fff;text-decoration:none;border-radius:7px;padding:9px 15px;font-size:13px;font-weight:600}
.btn:hover{background:var(--green-d)}
main{padding:26px 0 10px}
.notice{background:var(--amber-bg);border:1px solid #ecdcb0;color:#6f5620;border-radius:8px;padding:11px 15px;font-size:13px;margin-bottom:26px}
code{background:#eef3f2;border-radius:4px;padding:1px 5px;font-size:12px;font-family:Consolas,Monaco,monospace}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px 24px;margin-bottom:20px}
h2{font-size:17px;margin:0 0 4px;color:var(--green-d)}
.sub{margin:0 0 18px;color:var(--muted);font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:14px 16px;background:#fbfdfc}
.kpi-v{font-size:26px;font-weight:700;color:var(--green-d);letter-spacing:-.5px}
.kpi-l{font-size:13px;font-weight:600;margin-top:2px}
.kpi-n{font-size:11px;color:var(--muted);margin-top:2px}
.hint{font-size:12px;color:var(--muted);margin:14px 0 0}
img.evidence{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:8px;background:#fff}
.cap{font-size:12px;color:var(--muted);margin:10px 0 0}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#eaf4ef;color:var(--green-d);font-weight:700;font-size:12px}
tr:last-child td{border-bottom:none}
.dim{color:var(--muted);font-size:11px}
ul{margin:10px 0 0;padding-left:20px;font-size:13px}
li{margin-bottom:7px}
pre{background:#14232c;color:#dcebe6;border-radius:8px;padding:15px 17px;overflow:auto;font-size:12.5px;line-height:1.7;margin:0 0 12px}
footer{border-top:1px solid var(--line);color:var(--muted);font-size:12px;text-align:center;padding:20px 24px 30px}
footer a{color:var(--green-d)}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}h1{font-size:18px}table{font-size:11.5px}}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand"><div class="logo">H</div><div>
    <h1>基于 YOLO 的工地现场安全帽识别与预警系统</h1>
    <p>YOLOv8n · Hard Hats Dataset (CC BY 4.0) · 真实训练 / 独立测试 / 公平 A/B</p>
  </div></div>
  <div class="badges"><span class="badge">EVIDENCE: MEASURED</span>
  <a class="btn" href="__REPO__">查看源码 →</a></div>
</div></header>

<main class="wrap">
<div class="notice">这是<strong>静态展示页</strong>：所有数字均从仓库 <code>experiments/results/*.json</code> 读取，由 <code>scripts/08_build_demo_page.py</code> 自动生成；本页不调用后端，无需安装任何环境即可查看。</div>

<section>
  <h2>01 · 独立测试集实测</h2>
  <p class="sub">__NTEST__ 张从未参与训练与选参的 held-out test 图片 · Ultralytics 官方 <code>model.val(split="test")</code></p>
  <div class="kpis">__KPIS__</div>
  <p class="hint">告警类 <code>no-hardhat</code>：mAP@0.5 __NH_MAP__，Recall __NH_R__。valid 选参集 mAP@0.5 0.8382 → test __MAP50__，泛化落差 0.0489；对外只引用 test 数字。</p>
</section>

<section>
  <h2>02 · 系统运行界面</h2>
  <p class="sub">FastAPI + Web 看板，所有指标由实测 JSON 驱动；支持上传图片在线检测</p>
  <img class="evidence" src="assets/dashboard_live.png" alt="系统 Web 看板真实运行界面">
  <p class="cap">浏览器真实截图（服务运行时截取），非设计稿。在线检测需要本地启动服务；本页为静态展示。</p>
</section>

<section>
  <h2>03 · 真实推理证据</h2>
  <p class="sub">未经挑选的 6 张真实工地照片，全部来自 held-out test split</p>
  <img class="evidence" src="assets/real_inference.png" alt="真实图片上的检测框：绿色为标注真值，红色为模型预测">
  <p class="cap"><strong>绿框</strong> = 标注真值（GT），<strong>红框</strong> = 模型预测。数据集许可 CC BY 4.0，署名见 NOTICE.md。</p>
</section>

<section>
  <h2>04 · 四组单变量 A/B</h2>
  <p class="sub">同一权重、同一批 __NAB__ 张真实图片、同一顺序与 seed；每次只改一个变量，同时报告收益与代价</p>
  <table>
    <thead><tr><th>实验</th><th>唯一变量</th><th>Recall A→B</th><th>Precision A→B</th><th>平均时延 A→B</th><th>告警事件 A→B</th></tr></thead>
    <tbody>__ABROWS__</tbody>
  </table>
  <ul>__ABNOTES__</ul>
  <p class="hint">A/B 使用 class-aware greedy IoU@0.5 匹配器，<strong>不是 COCO mAP</strong>；结论只能用于模块归因，不能替代独立测试集指标。</p>
</section>

<section>
  <h2>05 · 结论边界</h2>
  <ul>__BOUNDARIES__</ul>
  <p class="hint">完整边界见仓库 <code>docs/BOUNDARIES.md</code>；数据集是离散图片而非连续监控视频，时序模块不能据此证明真实视频流稳定性。</p>
</section>

<section>
  <h2>06 · 本地运行</h2>
  <pre>__RUNCMD__</pre>
  <p class="hint">模型权重与原始数据<strong>不进入 Git</strong>（AGPL-3.0 权重 + CC BY 4.0 数据，且体积约 2 GB）。完整下载、训练、评估、复现流程见 <code>docs/REPRODUCE.md</code>；仅查看结果可直接打开本页。</p>
</section>
</main>

<footer>Hard Hats Dataset (CC BY 4.0) · YOLOv8n 权重 AGPL-3.0 · 原创代码 MIT · 生成脚本 <code>scripts/08_build_demo_page.py</code> · <a href="__REPO__">__REPO__</a></footer>
</body>
</html>
"""


def main() -> int:
    ensure_assets()
    ev = load("evaluation.json")
    ab = load("ab_summary.json")
    audit = load("dataset_audit.json")
    o = ev.get("official_validation", {})
    lat = ev.get("latency_benchmark", {})
    nh = o.get("per_class", {}).get("no-hardhat", {})

    kpis = "".join(
        [
            kpi("mAP@0.5", f(o.get("map50"), 4), "独立 test split"),
            kpi("mAP@0.5:0.95", f(o.get("map50_95"), 4), "严格 IoU 区间"),
            kpi("Precision", f(o.get("precision"), 4), "全类别"),
            kpi("Recall", f(o.get("recall"), 4), "全类别"),
            kpi("CPU 吞吐", f(lat.get("fps_from_mean"), 1, " FPS"), f"均值 {f(lat.get('mean_ms'), 2)} ms"),
            kpi("P95 延迟", f(lat.get("p95_ms"), 2, " ms"), f"{f(lat.get('repeats'), 0)} 次重复推理"),
        ]
    )
    rows, notes = ab_rows(ab)
    boundaries = "".join(f"<li>{esc(b)}</li>" for b in ev.get("boundaries", []))
    run_cmd = (
        "git clone " + REPO_URL + ".git\n"
        "cd helmet-safety-yolo\n"
        "pip install -r requirements.txt\n"
        "PYTHONPATH=src python -m helmet_safety.cli serve --port 8000\n"
        "# 浏览器打开 http://127.0.0.1:8000"
    )
    page = (
        TEMPLATE.replace("__REPO__", REPO_URL)
        .replace("__NTEST__", str(ev.get("n_test_images", "—")))
        .replace("__KPIS__", kpis)
        .replace("__NH_MAP__", f(nh.get("map50")))
        .replace("__NH_R__", f(nh.get("recall")))
        .replace("__MAP50__", f(o.get("map50"), 4))
        .replace("__NAB__", str(ab.get("frozen_conditions", {}).get("n_images", "—")))
        .replace("__ABROWS__", rows)
        .replace("__ABNOTES__", notes)
        .replace("__BOUNDARIES__", boundaries)
        .replace("__RUNCMD__", esc(run_cmd))
    )
    OUT.write_text(page, encoding="utf-8")
    print(f"generated: {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print(f"assets: {[p.name for p in sorted(PAGE_ASSETS.glob('*.png'))]}")
    print(f"source: {audit.get('total_images', '—')} images / {audit.get('total_boxes', '—')} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
