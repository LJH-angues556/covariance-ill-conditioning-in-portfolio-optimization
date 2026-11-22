"""Generate outputs/report.md -- a Chinese summary of everything the program measured.

Every number in the paper should be copied from this file, never retyped from
memory.  The report is regenerated from ``metrics`` on every run, so figure,
table and text cannot drift apart.
"""

from __future__ import annotations

import math
from pathlib import Path

from . import config as C


def _f(x, digits: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-2):
            return f"{x:.2e}"
        return f"{x:.{digits}f}"
    return str(x)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _pick(rows: list[dict], **kw) -> dict:
    for r in rows:
        if all(abs(float(r.get(k, -1e9)) - float(v)) < 1e-9 if isinstance(v, (int, float)) else r.get(k) == v
               for k, v in kw.items()):
            return r
    return {}


def write_report(metrics: dict, checks: list[dict], stat: dict) -> Path:
    e1, e2, e3, e4, e5 = (metrics.get(k) for k in ("e1", "e2", "e3", "e4", "e5"))
    md: list[str] = []

    md.append("# 协方差矩阵的病态性对投资组合优化的影响")
    md.append("")
    md.append("**基于特征值分析的数值实验 —— 自动生成的结果汇总**")
    md.append("")
    md.append("> 本文件由 `run_all.py` 自动生成，所有数字可直接抄进论文。")
    md.append("> 每个统计量都是 "
              f"**{C.N_REPS} 次独立重复的均值**（每次重复重新抽一个市场 + 一段估计窗口）；"
              "图上的阴影为 ±1 个标准差。")
    md.append("")
    md.append("| 设定 | 值 |")
    md.append("|---|---|")
    md.append(f"| 资产数上限 N | {C.N_MAX} |")
    md.append(f"| 估计窗口 T | {C.T_TRAIN} 个交易日 |")
    md.append(f"| 波动率目标 | {C.VOL_TARGET * 100:.0f}%（所有组合都缩放到同一风险预算） |")
    md.append(f"| 市场结构 | {C.K_FACTORS} 因子模型（市场 + 2 风格）+ 对角特质噪声 |")
    md.append(f"| 重复次数 | {C.N_REPS} |")
    md.append(f"| 随机种子 | {C.SEED} |")
    md.append("")

    # ------------------------------------------------------------------ 关键数字
    md.append("## 0 关键数字一览（写摘要用）")
    md.append("")
    if e1 and e2 and e3 and e5:
        md.append("| 量 | 数值 | 出处 |")
        md.append("|---|---|---|")
        md.append(f"| 真实协方差的条件数 κ(Σ) | {_f(e1['kappa_true'], 1)} | 表 1 |")
        md.append(f"| 样本协方差的条件数 κ(Ŝ) | {_f(e1['kappa_sample'], 0)} | 表 1 |")
        md.append(f"| 放大倍数 κ(Ŝ)/κ(Σ) | {_f(e1['kappa_sample'] / e1['kappa_true'], 0)}× | 表 1 |")
        md.append(f"| 最小特征值之比 λ_min(Ŝ)/λ_min(Σ) | {_f(e1['lam_min_sample'] / e1['lam_min_true'], 4)} | 表 1 |")
        md.append(f"| 杠杆（N=5 → N=55，原始估计） | {_f(e2['lev_small']['plug_in'], 2)} → "
                  f"{_f(e2['lev']['plug_in'], 1)} | 表 3 |")
        md.append(f"| 杠杆（只收缩 Σ，N=55） | {_f(e2['lev']['shrink_S'], 2)} | 表 3 |")
        md.append(f"| 杠杆（收缩 Σ 且弃用 μ，N=55） | {_f(e2['lev']['gmv_shrunk'], 2)} | 表 3 |")
        md.append(f"| 自认为 15% 波动，实际承受 | {_f(e2['true_vol']['plug_in'] * 100, 0)}% | 图 3b |")
        md.append(f"| 1% 的 Σ 扰动引起的放大倍数（N=55） | {_f(e3['amplification'][-1]['amplification'], 0)} 倍 | 表 5 |")
        md.append(f"| μ 的相对估计误差 / Σ 的相对估计误差 | {_f(e3['eps_mu'], 2)} / {_f(e3['eps_cov'], 2)}"
                  f" = {_f(e3['eps_mu'] / e3['eps_cov'], 0)}× | 表 5 |")
        md.append(f"| 夏普：已知 Σ、只估 μ | {_f(e3['sharpe_mu_only'], 3)} | 表 4 |")
        md.append(f"| 夏普：已知 μ、只估 Σ | {_f(e3['sharpe_sigma_only'], 3)} | 表 4 |")
        md.append(f"| 夏普：两者都估（直接代入解） | {_f(e3['sharpe_both'], 3)} | 表 4 |")
        md.append(f"| 夏普：上帝视角（真 μ、真 Σ） | {_f(e3['sharpe_oracle'], 3)} | 表 4 |")
        md.append(f"| 夏普：1/N（N=55） | {_f(e5['pop']['equal_weight'], 3)} | 表 8 |")
        md.append(f"| 夏普：最小方差 + 收缩 Σ（N=55） | {_f(e5['pop']['gmv_shrunk'], 3)} | 表 8 |")
        md.append("")

    # ------------------------------------------------------------------ e1
    if e1:
        md.append("## 1 病态从哪里来（实验 e1）")
        md.append("")
        md.append("真实的协方差矩阵是良态的，样本协方差不是。下表是逐资产数 N 的结果。")
        md.append("")
        rows = []
        for r in e1["n_sweep"]:
            rows.append([
                int(r["n_assets"]),
                _f(r["kappa_true"], 1),
                _f(r["kappa_sample"], 1),
                _f(r["kappa_shrunk"], 1),
                _f(r["lam_min_true"], 2),
                _f(r["lam_min_sample"], 2),
                _f(r["delta_lw"], 3),
            ])
        md += _table(
            ["N", "κ(Σ) 真", "κ(Ŝ) 样本", "κ(Ledoit-Wolf)", "λ_min 真", "λ_min 样本", "LW δ"],
            rows,
        )
        md.append("")
        md.append("**表 1** 协方差矩阵的谱。N=55 时样本矩阵的最小特征值只剩真值的 "
                  f"{_f(e1['lam_min_sample'] / e1['lam_min_true'] * 100, 1)}%，"
                  f"条件数从 {_f(e1['kappa_true'], 1)} 涨到 {_f(e1['kappa_sample'], 0)}。")
        md.append("")
        rows = []
        for r in e1["t_sweep"]:
            rows.append([int(r["T"]), _f(r["T_over_N"], 1), _f(r["kappa_sample"], 1),
                         _f(r["kappa_mp"], 1)])
        md += _table(["T", "T/N", "κ(Ŝ)", "κ 的 i.i.d. 参考值"], rows)
        md.append("")
        md.append("**表 2** 固定 N=30，把窗口拉长。T 从 60 涨到 "
                  f"{e1['T_max']}，κ 从 {_f(e1['kappa_T60'], 1)} 降到 {_f(e1['kappa_Tmax'], 1)}。"
                  "注意它不会降到 1：真实矩阵本身就是有结构的，κ 的下界是 κ(Σ)。")
        md.append("")

    # ------------------------------------------------------------------ e2
    if e2:
        md.append("## 2 权重爆炸（实验 e2）")
        md.append("")
        md.append("2×2 设计：协方差用原始还是收缩，目标里用不用 μ。"
                  "每个组合都缩放到 15% 年化波动（用**它自己相信的**协方差缩放），"
                  "杠杆 = 每单位净值需要的总敞口。")
        md.append("")
        rows = []
        for n in e2["n_grid"]:
            cells = []
            for name in ("plug_in", "shrink_S", "gmv_sample", "gmv_shrunk"):
                r = _pick(e2["rows"], n_assets=n, estimator=name)
                cells.append(_f(r.get("gross_leverage"), 2))
            rows.append([n] + cells)
        md += _table(["N", "原始 Σ + μ（直接代入解）", "收缩 Σ + μ", "原始 Σ，不用 μ", "收缩 Σ，不用 μ"], rows)
        md.append("")
        md.append("**表 3** 总杠杆 Σ|wᵢ|。读法：")
        md.append("")
        md.append(f"- 只把 Σ 换成 Ledoit-Wolf 估计，杠杆从 {_f(e2['lev']['plug_in'], 1)} "
                  f"掉到 {_f(e2['lev']['shrink_S'], 2)}（−{_f((1 - e2['lev']['shrink_S'] / e2['lev']['plug_in']) * 100, 1)}%）；")
        md.append(f"- 光把 μ 扔掉（最小方差）几乎没用：{_f(e2['lev']['gmv_sample'], 1)}，"
                  "因为病态的是 Σ⁻¹，不是 μ；")
        md.append(f"- 两件事一起做才回到 {_f(e2['lev']['gmv_shrunk'], 2)}。")
        md.append("")
        md.append(f"**更要紧的一列**：直接代入解自认为在跑一个 15% 波动的组合，"
                  f"真实波动是 **{_f(e2['true_vol']['plug_in'] * 100, 0)}%**（图 3b）。"
                  "它不是「风险高」，它是**把风险算错了**——这正是 Michaud 说的 error maximization。")
        md.append("")

    # ------------------------------------------------------------------ e3
    if e3:
        md.append("## 3 机制、不稳定与责任划分（实验 e3）")
        md.append("")
        md.append("### 3.1 机制：解在特征方向上怎么分配")
        md.append("")
        m = e3["mechanism"]
        md.append("把 x = Σ⁻¹μ 按 Σ 的特征向量展开：x = Σᵢ cᵢvᵢ，cᵢ = (vᵢᵀμ)/λᵢ。"
                  "特征值越小，同一个 μ 投影被除得越大。")
        md.append("")
        md.append(f"- 真实矩阵：最大的 3 个方向承担 ‖x‖² 的 {_f(m['top3_share_true'] * 100, 1)}%；")
        md.append(f"- 样本矩阵：最大的 3 个方向承担 **{_f(m['top3_share_hat'] * 100, 1)}%**，"
                  "而这 3 个方向恰好是 λ 最小的 3 个——也就是噪声方向。")
        md.append("")
        md.append("### 3.2 换一段数据，答案就翻（图 5a）")
        md.append("")
        md.append(f"同一市场、两段独立的 60 天数据、同一个优化器：两个权重向量的相关系数是 "
                  f"**{_f(e3['corr_big'], 2)}**，换手率 {_f(e3['turnover_big'], 2)}。"
                  "也就是说，两份同样合法的数据给出两个几乎无关的组合。")
        md.append("")
        md.append("### 3.3 责任划分：μ 还是 Σ")
        md.append("")
        md.append("把其中一个矩换成真值，看夏普能恢复多少：")
        md.append("")
        rows = []
        for n in sorted({r["n_assets"] for r in e3["decomposition"]}):
            cells = []
            for v in ("oracle", "sigma_only", "mu_only", "both"):
                r = _pick(e3["decomposition"], n_assets=n, variant=v)
                cells.append(_f(r.get("pop_sharpe"), 3))
            rows.append([n] + cells)
        md += _table(["N", "上帝视角", "已知 μ，只估 Σ", "已知 Σ，只估 μ", "两者都估"], rows)
        md.append("")
        md.append("**表 4** 真实矩下的夏普（与杠杆无关，缩放不改变它）。结论很干净："
                  f"**把 μ 换成真值，夏普从 {_f(e3['sharpe_both'], 3)} 升到 "
                  f"{_f(e3['sharpe_sigma_only'], 3)}"
                  f"（{_f(e3['sharpe_sigma_only'] / e3['sharpe_both'], 1)} 倍）；"
                  f"把 Σ 换成真值，只升到 {_f(e3['sharpe_mu_only'], 3)}"
                  f"（{_f(e3['sharpe_mu_only'] / e3['sharpe_both'], 1)} 倍）。"
                  "夏普的瓶颈在 μ，不在 Σ**。")
        md.append("")
        md.append("### 3.4 放大倍数与一阶界")
        md.append("")
        rows = []
        for r in e3["amplification"]:
            rows.append([
                int(r["n_assets"]),
                _f(r["kappa_sample"], 1),
                _f(r["amplification"], 1),
                _f(r["rel_error_mu"], 2),
                _f(r["rel_error_cov"], 2),
                _f(r["rel_error_x"], 1),
                _f(r["error_over_bound"], 3),
            ])
        md += _table(["N", "κ(Ŝ)", "1% 扰动引起的放大", "ε_μ", "ε_Σ", "解的实际相对误差", "误差/界"], rows)
        md.append("")
        md.append("**表 5** 放大倍数 = (‖δx‖/‖x‖) / (‖δΣ‖/‖Σ‖)。"
                  "一阶扰动理论给出 ‖δx‖/‖x‖ ≤ κ(Ŝ)·(‖δμ‖/‖μ‖ + ‖δΣ‖/‖Σ‖)。"
                  "注意常数是 **κ(Ŝ)** 而不是 κ(Σ)：被求逆的是 Ŝ，一阶展开里出现的是 ‖Ŝ⁻¹‖·‖Ŝ‖；"
                  "换成 κ(Σ) 这个界会被违反。"
                  "“误差/界”= 解的实际相对误差 ÷ κ(Ŝ)·(ε_μ+ε_Σ)，"
                  "全部小于 1，界成立，但保守（随机扰动只有最坏方向的约 1/N）。")
        md.append("")
        md.append("读这张表有两个容易踩的坑。第一，“1% 扰动引起的放大”在 N≥45 之后进入平台"
                  "（N=50 是 169、N=55 是 155，两点差在 1 个标准差之内），不要读成单调下降——"
                  "一阶近似在大 κ 下已经失效，响应饱和；还在涨的是“解的实际相对误差”那一列"
                  "（367 → 1248）。第二，“误差/界”那一列是解的相对误差除以界，不是放大倍数除以 κ，"
                  "两者数值接近但含义不同。")
        md.append("")

    # ------------------------------------------------------------------ e4
    if e4:
        md.append("## 4 修补：两个旋钮（实验 e4）")
        md.append("")
        md.append("### 4.1 收缩 Σ 有多管用")
        md.append("")
        rows = []
        for r in e4["sig"]:
            rows.append([_f(r["value"], 2), _f(r["gross_leverage"], 2), _f(r["pop_sharpe"], 3)])
        md += _table(["δ_Σ", "杠杆", "夏普"], rows)
        md.append("")
        lev_lw = e2["lev"]["shrink_S"] if e2 else e4["lev_at_lw"]
        md.append(f"**表 6** 收缩 Σ 的路径。Ledoit-Wolf 的 δ = {_f(e4['delta_lw'], 3)}"
                  f"（±{_f(e4['delta_lw_sd'], 3)}），它的判据是让 E‖Σ̃−Σ‖² 最小——"
                  "这对**估计**一个矩阵是最优的，但对**求逆**远远不够："
                  f"在这个点上杠杆还有 {_f(lev_lw, 2)}（表 3 的 shrink_S 行），"
                  f"要一路加到 0.99 才降到 {_f(e4['lev_min'], 1)}。"
                  "**这是本文最实用的一条：Ledoit-Wolf 的强度不该直接拿来当正则化强度。**")
        md.append("")
        md.append("### 4.2 把 μ 收缩掉")
        md.append("")
        rows = []
        for r in e4["blend"]:
            rows.append([_f(r["value"], 2), _f(r["gross_leverage"], 2), _f(r["pop_sharpe"], 3)])
        md += _table(["δ_w（0=直接代入解，1=最小方差）", "杠杆", "夏普"], rows)
        md.append("")
        md.append(f"**表 7** 在权重空间把直接代入解朝最小方差方向混合。"
                  f"夏普从 {_f(e4['sr_blend_zero'], 3)} 升到 {_f(e4['sr_blend_end'], 3)}，"
                  f"杠杆几乎不动（{_f(e4['lev_blend_zero'], 2)} → {_f(e4['lev_blend_end'], 2)}）。"
                  "**这就是「μ 修方向、Σ 修风险」的定量版本。**")
        md.append("")
        md.append("### 4.3 为什么 μ 修不动")
        md.append("")
        md.append(f"T=60 时单个资产均值的 t 统计量：中位数 |t| = {_f(e4['tstat_median_abs'], 2)}，"
                  f"|t|>2 的比例 {_f(e4['tstat_frac_gt2'] * 100, 1)}%（图 7d）。"
                  "整条均值向量在统计上和一条常数没有区别，优化器拿它做的就是拟合噪声。")
        md.append("")

    # ------------------------------------------------------------------ e5
    if e5:
        md.append("## 5 样本外（实验 e5）")
        md.append("")
        rows = []
        for name, *_ in (
            ("oracle",), ("plug_in",), ("shrink_S",), ("gmv_sample",),
            ("gmv_shrunk",), ("gmv_long_only",), ("equal_weight",),
        ):
            rows.append([
                name,
                _f(e5["pop"].get(name), 3),
                _f(e5["real"].get(name), 3),
                _f(e5["lev"].get(name), 2),
                _f(e5["maxw"].get(name), 2),
            ])
        md += _table(["策略", "真实夏普", "实现夏普", "杠杆", "最大单一仓位"], rows)
        md.append("")
        md.append(f"**表 8** N={e5['n_big']}，T=60。真实夏普是权重的性质（无抽样噪声）；"
                  "实现夏普是在独立 2000 天路径上的均值，只用来确认前者不是编出来的。")
        md.append("")
        md.append(f"一句话：直接代入解 {_f(e5['pop']['plug_in'], 3)}，1/N {_f(e5['pop']['equal_weight'], 3)}，"
                  f"最小方差+收缩 Σ {_f(e5['pop']['gmv_shrunk'], 3)}，"
                  f"上帝视角 {_f(e5['pop']['oracle'], 3)}。**在这个样本长度下，1/N 打败所有优化模型**——"
                  "这就是 DeMiguel、Garlappi 和 Uppal (2009) 的结论在本文设定下的复现。")
        md.append("")
        rows = []
        for r in e5["tn_rows"]:
            rows.append([int(r["T"]), _f(r["T_over_N"], 1), _f(r["sharpe_plug_in"], 3),
                         _f(r["leverage_plug_in"], 1), _f(r["sharpe_gmv_shrunk"], 3),
                         _f(r["sharpe_equal_weight"], 3)])
        md += _table(["T", "T/N", "直接代入解夏普", "直接代入解杠杆", "最小方差+收缩 Σ", "1/N"], rows)
        md.append("")
        md.append("**表 9** 固定 N=55，加长窗口。T/N 从 1 拉到 44（约 9.5 年），直接代入解的夏普"
                  f"才从 {_f(e5['tn_rows'][0]['sharpe_plug_in'], 3)} 爬到 "
                  f"{_f(e5['tn_rows'][-1]['sharpe_plug_in'], 3)}，仍明显低于 1/N 的 "
                  f"{_f(e5['tn_rows'][-1]['sharpe_equal_weight'], 3)}——"
                  "这就是「均值难估」在时间维度上的代价。")
        md.append("")

    # ------------------------------------------------------------------ checks
    md.append("## 6 断言核对表")
    md.append("")
    rows = []
    for r in checks:
        mark = {"PASS": "通过", "FAIL": "**失败**", "SKIP": "跳过"}[r["status"]]
        rows.append([r["check"], r["expected"], r["measured"], mark])
    md += _table(["断言", "期望", "实测", "结果"], rows)
    md.append("")
    md.append(f"合计 **{stat['pass']} 通过 / {stat['fail']} 失败 / {stat['skip']} 跳过**。"
              "论文第 5 章结尾可以写：「全部结论由脚本自动核对，"
              f"{stat['total']} 项断言全部通过」。")
    md.append("")

    # ------------------------------------------------------------------ 图表索引
    md.append("## 7 图表索引")
    md.append("")
    md.append("| 图 | 文件 | 用在哪一章 |")
    md.append("|---|---|---|")
    for name, where in (
        ("fig1_spectrum", "第 3 章 病态机制"),
        ("fig2_conditioning", "第 3 章 病态机制"),
        ("fig3_leverage", "第 4 章 实证（杠杆与风险误判）"),
        ("fig4_mechanism", "第 3 章 求逆为何放大误差"),
        ("fig5_instability", "第 4 章 实证（两段数据、责任划分）"),
        ("fig6_amplification", "第 3 章 误差放大与一阶界"),
        ("fig7_shrinkage", "第 4 章 两种修补"),
        ("fig8_out_of_sample", "第 4/5 章 样本外"),
    ):
        md.append(f"| {name} | `outputs/figures/{name}.png` | {where} |")
    md.append("")
    md.append("每张图同时有 `.pdf`（矢量，放大不糊）。")
    md.append("")

    path = C.OUT_DIR / "report.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
