#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# mDPM E-step 消融实验 — 一键运行
# ═══════════════════════════════════════════════════════════════
#
# 用法:
#   chmod +x run_mDPM_ablation.sh
#   nohup ./run_mDPM_ablation.sh > mDPM_ablation.log 2>&1 &
#
# 预计时间: 每个 config×seed 约 40-60 分钟 (GPU), 总共 ~15-25 小时
# 按 Ctrl+C 中断后, 已完成的实验结果不会丢失
# ═══════════════════════════════════════════════════════════════

set -e  # 任何命令出错立即停止

SCRIPT="python mDPM_aligned.py"
EPOCHS=60
SEEDS=(42 123 2024)

echo "╔══════════════════════════════════════════════════════╗"
echo "║  mDPM E-step Ablation — $(date)           ║"
echo "║  ${#SEEDS[@]} seeds × N configs, ${EPOCHS} epochs each        ║"
echo "╚══════════════════════════════════════════════════════╝"

# ──────────────────────────────────────────────
# 辅助函数: 跳过已完成的实验
# ──────────────────────────────────────────────
run_if_needed() {
    local name="$1"
    shift
    local outdir="./mDPM_ablation/${name}"

    # 检查是否已有 best_model.pt (说明已跑完)
    if [ -f "${outdir}/best_model.pt" ]; then
        echo ""
        echo "⏭  [${name}] — already done, skipping"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "▶  [${name}] — starting..."
    echo "   CMD: $SCRIPT $@ --output_dir ${outdir}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    $SCRIPT "$@" --output_dir "${outdir}" --epochs ${EPOCHS}

    echo "✅ [${name}] — done"
}

# ══════════════════════════════════════════════
# Group 1: 最有希望的组合 (优先跑)
# ══════════════════════════════════════════════
echo ""
echo "═══ Group 1: Key combinations ═══"

for seed in "${SEEDS[@]}"; do

    # ★ 最有希望: fixed E-step + single M-step + Z-score ON
    run_if_needed "fixed_single_zscore_s${seed}" \
        --mode unsup --estep fixed --mstep single --zscore on --seed ${seed}

    # 对照: fixed + single + Z-score OFF (隔离 Z-score 效果)
    run_if_needed "fixed_single_nozscore_s${seed}" \
        --mode unsup --estep fixed --mstep single --zscore off --seed ${seed}

    # 对照: multi timestep + single + Z-score ON
    run_if_needed "multi_single_zscore_s${seed}" \
        --mode unsup --estep multi --mstep single --zscore on --seed ${seed}

done

# ══════════════════════════════════════════════
# Group 2: 原始 baseline 复现 (3 seeds)
# ══════════════════════════════════════════════
echo ""
echo "═══ Group 2: Baseline reproduction ═══"

for seed in "${SEEDS[@]}"; do

    # 原始 0.69 配置: random + single + zscore off
    run_if_needed "baseline_random_s${seed}" \
        --mode unsup --estep random --mstep single --zscore off --seed ${seed}

done

# ══════════════════════════════════════════════
# Group 3: 补充消融 (如果时间允许)
# ══════════════════════════════════════════════
echo ""
echo "═══ Group 3: Additional ablations ═══"

for seed in "${SEEDS[@]}"; do

    # Weighted soft-EM + Z-score (对比 single)
    run_if_needed "fixed_weighted_zscore_s${seed}" \
        --mode unsup --estep fixed --mstep weighted --zscore on --seed ${seed}

    # Scale 消融: 低 target scale (5→50 instead of 5→134)
    run_if_needed "fixed_single_zscore_lowscale_s${seed}" \
        --mode unsup --estep fixed --mstep single --zscore on \
        --target_scale 50.0 --seed ${seed}

    # K=15 过估计 (对比 mVAE 的相反结果)
    # 注意: 需要 common_dpm.py 中 Config 支持 --num_classes 参数
    # 如果不支持, 注释掉这两行
    # run_if_needed "K15_fixed_single_zscore_s${seed}" \
    #     --mode unsup --estep fixed --mstep single --zscore on \
    #     --num_classes 15 --seed ${seed}

done

# ══════════════════════════════════════════════
# 汇总结果
# ══════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  ALL DONE — $(date)                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Results saved in ./mDPM_ablation/"
echo ""

# 自动汇总所有实验的 best acc
echo "┌─────────────────────────────────────────┬──────────┐"
echo "│ Configuration                           │ Best Acc │"
echo "├─────────────────────────────────────────┼──────────┤"

for dir in ./mDPM_ablation/*/; do
    name=$(basename "$dir")
    config_file="${dir}config.json"
    if [ -f "$config_file" ]; then
        acc=$(python3 -c "import json; d=json.load(open('${config_file}')); print(f'{d.get(\"best_acc\", 0):.4f}')" 2>/dev/null || echo "N/A")
        printf "│ %-39s │ %8s │\n" "$name" "$acc"
    fi
done

echo "└─────────────────────────────────────────┴──────────┘"
