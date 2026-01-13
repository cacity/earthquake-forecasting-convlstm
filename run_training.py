#!/usr/bin/env python
"""
完整的训练流程脚本
用途：从零开始，下载数据、处理数据、训练模型
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(cmd, description=""):
    """运行命令并实时输出"""
    print(f"\n{'='*80}")
    if description:
        print(f"▶ {description}")
    print(f"{'='*80}")
    print(f"命令: {' '.join(cmd)}")
    print()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"\n❌ 命令执行失败，返回码: {process.returncode}")
            return False
        else:
            print(f"\n✅ 完成")
            return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return False

def check_file(path, description=""):
    """检查文件是否存在"""
    p = Path(path)
    exists = p.exists()
    size_str = ""
    if exists and p.is_file():
        size = p.stat().st_size
        if size > 1024*1024*1024:
            size_str = f" ({size/1024/1024/1024:.2f} GB)"
        elif size > 1024*1024:
            size_str = f" ({size/1024/1024:.2f} MB)"
        elif size > 1024:
            size_str = f" ({size/1024:.2f} KB)"
        else:
            size_str = f" ({size} bytes)"

    status = "✅" if exists else "❌"
    desc_str = f" - {description}" if description else ""
    print(f"  {status} {path}{size_str}{desc_str}")
    return exists

def create_symlinks():
    """创建符号链接，让评估脚本能找到文件"""
    print(f"\n{'='*80}")
    print("创建符号链接（适配评估脚本路径）")
    print(f"{'='*80}\n")

    # 确保目录存在
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # 定义链接映射
    links = [
        ("data/processed/splits_L12_H1/train_Y.npy", "data/processed/train_labels.npy"),
        ("data/processed/splits_L12_H1/val_Y.npy", "data/processed/val_labels.npy"),
        ("data/processed/splits_L12_H1/test_Y.npy", "data/processed/test_labels.npy"),
        ("data/processed/splits_L12_H1/test_X.npy", "data/processed/test_features.npy"),
    ]

    for source, link in links:
        source_path = Path(source)
        link_path = Path(link)

        if source_path.exists():
            # 删除已存在的链接
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()

            # 创建相对路径的符号链接
            relative_source = Path("splits_L12_H1") / source_path.name
            link_path.symlink_to(relative_source)
            print(f"  ✅ {link} -> {relative_source}")
        else:
            print(f"  ⚠️  源文件不存在: {source}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   IEEE Access 论文 - 完整训练流程                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

这个脚本将执行以下步骤:
  1. 下载 USGS 地震数据 (2000-2025)
  2. 构建张量 (grid-based features)
  3. 创建样本 (lookback=12, horizon=1)
  4. 划分训练/验证/测试集
  5. 训练 ConvLSTM 模型 (50 epochs, early stopping)

预计总时间: 30-60 分钟（取决于网络速度和硬件）

注意:
  - 需要稳定的网络连接（下载 USGS 数据）
  - 建议有 GPU（CPU 也可以，但会慢一些）
    """)

    input("按 Enter 继续，或 Ctrl+C 取消...")

    # 进入项目目录
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    print(f"\n工作目录: {project_dir.absolute()}")
    print(f"Python: {sys.executable}")
    print(f"版本: {sys.version}")

    # 运行 pipeline
    success = run_command(
        [sys.executable, "run_pipeline.py", "--config", "configs/pipeline_for_paper.json"],
        description="运行完整 Pipeline"
    )

    if not success:
        print("\n❌ Pipeline 执行失败！")
        print("请检查错误信息，修复问题后重新运行。")
        return 1

    # 检查生成的文件
    print(f"\n{'='*80}")
    print("检查生成的文件")
    print(f"{'='*80}\n")

    print("1. 原始数据:")
    check_file("data/interim/events_southwest.parquet", "USGS 地震事件")

    print("\n2. 处理后的张量:")
    check_file("data/processed/X.npy", "特征张量")
    check_file("data/processed/Y.npy", "标签张量")
    check_file("data/processed/grid_meta.json", "网格元数据")

    print("\n3. 训练/验证/测试集:")
    check_file("data/processed/splits_L12_H1/train_X.npy", "训练特征")
    check_file("data/processed/splits_L12_H1/train_Y.npy", "训练标签")
    check_file("data/processed/splits_L12_H1/val_X.npy", "验证特征")
    check_file("data/processed/splits_L12_H1/val_Y.npy", "验证标签")
    check_file("data/processed/splits_L12_H1/test_X.npy", "测试特征")
    check_file("data/processed/splits_L12_H1/test_Y.npy", "测试标签")

    print("\n4. 训练好的模型:")
    model_exists = check_file("outputs/convlstm/best_model.pth", "最佳模型权重")
    preds_exists = check_file("outputs/convlstm/test_preds.npy", "测试集预测")
    test_logits = check_file("outputs/convlstm/test_logits.npy", "测试集 logits")
    val_logits = check_file("outputs/convlstm/val_logits.npy", "验证集 logits")

    # 创建符号链接
    create_symlinks()

    # 最终检查
    print(f"\n{'='*80}")
    print("最终检查 - 评估脚本所需文件")
    print(f"{'='*80}\n")

    all_ready = True
    required_files = [
        ("outputs/convlstm/test_preds.npy", "模型预测"),
        ("outputs/convlstm/test_logits.npy", "测试 logits"),
        ("outputs/convlstm/val_logits.npy", "验证 logits"),
        ("data/processed/train_labels.npy", "训练标签"),
        ("data/processed/val_labels.npy", "验证标签"),
        ("data/processed/test_labels.npy", "测试标签"),
        ("data/processed/test_features.npy", "测试特征"),
    ]

    for file_path, desc in required_files:
        exists = check_file(file_path, desc)
        if not exists:
            all_ready = False

    # 打印下一步指示
    print(f"\n{'='*80}")
    if all_ready:
        print("🎉 所有准备工作完成！")
        print(f"{'='*80}\n")
        print("你现在可以:")
        print("\n1. 生成基线模型预测（5分钟）:")
        print("   python generate_baselines.py")
        print("\n2. 运行完整评估（30-60分钟）:")
        print("   python scripts/run_comprehensive_evaluation.py \\")
        print("     --model-preds outputs/convlstm/test_preds.npy \\")
        print("     --test-labels data/processed/test_labels.npy \\")
        print("     --model-logits outputs/convlstm/test_logits.npy \\")
        print("     --val-labels data/processed/val_labels.npy \\")
        print("     --val-logits outputs/convlstm/val_logits.npy \\")
        print("     --train-labels data/processed/train_labels.npy \\")
        print("     --test-features data/processed/test_features.npy \\")
        print("     --output-dir paper_evaluation_results \\")
        print("     --n-bootstrap 1000")
        print("\n详细说明请查看: QUICKSTART.md")
    else:
        print("⚠️  有文件缺失，请检查上面的错误信息")
        print(f"{'='*80}\n")
        return 1

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
