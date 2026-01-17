"""
分析训练日志文件，检查训练稳定性
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_training_log(log_file):
    """加载训练日志"""
    iterations = []
    loss_d = []
    loss_g_adv = []
    loss_reg = []
    
    if not os.path.exists(log_file):
        print(f" 日志文件不存在: {log_file}")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['Iteration']))
            loss_d.append(float(row['Loss_D']))
            loss_g_adv.append(float(row['Loss_G_adv']))
            loss_reg.append(float(row['Loss_Reg']))
    
    return {
        'iterations': np.array(iterations),
        'loss_d': np.array(loss_d),
        'loss_g_adv': np.array(loss_g_adv),
        'loss_reg': np.array(loss_reg)
    }

def analyze_stability(data):
    """分析训练稳定性"""
    print("\n" + "="*70)
    print("📊 训练稳定性分析")
    print("="*70)
    
    # 基本统计
    print(f"\n✓ 总迭代次数: {len(data['iterations'])}")
    
    # Loss_D 分析
    print(f"\n📈 判别器损失 (Loss_D):")
    print(f"   平均值: {data['loss_d'].mean():.6f}")
    print(f"   标准差: {data['loss_d'].std():.6f}")
    print(f"   最小值: {data['loss_d'].min():.6f}")
    print(f"   最大值: {data['loss_d'].max():.6f}")
    
    # Loss_G_adv 分析
    print(f"\n📈 生成器对抗损失 (Loss_G_adv):")
    print(f"   平均值: {data['loss_g_adv'].mean():.6f}")
    print(f"   标准差: {data['loss_g_adv'].std():.6f}")
    print(f"   最小值: {data['loss_g_adv'].min():.6f}")
    print(f"   最大值: {data['loss_g_adv'].max():.6f}")
    
    # Loss_Reg 分析
    print(f"\n📈 核正则化损失 (Loss_Reg):")
    print(f"   平均值: {data['loss_reg'].mean():.6f}")
    print(f"   标准差: {data['loss_reg'].std():.6f}")
    print(f"   最小值: {data['loss_reg'].min():.6f}")
    print(f"   最大值: {data['loss_reg'].max():.6f}")
    
    # 趋势分析 (使用后半部分与前半部分的比较)
    mid_point = len(data['iterations']) // 2
    first_half_d = data['loss_d'][:mid_point]
    second_half_d = data['loss_d'][mid_point:]
    first_half_g = data['loss_g_adv'][:mid_point]
    second_half_g = data['loss_g_adv'][mid_point:]
    first_half_r = data['loss_reg'][:mid_point]
    second_half_r = data['loss_reg'][mid_point:]
    
    print(f"\n📊 前后期对比:")
    print(f"   Loss_D: 前期平均={first_half_d.mean():.6f}, 后期平均={second_half_d.mean():.6f}")
    d_trend = (second_half_d.mean() - first_half_d.mean()) / first_half_d.mean() * 100
    print(f"           变化趋势: {d_trend:+.2f}%")
    
    print(f"   Loss_G_adv: 前期平均={first_half_g.mean():.6f}, 后期平均={second_half_g.mean():.6f}")
    g_trend = (second_half_g.mean() - first_half_g.mean()) / first_half_g.mean() * 100
    print(f"              变化趋势: {g_trend:+.2f}%")
    
    print(f"   Loss_Reg: 前期平均={first_half_r.mean():.6f}, 后期平均={second_half_r.mean():.6f}")
    r_trend = (second_half_r.mean() - first_half_r.mean()) / first_half_r.mean() * 100
    print(f"            变化趋势: {r_trend:+.2f}%")
    
    # 稳定性评估
    print(f"\n⚠️  稳定性评估:")
    d_cv = data['loss_d'].std() / data['loss_d'].mean()  # 变异系数
    g_cv = data['loss_g_adv'].std() / data['loss_g_adv'].mean()
    r_cv = data['loss_reg'].std() / data['loss_reg'].mean()
    
    print(f"   Loss_D 变异系数: {d_cv:.4f} {'✓ 稳定' if d_cv < 0.3 else '⚠️ 波动较大' if d_cv < 0.5 else '❌ 非常不稳定'}")
    print(f"   Loss_G_adv 变异系数: {g_cv:.4f} {'✓ 稳定' if g_cv < 0.3 else '⚠️ 波动较大' if g_cv < 0.5 else '❌ 非常不稳定'}")
    print(f"   Loss_Reg 变异系数: {r_cv:.4f} {'✓ 稳定' if r_cv < 0.3 else '⚠️ 波动较大' if r_cv < 0.5 else '❌ 非常不稳定'}")
    
    # 梯度爆炸检测
    print(f"\n⚡ 异常值检测:")
    d_outliers = np.sum(data['loss_d'] > data['loss_d'].mean() + 3*data['loss_d'].std())
    g_outliers = np.sum(data['loss_g_adv'] > data['loss_g_adv'].mean() + 3*data['loss_g_adv'].std())
    r_outliers = np.sum(data['loss_reg'] > data['loss_reg'].mean() + 3*data['loss_reg'].std())
    
    print(f"   Loss_D 异常值数: {d_outliers} {'✓ 无' if d_outliers == 0 else f'⚠️ {d_outliers}次'}")
    print(f"   Loss_G_adv 异常值数: {g_outliers} {'✓ 无' if g_outliers == 0 else f'⚠️ {g_outliers}次'}")
    print(f"   Loss_Reg 异常值数: {r_outliers} {'✓ 无' if r_outliers == 0 else f'⚠️ {r_outliers}次'}")
    
    # 综合判断
    print(f"\n🎯 综合判断:")
    stability_score = 0
    if d_cv < 0.3 and g_cv < 0.3:
        stability_score += 2
        print("   ✓ 两个主损失函数都相对稳定")
    elif d_cv < 0.5 and g_cv < 0.5:
        stability_score += 1
        print("   ⚠️ 两个主损失函数波动中等")
    else:
        print("   ❌ 两个主损失函数波动较大")
    
    if abs(d_trend) < 20 and abs(g_trend) < 20:
        stability_score += 1
        print("   ✓ 损失值趋势稳定，无明显恶化")
    elif abs(d_trend) < 40 and abs(g_trend) < 40:
        print("   ⚠️ 损失值有一定波动")
    else:
        print("   ❌ 损失值趋势明显恶化")
    
    if d_outliers == 0 and g_outliers == 0:
        stability_score += 1
        print("   ✓ 无明显梯度爆炸现象")
    else:
        print(f"   ⚠️ 检测到 {d_outliers + g_outliers} 个异常尖峰")
    
    print(f"\n   稳定性评分: {stability_score}/4")
    if stability_score >= 3:
        print("   💚 训练较稳定，可继续")
    elif stability_score >= 2:
        print("   🟡 训练基本稳定，但需要监控")
    else:
        print("   🔴 训练不稳定，建议调整超参数")
    
    print("="*70 + "\n")

def plot_training_curves(data, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Loss_D
    axes[0].plot(data['iterations'], data['loss_d'], linewidth=1.5, label='Loss_D')
    axes[0].set_ylabel('Loss_D', fontsize=12)
    axes[0].set_title('Discriminator Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Loss_G_adv
    axes[1].plot(data['iterations'], data['loss_g_adv'], linewidth=1.5, color='orange', label='Loss_G_adv')
    axes[1].set_ylabel('Loss_G_adv', fontsize=12)
    axes[1].set_title('Generator Adversarial Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Loss_Reg
    axes[2].plot(data['iterations'], data['loss_reg'], linewidth=1.5, color='green', label='Loss_Reg')
    axes[2].set_ylabel('Loss_Reg', fontsize=12)
    axes[2].set_xlabel('Iteration', fontsize=12)
    axes[2].set_title('Kernel Regularization Loss', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_path}")
    plt.close()

if __name__ == "__main__":
    log_file = r"output\kernelgan_out_denoised_single_kernel\training_log.txt"
    
    data = load_training_log(log_file)
    if data is not None:
        analyze_stability(data)
        
        output_dir = os.path.dirname(log_file)
        plot_training_curves(data, output_dir)
