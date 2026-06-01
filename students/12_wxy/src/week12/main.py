import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 修复导入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.metrics import rmse, mae
from utils.models import CustomOLS

plt.rcParams['font.size'] = 12
BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def polynomial_features(X, degree):
    X = X.reshape(-1, 1)
    X_scaled = (X - np.mean(X)) / np.std(X)
    features = [np.ones_like(X_scaled)]
    for d in range(1, degree + 1):
        features.append(X_scaled ** d)
    return np.hstack(features[1:])

def generate_data(n_samples=150):
    rng = np.random.RandomState(42)
    x = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = (np.sin(x) + 0.1 * x**2).ravel()
    noise = rng.normal(0, 0.3, size=n_samples)
    y = y_true + noise
    
    return x, y, y_true

def train_test_split(x, y, test_ratio=0.3):
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(x))
    test_size = int(len(x) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

# ====================== Task A：1/4/15阶候选模型对比 ======================
def run_candidate_models():
    print("[1/5] 训练 1/4/15 阶多项式模型")
    x, y, y_true = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    degrees = [1, 4, 15]
    colors = ["orange", "green", "red"]
    labels = ["Degree 1", "Degree 4", "Degree 15"]

    plt.figure(figsize=(12, 5))
    plt.scatter(x_train, y_train, c="tab:blue", alpha=0.5, label="Train")
    plt.scatter(x_test, y_test, c="tab:gray", alpha=0.5, label="Test")
    plt.plot(x, y_true, "k--", lw=2.5, label="True Function: sin(x) + 0.1x²")

    results = []
    # 排序x保证曲线连续平滑
    sort_idx = np.argsort(x.ravel())
    x_sorted = x[sort_idx]
    for d, c, lab in zip(degrees, colors, labels):
        Xp_train = polynomial_features(x_train, d)
        Xp_full = polynomial_features(x_sorted, d)
        model = CustomOLS(fit_intercept=True, alpha=0.0)
        model.fit(Xp_train, y_train)
        y_curve = model.predict(Xp_full)

        y_pred_tr = model.predict(Xp_train)
        y_pred_te = model.predict(polynomial_features(x_test, d))
        tr_rmse = rmse(y_train, y_pred_tr)
        te_rmse = rmse(y_test, y_pred_te)

        results.append((d, tr_rmse, te_rmse))
        plt.plot(x_sorted, y_curve, c=c, lw=2.5,
                 label=f"{lab} | Tr={tr_rmse:.2f} Te={te_rmse:.2f}")

    plt.title("Candidate Models: Underfit / Optimal / Overfit\nTrue Function: sin(x) + 0.1x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "candidate_models.png"), dpi=150)
    plt.close()
    return results

# ====================== Task B：1~18阶误差曲线扫描 ======================
def run_error_curve():
    print("[2/5] 扫描模型复杂度 1~18")
    x, y, y_true = generate_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    degrees = list(range(1, 19))
    tr_list = []
    te_list = []

    for d in degrees:
        Xp_tr = polynomial_features(x_train, d)
        Xp_te = polynomial_features(x_test, d)
        model = CustomOLS(alpha=0.0)
        model.fit(Xp_tr, y_train)
        tr_list.append(rmse(y_train, model.predict(Xp_tr)))
        te_list.append(rmse(y_test, model.predict(Xp_te)))

    plt.figure(figsize=(10, 5))
    plt.plot(degrees, tr_list, "o-", label="Train RMSE")
    plt.plot(degrees, te_list, "o-", label="Test RMSE")
    plt.xlabel("Degree (Complexity)")
    plt.ylabel("RMSE")
    plt.title("Train vs Test Error Across Complexity\nTrue Function: sin(x) + 0.1x²")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "error_curves.png"), dpi=150)
    plt.close()
    return [(d, t, e, e-t) for d, t, e in zip(degrees, tr_list, te_list)]

# ====================== Task C：方差可视化演示 ======================
def run_variance_demo(n_repeat=10):
    print("[3/5] 方差可视化（多次抽样）")
    x, y, y_true = generate_data(n_samples=80)
    rng = np.random.RandomState(42)
    degrees = [2, 15]
    plt.figure(figsize=(12, 5))
    stds = {}
    sort_idx = np.argsort(x.ravel())
    x_sorted = x[sort_idx]

    for i, (d, title) in enumerate(zip(degrees, ["Low Variance (Deg2)", "High Variance (Deg15)"])):
        plt.subplot(1, 2, i+1)
        plt.scatter(x, y, s=20, alpha=0.5, c="gray")
        plt.plot(x_sorted, y_true[sort_idx], "k--", lw=2, label="True: sin(x)+0.1x²")
        preds = []

        for _ in range(n_repeat):
            idx = rng.choice(len(x), 60, replace=False)
            xs, ys = x[idx], y[idx]
            Xp = polynomial_features(xs, d)
            m = CustomOLS(alpha=0)
            m.fit(Xp, ys)
            yp = m.predict(polynomial_features(x_sorted, d))
            preds.append(yp)
            plt.plot(x_sorted, yp, alpha=0.6, lw=1)

        preds = np.array(preds)
        m_std = np.mean(np.std(preds, axis=0))
        stds[d] = m_std
        plt.title(f"{title}\nMean Std = {m_std:.3f}")
        plt.legend()

    plt.suptitle("Variance Demo: Low vs High Complexity\nTrue Function: sin(x) + 0.1x²")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "variance_demo.png"), dpi=150)
    plt.close()
    return stds

# ====================== Task D：RMSE与MAE异常值对比 ======================
def run_loss_demo():
    print("[4/5] 异常值对 RMSE / MAE 影响")
    rng = np.random.RandomState(42)
    y_true = np.linspace(0, 10, 50)
    y_clean = y_true + rng.normal(0, 0.5, 50)
    y_bad = y_clean.copy()
    y_bad[10] += 10  # 人工添加极端异常点

    rmse_clean = rmse(y_true, y_clean)
    mae_clean = mae(y_true, y_clean)
    rmse_out = rmse(y_true, y_bad)
    mae_out = mae(y_true, y_bad)

    res = {
        "clean": (rmse_clean, mae_clean),
        "outlier": (rmse_out, mae_out)
    }

    plt.figure(figsize=(10,5))
    plt.scatter(range(len(y_true)), y_true, label="True", s=30)
    plt.scatter(range(len(y_clean)), y_clean, alpha=0.6, label="Clean Pred")
    plt.scatter(range(len(y_bad)), y_bad, alpha=0.6, label="With Outlier")
    plt.title(f"RMSE sensitivity: {rmse_clean:.2f} → {rmse_out:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "loss_outlier_comparison.png"), dpi=150)
    plt.close()
    return res

# ====================== 自动生成summary.md实验报告 ======================
def write_report(candidate, err_table, var_res, loss_res):
    print("[5/5] 生成 summary.md")
    path = os.path.join(BASE_DIR, "results", "summary.md")

    d1_tr, d1_te = candidate[0][1], candidate[0][2]
    d4_tr, d4_te = candidate[1][1], candidate[1][2]
    d15_tr, d15_te = candidate[2][1], candidate[2][2]
    best_degree = min(err_table, key=lambda x: x[2])[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Week12 Bias-Variance 实验报告\n\n")
        f.write("## A1 数据生成说明\n")
        f.write("1. 样本总量150，满足不少于100的要求；\n")
        f.write("2. 自定义非线性真实函数：**sin(x) + 0.1x²**（正弦函数 + 二次趋势）；\n")
        f.write("3. 叠加均值0、标准差0.3的高斯随机噪声生成观测y；\n")
        f.write("4. 随机划分70%训练集、30%测试集。\n\n")

        f.write("## 一、候选模型对比（Task A）\n")
        f.write("三个模型：Degree 1、Degree 4、Degree 15\n\n")
        f.write(f"- Degree 1 模型：训练RMSE = {d1_tr:.2f}，测试RMSE = {d1_te:.2f}\n")
        f.write(f"- Degree 4 模型：训练RMSE = {d4_tr:.2f}，测试RMSE = {d4_te:.2f}\n")
        f.write(f"- Degree 15 模型：训练RMSE = {d15_tr:.2f}，测试RMSE = {d15_te:.2f}\n\n")

        f.write("### 回答问题\n")
        f.write("- **Degree 1 最像欠拟合**，模型为一次直线，过于简单，无法拟合 sin(x) + 0.1x² 的非线性趋势，整体偏差高。\n")
        f.write("- **Degree 15 最像过拟合**，模型复杂度极高，过度学习训练集噪声，训练误差偏低但测试误差大幅上升，方差极高。\n")
        f.write("- **选择 Degree 4 上线**，该模型拟合曲线贴近真实函数，偏差与方差权衡最优，泛化能力最强。\n\n")

        f.write("## 二、模型复杂度与误差曲线（Task B）\n")
        f.write("| 复杂度Degree | 训练RMSE | 测试RMSE | 泛化Gap |\n")
        f.write("|-------------|----------|----------|---------|\n")
        for row in err_table:
            f.write(f"| {row[0]:<11} | {row[1]:<8.3f} | {row[2]:<8.3f} | {row[3]:<7.3f} |\n")

        f.write(f"\n**测试误差最低的复杂度：{best_degree}**\n")
        f.write("**泛化Gap最大：高次多项式（10~18阶）**\n")
        f.write("训练误差最低不代表模型最好，高复杂度模型极易过拟合，在测试集上泛化能力反而变差。\n\n")

        f.write("## 三、方差可视化（Task C）\n")
        f.write(f"- 低方差模型 Degree 2：平均预测标准差 = {var_res[2]:.3f}\n")
        f.write(f"- 高方差模型 Degree 15：平均预测标准差 = {var_res[15]:.3f}\n\n")
        f.write("> high variance model 的危险，不是它不会拟合训练集，\n")
        f.write("> 而是它对 **训练样本的微小变化** 过于敏感。\n\n")
        f.write("低方差模型多次抽样拟合曲线几乎重合；高方差模型每次抽样曲线差异巨大，稳定性差。\n\n")

        f.write("## 四、异常值对 RMSE / MAE 的影响（Task D）\n")
        f.write("| 场景 | RMSE | MAE |\n")
        f.write("|------|------|-----|\n")
        f.write(f"| 干净预测 | {loss_res['clean'][0]:.2f} | {loss_res['clean'][1]:.2f} |\n")
        f.write(f"| 含一个大异常值 | {loss_res['outlier'][0]:.2f} | {loss_res['outlier'][1]:.2f} |\n\n")

        f.write("### 业务解释\n")
        f.write("1. RMSE 引入平方运算，会放大大误差权重，因此更容易被极端异常值剧烈拉高。\n")
        f.write("2. 如果线上系统单次大错误业务代价极高，更适合关注 RMSE，它能快速暴露严重预测偏差。\n")
        f.write("3. 若数据天然存在较多异常值，MAE 对极端值更稳健，更适合作为模型评价指标。\n\n")

        f.write("## 五、必答总结\n")
        f.write("### 1. 三条核心结论\n")
        f.write("① 模型复杂度升高，训练误差整体持续下降，测试误差先下降后上升，上升阶段代表出现过拟合。\n")
        f.write("② 高方差模型对训练样本的轻微改动就会产生巨大预测波动，拟合曲线抖动剧烈，泛化稳定性极差。\n")
        f.write("③ RMSE对极端误差敏感，MAE对异常值更耐受，指标选择需要匹配业务的误差损失偏好。\n\n")

        f.write("### 2. 最能代表过拟合的图\n")
        f.write("**variance_demo.png 方差对比图** 最能直观代表过拟合现象。\n")
        f.write("15阶高复杂度模型在10次不同抽样训练后，曲线形态差异极大，受训练集内随机噪声严重干扰，无法学到通用的底层函数规律，泛化能力弱。\n\n")

        f.write("### 3. 指标选择判断\n")
        f.write("- 业务重视严重大误差、数据集干净异常点少时，优先使用 RMSE。\n")
        f.write("- 数据噪声大、天然存在大量异常值时，优先选用 MAE。\n\n")

        f.write("### 4. 为什么要引入正则化？\n")
        f.write("模型复杂度过高会带来高方差与严重过拟合，更换训练样本后预测结果剧烈波动、不稳定。\n")
        f.write("正则化通过在损失函数中惩罚模型系数大小，约束模型有效复杂度、降低方差，让拟合曲线更加平滑稳定，提升模型泛化能力。\n")

# ====================== 程序主入口 ======================
def main():
    print("="*50)
    print(" Week12 偏差-方差可视化实验 ")
    print(" 真实函数: sin(x) + 0.1x² (正弦函数 + 二次趋势)")
    print("="*50)

    candidate = run_candidate_models()
    err_table = run_error_curve()
    var_res = run_variance_demo()
    loss_res = run_loss_demo()
    write_report(candidate, err_table, var_res, loss_res)

    print("\n✅ 全部实验执行完成！")
    print("📊 可视化图片保存路径：results/figures/")
    print("📝 实验报告自动生成路径：results/summary.md")

if __name__ == "__main__":
    main()
