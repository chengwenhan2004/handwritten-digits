"""复现手写数字识别实验，输出核验用的训练索引、测试预测和调参记录。"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


SEED = 42
DIGITS = np.arange(10)


def audit_data(pixels, labels):
    """检查像素、标签及重复图像；数据契约不满足时终止，避免静默污染评估。"""
    if pixels.ndim != 2 or pixels.shape != (labels.size, 64):
        raise ValueError("图像与标签需一一对应，每张图像需要 64 个像素。")
    if not np.isfinite(pixels).all() or not np.isfinite(labels).all():
        raise ValueError("检测到缺失值或无穷值，请核验数据来源。")
    if not np.array_equal(np.unique(labels), DIGITS):
        raise ValueError("标签必须覆盖 0 到 9 的十个类别。")
    if pixels.min() < 0 or pixels.max() > 16 or np.any(pixels != np.floor(pixels)):
        raise ValueError("像素应为 0 到 16 的整数。")
    duplicate_count = len(labels) - len(np.unique(pixels, axis=0))
    if duplicate_count:
        raise ValueError(f"检测到 {duplicate_count} 条重复图像，请先核对或分组再切分。")
    return {
        "n_samples": len(labels), "n_features": pixels.shape[1],
        "class_counts": np.bincount(labels, minlength=10).tolist(),
        "invalid_values": 0, "duplicate_images": duplicate_count,
        "pixel_min": int(pixels.min()), "pixel_max": int(pixels.max()),
        "zero_pixel_fraction": float(np.mean(pixels == 0)),
        "constant_pixel_indices": np.flatnonzero(np.ptp(pixels, axis=0) == 0).tolist(),
    }


class DigitExperiment:
    """固定训练/测试划分，并对每个候选模型使用同一组训练折。"""

    def __init__(self, pixels, labels):
        self.audit = audit_data(pixels, labels)
        # 16 是原始编码的已知上限，转换不需要从训练集或测试集估计参数。
        self.X = pixels / 16.0
        self.y = labels
        self.train, self.test = train_test_split(
            np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=SEED
        )
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def tune(self, name, estimator, grid):
        search = GridSearchCV(estimator, grid, scoring="f1_macro", cv=self.cv,
                              refit=True, error_score="raise", n_jobs=1)
        search.fit(self.X[self.train], self.y[self.train])
        scores = search.cv_results_
        records = []
        for i, params in enumerate(scores["params"]):
            records.append({
                "model": name, "params": json.dumps(params, sort_keys=True),
                "mean_macro_f1": float(scores["mean_test_score"][i]),
                "std_macro_f1": float(scores["std_test_score"][i]),
                "rank": int(scores["rank_test_score"][i]),
            })
        return search.best_estimator_, search.best_params_, search.best_score_, records

    def evaluate(self, fitted):
        actual = self.y[self.test]
        pred = fitted.predict(self.X[self.test])
        return pred, {
            "correct": int(np.sum(pred == actual)),
            "accuracy": float(accuracy_score(actual, pred)),
            "macro_f1": float(f1_score(actual, pred, average="macro", zero_division=0)),
            "recall_by_class": recall_score(actual, pred, labels=DIGITS, average=None).tolist(),
            "confusion": confusion_matrix(actual, pred, labels=DIGITS).tolist(),
        }


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def neighbors_for_errors(exp, knn, predictions):
    """列出 KNN 错分图像的训练邻居，并把模型内部行号映射回原始下标。"""
    for j, sample in enumerate(exp.test):
        if predictions["knn"][j] == exp.y[sample]:
            continue
        distances, positions = knn.kneighbors(exp.X[[sample]])
        for rank, (distance, pos) in enumerate(zip(distances[0], positions[0]), start=1):
            neighbor = exp.train[pos]
            yield {
                "index": sample, "true": exp.y[sample], "knn": predictions["knn"][j],
                "svm": predictions["svm"][j], "rank": rank, "neighbor_index": neighbor,
                "neighbor_label": exp.y[neighbor], "distance": distance,
            }


def run_experiment(output):
    pixels, labels = load_digits(return_X_y=True)
    exp = DigitExperiment(pixels, labels)
    baseline = DummyClassifier(strategy="most_frequent").fit(exp.X[exp.train], labels[exp.train])
    fitted = {"baseline": (baseline, {"strategy": "most_frequent"}, None)}
    candidate_models = [
        ("knn", KNeighborsClassifier(metric="euclidean", algorithm="brute"),
         {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"]}),
        ("svm", SVC(kernel="rbf"),
         {"C": [1, 10, 100], "gamma": [0.01, 0.1, 1.0]}),
    ]
    tuning = []
    for name, estimator, grid in candidate_models:
        best, params, cv_f1, rows = exp.tune(name, estimator, grid)
        fitted[name] = best, params, cv_f1
        tuning.extend(rows)

    chosen = max(("knn", "svm"), key=lambda name: fitted[name][2])
    predictions, metrics = {}, {}
    for name, (best, params, cv_f1) in fitted.items():
        predictions[name], metrics[name] = exp.evaluate(best)
        metrics[name].update(best_params=params, best_cv_f1=cv_f1)

    actual = labels[exp.test]
    knn_right = predictions["knn"] == actual
    svm_right = predictions["svm"] == actual
    summary = {
        "seed": SEED, "audit": exp.audit,
        "split": {
            "n_train": len(exp.train), "n_test": len(exp.test),
            "train_counts": np.bincount(labels[exp.train], minlength=10).tolist(),
            "test_counts": np.bincount(actual, minlength=10).tolist(),
        },
        "cv_selected_model": chosen, "models": metrics,
        "paired_counts": {
            "both_right": int(np.sum(knn_right & svm_right)),
            "knn_only_right": int(np.sum(knn_right & ~svm_right)),
            "svm_only_right": int(np.sum(~knn_right & svm_right)),
            "both_wrong": int(np.sum(~knn_right & ~svm_right)),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "tuning.csv", tuning,
              ["model", "params", "mean_macro_f1", "std_macro_f1", "rank"])
    write_csv(output / "training.csv",
              ({"index": i, "true": labels[i]} for i in exp.train), ["index", "true"])
    test_rows = [
        {"index": i, "true": labels[i], **{name: pred[j] for name, pred in predictions.items()}}
        for j, i in enumerate(exp.test)
    ]
    write_csv(output / "predictions.csv", test_rows, ["index", "true", "baseline", "knn", "svm"])
    write_csv(output / "neighbors.csv", neighbors_for_errors(exp, fitted["knn"][0], predictions),
              ["index", "true", "knn", "svm", "rank", "neighbor_index", "neighbor_label", "distance"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, result in metrics.items():
        print(f"{name:8s} accuracy={result['accuracy']:.4f} macro-F1={result['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    try:
        run_experiment(args.output)
    except (ValueError, OSError) as exc:
        sys.exit(f"实验中止：{exc}")
