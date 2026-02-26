"""
Mandar Bangalore Arun
Handwritten Digit Image Classification
"""

#Required packages

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# PREPARING AND LOADING THE DATA
# -----------------------------------------

X_train = pd.read_csv('X_train.csv').values
X_test = pd.read_csv('X_test.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

#Load multi-label targets: [is_even, is_greater_than_5, is_prime]
y_mll_train = pd.read_csv('y_MLL_train.csv').values
y_mll_test = pd.read_csv('y_MLL_test.csv').values

print(f"X_train shape: {X_train.shape} (pixel range: {X_train.min()} to {X_train.max()})")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape} classes: {np.unique(y_train)}")
print(f"y_test shape: {y_test.shape}")
print(f"y_MLL_train shape: {y_mll_train.shape}")
print(f"y_MLL_test shape: {y_mll_test.shape}")

#Distribution of classes
print("\n Class distribution (train):")
for d in range(10):
    count = (y_train == d).sum()
    print(f" Digit {d}: {count} samples")

#Normalising pixel features to [0,1]
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

#Feature standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_test_scaled = scaler.transform(X_test_norm)

property_names = ['is_even', 'is_greater_than_5', 'is_prime']

# --------------------------------------------------
# Part 1: Multi-class classification with softmax
# --------------------------------------------------

print("\n" + "=" * 70)
print("Part 1: MULTI-CLASS CLASSIFICATION USING SOFTMAX (10-WAY)")
print("=" * 70)

softmax_model = LogisticRegression(
    solver='lbfgs',
    C=1.0,
    max_iter=5000,
    random_state=42
)
softmax_model.fit(X_train_scaled,y_train)
print("Softmax model trained successfully")
print(f"Coefficient matrix shape: {softmax_model.coef_.shape}")

# (b.i) Plot distribution of learned coefficients for class 0 (red) and class 7 (blue)

coef_class_0 = softmax_model.coef_[0]
coef_class_7 = softmax_model.coef_[7]

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(coef_class_0, bins=30, alpha = 0.6, color='red', label='Class 0', edgecolor='darkred')
ax.hist(coef_class_7, bins=30, alpha = 0.6, color='blue', label='Class 7', edgecolor='darkblue')
ax.set_xlabel('Coefficient Value',fontsize = 12)
ax.set_ylabel('Frequency',fontsize = 12)
ax.set_title('Distribution of learned coefficients: Class 0 vs Class 7',fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha = 0.3)
plt.tight_layout()
plt.savefig('Part 1.1.png',dpi = 150, bbox_inches='tight')
plt.close()

#  (b.ii)Training and testing accuracy

y_train_pred = softmax_model.predict(X_train_scaled)
y_test_pred = softmax_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test,y_test_pred)

print(f"Training accuracy: {train_acc:.4f} ({train_acc*100:.2f})")
print(f"Testing accuracy: {test_acc:.4f} ({test_acc*100:.2f})")

# (b.iii) Testing accuracy for each class

per_class_acc = []
for digit in range(10):
    mask = (y_test == digit)
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], y_test_pred[mask])
    else:
        class_acc = 0.0
    per_class_acc.append(class_acc)
    print(f"Digit: {digit}: {class_acc:.4f} ({class_acc*100:.1f}%) [n={mask.sum()}]")

# Bar plot of per-class accuracy
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(10), per_class_acc, color='steelblue', edgecolor='navy', alpha=0.8)
ax.set_xlabel('Digit Class', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Per-Class Test Accuracy (Softmax Logistic Regression)', fontsize=13)
ax.set_xticks(range(10))
ax.set_ylim(0.7, 1.05)
ax.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, per_class_acc):
    ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('Part-1.2.png', dpi=150, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# (b.iv) Optional: Display a grid of 10 test images with true/predicted labels
# -------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(14, 8))
plt.subplots_adjust(hspace=0.4)
for digit in range(10):
    ax = axes[digit // 5, digit % 5]
    idx = np.where(y_test == digit)[0]
    if len(idx) > 0:
        sample_idx = idx[0]
        img = X_test[sample_idx].reshape(8, 8)
        ax.imshow(img, cmap='gray_r', interpolation='nearest')
        true_label = y_test[sample_idx]
        pred_label = y_test_pred[sample_idx]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', fontsize=10, color=color)
    ax.axis('off')
plt.suptitle('Sample Test Images: One Per Digit Class (Green=Correct, Red=Wrong)', fontsize=13)
plt.savefig('Part_1_b_iv.png', dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------------------------
# Part 2: MULTI-LABEL PROPERTIES
# ---------------------------------------------
print("\n" + "=" * 70)
print("PART 2: Multilabel properties")
print("+" * 70)

multilabel_models = {}
pixel_train_accs = {}
pixel_test_accs = {}

for col_idx, prop_name in enumerate(property_names):
    y_tr = y_mll_train[:, col_idx]
    y_te = y_mll_test[:,col_idx]

    model = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train_scaled,y_tr)
    multilabel_models[prop_name] = model

    #PREDICTIONS
    y_tr_pred = model.predict(X_train_scaled)
    y_te_pred = model.predict(X_test_scaled)

    train_acc_prop = accuracy_score(y_tr, y_tr_pred)
    test_acc_prop = accuracy_score(y_te, y_te_pred)
    pixel_train_accs[prop_name] = train_acc_prop
    pixel_test_accs[prop_name] = test_acc_prop

    print(f"  {prop_name:20s} | Train: {train_acc_prop:.4f} ({train_acc_prop * 100:.2f}%)"
          f"  | Test: {test_acc_prop:.4f} ({test_acc_prop * 100:.2f}%)")

# Classification reports
print("\nDetailed Classification Reports (Test Set):")
for col_idx, prop_name in enumerate(property_names):
    y_te = y_mll_test[:, col_idx]
    y_te_pred = multilabel_models[prop_name].predict(X_test_scaled)
    print(f"\n--- {prop_name} ---")
    print(classification_report(y_te, y_te_pred, target_names=['False (0)', 'True (1)']))

# -----------------------------------
# PART 3: HIERARCHICAL BRIDGE
# -----------------------------------

print("\n" + "=" * 70)
print("PART 3: The Hierarchical Bridge")
print("=" * 70)

X_train_proba = softmax_model.predict_proba(X_train_scaled)
X_test_proba = softmax_model.predict_proba(X_test_scaled)

print(f"\nNew feature matrix X_new (train): {X_train_proba.shape}  (10 class probabilities)")
print(f"New feature matrix X_new (test):  {X_test_proba.shape}")
print(f"\nSample probability vector for first training example:")
print(f"  p(x) = {np.round(X_train_proba[0], 4)}")
print(f"  True label: {y_train[0]}, Predicted: {np.argmax(X_train_proba[0])}")

hierarchical_models = {}
hier_train_accs = {}
hier_test_accs = {}

print(f"\n{'='*75}")
print(f"{'Property':20s} | {'Part 2: Raw Pixels':25s} | {'Part 3: Class Probs':25s}")
print(f"{'':20s} | {'Train':>10s}  {'Test':>10s} | {'Train':>10s}  {'Test':>10s}")
print(f"{'='*75}")

for col_idx, prop_name in enumerate(property_names):
    y_tr = y_mll_train[:, col_idx]
    y_te = y_mll_test[:, col_idx]

    model_hier = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        max_iter=5000,
        random_state=42
    )
    model_hier.fit(X_train_proba, y_tr)
    hierarchical_models[prop_name] = model_hier

    train_acc_hier = accuracy_score(y_tr, model_hier.predict(X_train_proba))
    test_acc_hier = accuracy_score(y_te, model_hier.predict(X_test_proba))
    hier_train_accs[prop_name] = train_acc_hier
    hier_test_accs[prop_name] = test_acc_hier

    print(f"  {prop_name:20s} | {pixel_train_accs[prop_name]:>8.4f}  {pixel_test_accs[prop_name]:>8.4f}"
          f"  | {train_acc_hier:>8.4f}  {test_acc_hier:>8.4f}")

print(f"{'=' * 75}")

# -------------------------------------------------------
# (e) Discussion
# -------------------------------------------------------
print("\n" + "=" * 70)
print("DISCUSSION (Part 3e): Probability Features vs Raw Pixels")
print("=" * 70)
print("""
The results clearly show that using class probability features p(x) from the softmax
model (Part 3) substantially outperforms using raw pixel features (Part 2) for
predicting all three properties (is_even, is_greater_than_5, is_prime).

WHY p(x) IS A STRONGER REPRESENTATION:

1. These properties are deterministic functions of digit identity. If you know a digit
   is "6", you immediately know: is_even=True, is_greater_than_5=True, is_prime=False.
   The probability vector p(x) directly encodes this identity information.

2. The softmax model from Part 1 achieves high accuracy, meaning p(x) is a very
   reliable 10-dimensional summary of the 64-dimensional pixel space. The mapping
   from p(x) to properties is nearly linear, e.g.:
     P(is_even) ~ p(y=0) + p(y=2) + p(y=4) + p(y=6) + p(y=8)

3. Raw pixels (Part 2) force a single binary logistic regression to implicitly learn
   BOTH digit recognition AND the property mapping simultaneously -- a much harder
   task from a 64-dim pixel space where the relationship is highly nonlinear.

4. The hierarchical approach achieves dimensionality reduction (64 -> 10) while
   preserving the most task-relevant information, acting as an information bottleneck.

5. The only scenario where raw pixels could outperform is if the softmax model were
   very inaccurate for certain digits, propagating errors downstream. But since
   Part 1 achieves high accuracy, this effect is minimal.
""")
