## Why normalizing the input before feeding into model ?
### 🔹 What Is Normalization

Normalization means adjusting each input feature so that it has:

* **Mean (average) = 0**
* **Standard deviation = 1**

This is done using the formula:

x' = (x - μ) / σ

where

* ( \mu ) = average value of that feature
* ( \sigma ) = spread (standard deviation) of that feature

---

### 🔹 Why We Need It

Different features in your dataset can have very different ranges.
Example:

* Height (in meters): 1 – 2
* Income (in dollars): 10,000 – 100,000

If we directly feed such data into a model:

* The **larger-valued feature** (income) will dominate the learning.
* The **smaller-valued feature** (height) will hardly affect the model.

So the model gives more importance to big-number features just because of their scale — not because they are more useful.
Normalization fixes this problem by making all features **comparable in scale**.

---

### 🔹 What Happens Without Normalization

During training:

* The model updates its weights based on the size of each feature.
* A feature with large numbers produces **large weight updates**.
* A feature with small numbers produces **tiny weight updates**.

Because of this imbalance:

* The model **keeps adjusting unevenly**, moving a lot for one feature and little for another.
* It **takes longer** for the training to reach the best result (the minimum loss).
* Sometimes it may **oscillate** (keep jumping around) instead of smoothly improving.

---

### 🔹 What Happens With Normalization

When we normalize:

* All features are on **a similar scale**.
* The model treats each feature **fairly and equally**.
* Training becomes **stable and faster**.
* The model **converges (reaches the best result)** more quickly.

---


### 🔹 Key Idea

> Normalization makes all input features equal in scale so that the model can learn efficiently, quickly, and stably.

## 🧠 Batch Normalization — Crisp Interview Notes

### 🔹 Why We Need Batch Normalization

* Hidden layer inputs (activations) behave just like raw inputs — they can have very different ranges.
* This causes **internal covariate shift**: changing activations during training make learning unstable.
* **Batch Normalization (BN)** normalizes these activations, keeping them on a consistent scale across layers.

➡️ Same logic as input normalization, but applied **inside** the network.

---

### 🔹 How Batch Norm Works

Batch Norm is inserted **between two layers** and normalizes the outputs (activations) of one before passing them to the next.

#### Steps per mini-batch:

1. **Input Activations:** Take activations from the previous layer.
2. **Compute Mean & Variance:** For each feature in the mini-batch.
3. **Normalize:**
   ( \hat{x} = (x - \mu_{batch}) / \sqrt{\sigma_{batch}^2 + \epsilon} )
4. **Scale and Shift:**
   ( y = \gamma \hat{x} + \beta )

   * **γ (gamma)**: scale parameter (learnable)
   * **β (beta)**: shift parameter (learnable)

These let the network restore any necessary distribution (BN doesn’t restrict all activations to have zero mean, unit variance).

---

### 🔹 Parameters in Batch Norm

| Type          | Parameters                           | Description                                             |
| ------------- | ------------------------------------ | ------------------------------------------------------- |
| Learnable     | **γ (gamma)**, **β (beta)**          | Allow re-scaling and shifting of normalized activations |
| Non-learnable | **Moving Mean**, **Moving Variance** | Track running averages for inference                    |

Each BatchNorm layer keeps its own copy of these parameters.

---

### 🔹 During Training vs Inference

* **Training:** Uses mini-batch mean and variance. Also updates running averages (EMA) using a momentum parameter.
* **Inference:** Uses stored **moving mean** and **moving variance** instead of batch statistics (since only one sample is input).

---

### 🔹 Benefits

✅ Reduces **internal covariate shift**
✅ Allows **higher learning rates**
✅ Improves **convergence speed**
✅ Acts as a form of **regularization** (reduces overfitting)
✅ Makes the network **less sensitive to initialization**

---

### 🔹 Placement in Architecture

Can be placed:

* **Before activation** (as in the original paper)
* **After activation** (common in practice)

Both work — depends on experiment and framework convention.

---

### 🔹 Key Interview Points

* **Why BN?** → Stabilizes learning by normalizing hidden activations.
* **What does it learn?** → γ (scale) and β (shift).
* **Training vs Inference?** → Uses batch stats during training, moving averages during inference.
* **Advantages?** → Faster convergence,
* regularization(Batch Norm regularizes by adding mini-batch noise and stabilizing activations, helping the model generalize better — even without Dropout.),
* robustness to initialization(Batch Normalization makes a network robust to initialization because it normalizes activations to zero mean and unit variance, preventing exploding or vanishing values. Even if weights start poorly, BN rescales activations automatically, keeping gradients stable and training smooth from the beginning)
* **Equation?** → `y = γ * ((x - μ)/√(σ²+ε)) + β`

---

### 🔹 One-line Summary

> Batch Normalization keeps activations stable across layers, enabling faster, smoother, and more reliable training.
