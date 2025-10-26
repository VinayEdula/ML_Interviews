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
