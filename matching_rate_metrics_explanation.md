# Matching Rate (MR) Evaluation Metric in NILM

In the context of **Non-Intrusive Load Monitoring (NILM)**, the **Matching Rate (MR)** is a critical metric used to assess the similarity between a predicted appliance power sequence and the actual ground truth sequence. It is particularly useful because it accounts for both the timing and the magnitude of the power consumption events.

## 1. The Formula

The Matching Rate is calculated as the ratio of the sum of the element-wise minimums to the sum of the element-wise maximums across the entire sequence:

$$MR = \frac{\sum_{t=1}^{T} \min(\hat{y}_t, y_t)}{\sum_{t=1}^{T} \max(\hat{y}_t, y_t)}$$

Where:
- $\hat{y}_t$ is the predicted power at time $t$.
- $y_t$ is the actual (ground truth) power at time $t$.
- $T$ is the total length of the sequence.

## 2. Implementation in the Code

In this repository, the Matching Rate is implemented within the `NILMmetrics` class in `src/helpers/metrics.py`:

```python
# Matching Rate (Line 106)
metrics["MR"] = round(
    np.sum(np.minimum(y_hat, y)) / np.sum(np.maximum(y_hat, y)),
    self.round_to,
)
```

## 3. Detailed Logic and Interpretation

The metric provides a value between **0** and **1**:

- **Numerator ($\min$):** Represents the "Overlap" or the portion of energy that was correctly assigned. If the predicted power is higher than ground truth at a specific point, only the ground truth amount is counted as "matched". If the predicted power is lower, only the predicted amount is counted.
- **Denominator ($\max$):** Represents the "Union" or the total power envelope covered by either the prediction or the ground truth. This ensures that any "false alarm" (predicting power when there is none) or "over-prediction" increases the denominator, thereby penalizing the score.

### Example Scenarios:

| Scenario | Prediction ($\hat{y}$) | Ground Truth ($y$) | $\min(\hat{y}, y)$ | $\max(\hat{y}, y)$ | MR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Perfect Match** | [100, 200] | [100, 200] | 300 | 300 | **1.00** |
| **Under-prediction** | [50, 100] | [100, 200] | 150 | 300 | **0.50** |
| **Over-prediction** | [200, 400] | [100, 200] | 300 | 600 | **0.50** |
| **Complete Mismatch** | [0, 0] | [100, 200] | 0 | 300 | **0.00** |

## 4. Why use Matching Rate?

Unlike standard regression metrics like MAE (Mean Absolute Error) or MSE (Mean Squared Error), the Matching Rate is **normalized** and specifically tailored for "signal matching". It effectively captures:
1. **Temporal Alignment**: If the predicted event is shifted in time, the overlap decreases, lowering the MR.
2. **Magnitude Accuracy**: Even if perfectly aligned, a difference in power magnitude reduces the score.

This makes it a robust metric for NILM where both "when" and "how much" are equally important.
