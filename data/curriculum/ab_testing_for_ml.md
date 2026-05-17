# A/B Testing for ML: Bandits, CUPED, IPS, and Doubly-Robust Estimators

**Area F — Production ML Systems | Learning Memory OS Curriculum**

---

## 1. Why A/B Testing is Different for ML Systems

A/B testing for ML systems is harder than for UI changes. ML models affect the full system in non-obvious ways: changing a recommendation model affects what users click, which changes the training data for the next model iteration, which changes future recommendations — a feedback loop that can take weeks to stabilize. Model changes also have latency effects, cost effects, and rare-but-critical failure modes that simple metric comparisons miss.

Key differences from product A/B testing:
1. **Feedback loops**: model updates affect training data; effects compound over time
2. **Novelty effects**: new models look better initially because they recommend unexplored content
3. **Long-tail effects**: rare but high-value outcomes (large purchases, churn events) require large samples
4. **Multiple metrics**: ML systems optimize for composite objectives; must track all of them

```mermaid
flowchart TB
  Hypothesis[Hypothesis\n"new ranking model improves CTR"] --> SampleSize[Sample Size Calculation\npower analysis]
  SampleSize --> Assignment[User Assignment\ndeterministic hashing]
  Assignment --> Control[Control Group\nold model]
  Assignment --> Treatment[Treatment Group\nnew model]
  Control --> Metrics[Metric Collection\nclicks, conversions, latency]
  Treatment --> Metrics
  Metrics --> Analysis[Statistical Analysis\nt-test, Mann-Whitney]
  Analysis --> Decision[Ship / Kill / Continue]
```

---

## 2. Sample Size Calculation

Before running an experiment, determine the minimum sample size to detect a given effect size at the desired statistical power.

```python
import numpy as np
from scipy import stats

def minimum_sample_size(baseline_rate: float,
                          minimum_detectable_effect: float,
                          alpha: float = 0.05,
                          power: float = 0.80,
                          two_tailed: bool = True) -> int:
    """
    Compute minimum sample size per group for a proportion test.
    
    baseline_rate: current conversion/click rate (e.g. 0.05 = 5%)
    minimum_detectable_effect: absolute improvement (e.g. 0.005 = +0.5%)
    alpha: false positive rate (Type I error)
    power: 1 - Type II error (probability of detecting true effect)
    """
    treatment_rate = baseline_rate + minimum_detectable_effect
    # Z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_tailed else 1))
    z_power = stats.norm.ppf(power)
    
    p1, p2 = baseline_rate, treatment_rate
    p_pooled = (p1 + p2) / 2
    
    # Sample size formula for two proportions
    n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
           z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 /
          (p1 - p2) ** 2)
    return int(np.ceil(n))


def compute_test_duration_days(n_per_group: int,
                                 daily_users: int,
                                 traffic_fraction: float = 0.10) -> float:
    """
    How many days to run the experiment to collect enough samples.
    traffic_fraction: fraction of users assigned to experiment (both groups).
    """
    users_per_day_in_experiment = daily_users * traffic_fraction
    users_per_group_per_day = users_per_day_in_experiment / 2
    return n_per_group / users_per_group_per_day


# Example: email click rate experiment
n = minimum_sample_size(
    baseline_rate=0.05,      # 5% click rate
    minimum_detectable_effect=0.005,  # detect +0.5% absolute improvement
    alpha=0.05, power=0.80,
)
print(f"Required: {n:,} users per group")  # ~30,000

days = compute_test_duration_days(n, daily_users=500_000, traffic_fraction=0.10)
print(f"Experiment duration: {days:.1f} days at 10% traffic allocation")
```

---

## 3. Deterministic User Assignment

Experiments need to be: (1) reproducible, (2) consistent per user across sessions, (3) balanced across groups.

```python
import hashlib
from typing import Tuple

def hash_assignment(user_id: str, experiment_id: str,
                     salt: str = "",
                     n_buckets: int = 10000) -> int:
    """
    Deterministic bucket assignment via MD5 hash.
    Returns an integer in [0, n_buckets).
    Same user_id + experiment_id always returns same bucket.
    """
    key = f"{salt}:{experiment_id}:{user_id}"
    digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return digest % n_buckets

def assign_variant(user_id: str, experiment_id: str,
                    splits: dict,  # {"control": 0.5, "treatment": 0.5}
                    salt: str = "") -> str:
    """
    Assign a user to a variant based on hash bucket.
    splits: dict of {variant_name: fraction}, must sum to 1.0.
    Returns variant name.
    """
    assert abs(sum(splits.values()) - 1.0) < 1e-6, "Splits must sum to 1.0"
    bucket = hash_assignment(user_id, experiment_id, salt)
    cumulative = 0
    for variant, fraction in splits.items():
        cumulative += fraction
        if bucket < int(cumulative * 10000):
            return variant
    return list(splits.keys())[-1]  # fallback

# Test: verify balance
def verify_assignment_balance(n_users: int = 100_000,
                               experiment_id: str = "test_exp_001") -> dict:
    """Verify that hash assignment creates balanced groups."""
    counts = {}
    for i in range(n_users):
        variant = assign_variant(
            f"user_{i}", experiment_id,
            splits={"control": 0.5, "treatment": 0.5}
        )
        counts[variant] = counts.get(variant, 0) + 1
    fractions = {k: v/n_users for k, v in counts.items()}
    print(f"Assignment balance: {fractions}")  # should be ~0.5 each
    return fractions
```

---

## 4. Statistical Testing

```python
import numpy as np
from scipy import stats

def two_proportion_ztest(n_control: int, clicks_control: int,
                          n_treatment: int, clicks_treatment: int,
                          alpha: float = 0.05) -> dict:
    """
    Two-sided z-test for difference in proportions.
    H0: p_treatment = p_control
    Returns p-value, confidence interval, and recommendation.
    """
    p_c = clicks_control / max(n_control, 1)
    p_t = clicks_treatment / max(n_treatment, 1)
    diff = p_t - p_c
    
    # Pooled standard error under H0
    p_pooled = (clicks_control + clicks_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
    
    z = diff / max(se, 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # 95% CI for the difference
    ci_margin = 1.96 * np.sqrt(p_c*(1-p_c)/n_control + p_t*(1-p_t)/n_treatment)
    ci = (diff - ci_margin, diff + ci_margin)
    
    return {
        "p_control": p_c,
        "p_treatment": p_t,
        "absolute_lift": diff,
        "relative_lift": diff / max(p_c, 1e-9),
        "z_statistic": z,
        "p_value": p_value,
        "ci_95": ci,
        "significant": p_value < alpha,
        "recommendation": "ship" if (p_value < alpha and diff > 0) else "kill" if (p_value < alpha and diff < 0) else "continue",
    }

def welch_ttest_continuous(control_values: np.ndarray,
                             treatment_values: np.ndarray,
                             alpha: float = 0.05) -> dict:
    """
    Welch's t-test for continuous metrics (revenue, session duration).
    Does not assume equal variance between groups.
    """
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values,
                                        equal_var=False)
    diff = treatment_values.mean() - control_values.mean()
    return {
        "control_mean": control_values.mean(),
        "treatment_mean": treatment_values.mean(),
        "absolute_lift": diff,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }
```

---

## 5. CUPED: Variance Reduction via Pre-Experiment Covariate

CUPED (Controlled-experiment Using Pre-Experiment Data) reduces experiment noise by removing variance explained by pre-experiment behavior (e.g., last week's activity). This effectively increases statistical power — you need fewer users to reach significance.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def apply_cuped(y_control: np.ndarray, y_treatment: np.ndarray,
                 x_control: np.ndarray, x_treatment: np.ndarray) -> Tuple:
    """
    Apply CUPED variance reduction.
    y: outcome metric (e.g., revenue this week)
    x: pre-experiment covariate (e.g., revenue last week)
    
    CUPED-adjusted outcome: Y_adj = Y - theta * (X - E[X])
    theta = Cov(Y, X) / Var(X) -- estimated from pooled data
    
    Returns: (y_control_adj, y_treatment_adj, theta, variance_reduction_pct)
    """
    y_all = np.concatenate([y_control, y_treatment])
    x_all = np.concatenate([x_control, x_treatment])
    
    # Estimate theta via OLS: theta = Cov(Y,X) / Var(X)
    x_centered = x_all - x_all.mean()
    theta = np.cov(y_all, x_all)[0, 1] / np.var(x_all)
    
    # Apply adjustment
    x_c_mean = x_control.mean()
    x_t_mean = x_treatment.mean()
    y_control_adj = y_control - theta * (x_control - x_c_mean)
    y_treatment_adj = y_treatment - theta * (x_treatment - x_t_mean)
    
    # Compute variance reduction
    var_before = np.var(np.concatenate([y_control, y_treatment]))
    var_after = np.var(np.concatenate([y_control_adj, y_treatment_adj]))
    var_reduction_pct = (1 - var_after / var_before) * 100
    
    return y_control_adj, y_treatment_adj, theta, var_reduction_pct


def demonstrate_cuped():
    """Show CUPED reduces variance, enabling earlier significance detection."""
    rng = np.random.RandomState(42)
    n = 1000  # per group
    
    # Pre-experiment covariate (last week's revenue, correlated with this week's)
    x_control = rng.exponential(50, n)
    x_treatment = rng.exponential(50, n)
    
    # This week's revenue: correlated with last week + small treatment effect
    true_lift = 2.0
    y_control = x_control * 0.8 + rng.exponential(20, n)
    y_treatment = x_treatment * 0.8 + rng.exponential(20, n) + true_lift
    
    # Without CUPED
    result_raw = welch_ttest_continuous(y_control, y_treatment)
    print(f"Without CUPED: p={result_raw['p_value']:.4f}, significant={result_raw['significant']}")
    
    # With CUPED
    yc_adj, yt_adj, theta, var_red = apply_cuped(y_control, y_treatment, x_control, x_treatment)
    result_cuped = welch_ttest_continuous(yc_adj, yt_adj)
    print(f"With CUPED: p={result_cuped['p_value']:.4f}, significant={result_cuped['significant']}")
    print(f"Variance reduction: {var_red:.1f}%")
```

---

## 6. Thompson Sampling: Multi-Armed Bandit for Online Experimentation

When an A/B test has many variants and you want to minimize regret (cost of sending traffic to bad variants), use a bandit algorithm instead.

```python
import numpy as np
from typing import List, Tuple

class ThompsonSamplingBandit:
    """
    Thompson Sampling for Bernoulli rewards (click/no-click).
    Uses Beta-Bernoulli conjugate model.
    """
    def __init__(self, n_arms: int, prior_alpha: float = 1.0,
                  prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * prior_alpha  # successes + prior
        self.beta = np.ones(n_arms) * prior_beta    # failures + prior
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self.n_successes = np.zeros(n_arms, dtype=int)

    def select_arm(self) -> int:
        """Sample from each arm's posterior, return arm with highest sample."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: int):
        """Update posterior with observed reward (0 or 1)."""
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
        self.n_pulls[arm] += 1
        self.n_successes[arm] += reward

    def expected_ctrs(self) -> np.ndarray:
        """Posterior mean CTR for each arm."""
        return self.alpha / (self.alpha + self.beta)

    def credible_intervals(self, confidence: float = 0.95) -> List[Tuple]:
        """95% credible intervals for each arm's CTR."""
        lo = (1 - confidence) / 2
        hi = 1 - lo
        return [
            (float(np.quantile(np.random.beta(a, b, 10000), lo)),
             float(np.quantile(np.random.beta(a, b, 10000), hi)))
            for a, b in zip(self.alpha, self.beta)
        ]


def simulate_bandit(true_ctrs: List[float], n_rounds: int = 10_000,
                     seed: int = 42) -> dict:
    """
    Simulate Thompson Sampling bandit and compare to pure A/B test.
    Returns regret (missed clicks) for bandit vs. A/B test.
    """
    rng = np.random.RandomState(seed)
    n_arms = len(true_ctrs)
    optimal_ctr = max(true_ctrs)
    
    # Thompson Sampling
    bandit = ThompsonSamplingBandit(n_arms)
    bandit_regret = 0.0
    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = int(rng.random() < true_ctrs[arm])
        bandit.update(arm, reward)
        bandit_regret += optimal_ctr - true_ctrs[arm]
    
    # Pure A/B test (uniform allocation)
    ab_regret = sum(optimal_ctr - ctr for ctr in true_ctrs) * n_rounds / n_arms
    
    print(f"True CTRs: {true_ctrs}")
    print(f"Bandit estimated CTRs: {bandit.expected_ctrs().tolist()}")
    print(f"Bandit regret: {bandit_regret:.1f}")
    print(f"A/B test regret: {ab_regret:.1f}")
    print(f"Regret reduction: {(1 - bandit_regret/ab_regret)*100:.1f}%")
    return {"bandit_regret": bandit_regret, "ab_regret": ab_regret}
```

---

## 7. Inverse Propensity Scoring (IPS) for Offline Evaluation

When you can't run an online experiment (e.g., a new recommendation model), IPS lets you evaluate it offline using historical log data.

```python
import numpy as np

def ips_estimator(logged_actions: np.ndarray,
                   logged_rewards: np.ndarray,
                   logging_policy_probs: np.ndarray,
                   target_policy_probs: np.ndarray) -> float:
    """
    Inverse Propensity Score estimator for counterfactual evaluation.
    
    Estimates the expected reward of the target policy using data
    collected under the logging policy.
    
    logged_actions: (N,) actions taken by logging policy
    logged_rewards: (N,) rewards observed
    logging_policy_probs: (N,) P(action | context, logging_policy)
    target_policy_probs: (N,) P(action | context, target_policy)
    
    IPS = (1/N) * sum_i [ (pi_target(a_i) / pi_logging(a_i)) * r_i ]
    """
    # Importance weights: ratio of target to logging probability
    weights = target_policy_probs / np.maximum(logging_policy_probs, 1e-6)
    # Clip weights to reduce variance (common in practice: clip at 5-10x)
    weights = np.clip(weights, 0, 10.0)
    return float(np.mean(weights * logged_rewards))


def doubly_robust_estimator(logged_actions: np.ndarray,
                              logged_rewards: np.ndarray,
                              logging_policy_probs: np.ndarray,
                              target_policy_probs: np.ndarray,
                              reward_model_predictions: np.ndarray) -> float:
    """
    Doubly Robust (DR) estimator: combines IPS with a reward model.
    
    DR = (1/N) * sum_i [
        mu_hat(x_i, a_i)           # reward model prediction
        + (pi_target/pi_logging) * (r_i - mu_hat(x_i, a_i))  # IPS correction
    ]
    
    DR is unbiased if EITHER the reward model OR the propensity model is correct.
    This "doubly robust" property makes it more reliable than either alone.
    """
    weights = target_policy_probs / np.maximum(logging_policy_probs, 1e-6)
    weights = np.clip(weights, 0, 10.0)
    dm_term = reward_model_predictions  # direct model
    ips_correction = weights * (logged_rewards - reward_model_predictions)
    return float(np.mean(dm_term + ips_correction))
```

---

## 8. Common Misconceptions

**Misconception: "You should run A/B tests until you see p < 0.05, then stop."**
Correction: This is p-hacking / optional stopping, a serious methodological error. Checking significance repeatedly and stopping when significant inflates the false positive rate dramatically. Fix: pre-register the sample size, use sequential testing methods (SPRT, always-valid confidence intervals), or Bayesian approaches with explicit early-stopping rules.

**Misconception: "Statistical significance means the effect is large enough to matter."**
Correction: Statistical significance only means the effect is distinguishable from zero given the sample size. With 10M users, a 0.001% CTR improvement will be statistically significant but practically meaningless. Always report practical significance (effect size, confidence interval) alongside p-values. Minimum Detectable Effect (MDE) should be set to business-meaningful thresholds.

**Misconception: "Novelty effects are just noise and will average out."**
Correction: Novelty effects are real and can last weeks. A new feature or model that users haven't seen before gets more engagement simply because it's novel. This inflates treatment effect estimates. The fix: run experiments for at least 2-4 weeks, or use holdout groups that start receiving the treatment at different times to separate novelty from true lift.

**Misconception: "Thompson Sampling is always better than A/B testing for model selection."**
Correction: Thompson Sampling minimizes regret during the experiment but may not achieve the statistical rigor needed to confidently choose a winner. For high-stakes decisions (major model changes), traditional A/B testing with pre-specified power and duration is more appropriate. Bandits are better for continuous optimization with many variants (ad creative selection, hyperparameter search).

**Misconception: "IPS estimators give unbiased estimates for any policy."**
Correction: IPS assumes the logging policy has positive probability for all actions the target policy might take (support condition). If the target policy takes actions the logging policy never explored (probability = 0), IPS is undefined. In practice, use clipped IPS with a moderate weight cap (5-10x) to control variance, and use DR estimators when you have a reward model.

---

## 9. Hands-On Labs

### Exercise 1: Sample Size Calculator

**Goal**: Build a sample size calculator and demonstrate the effect of MDE and baseline rate.

**Starter code**:
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def sample_size_sweep(baseline_rates: list, mdes: list,
                       alpha: float = 0.05, power: float = 0.80) -> np.ndarray:
    """
    Compute sample sizes for a grid of baseline_rates x MDEs.
    Returns (len(baseline_rates), len(mdes)) matrix.
    """
    results = np.zeros((len(baseline_rates), len(mdes)))
    for i, br in enumerate(baseline_rates):
        for j, mde in enumerate(mdes):
            results[i, j] = minimum_sample_size(br, mde, alpha, power)
    return results

def plot_sample_size_heatmap(results: np.ndarray, baseline_rates: list, mdes: list):
    """Plot sample size as a heatmap: baseline_rate x MDE."""
    # TODO: implement heatmap visualization
    pass
```

**Acceptance criteria**: The heatmap shows that (a) smaller MDE requires exponentially more samples, and (b) extreme baseline rates (near 0 or 1) require fewer samples than mid-range rates for the same relative lift.
**Stretch**: Extend to compute the required experiment duration (days) given daily active users = 1M and 10% traffic allocation. Overlay duration contours on the heatmap.

---

### Exercise 2: CUPED Variance Reduction Simulation

**Goal**: Show that CUPED reduces variance and achieves significance with fewer users.

**Starter code**:
```python
import numpy as np
from scipy import stats

def simulate_cuped_power(n_range: list, true_lift: float = 2.0,
                          correlation: float = 0.7,
                          n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Simulate experiment power with and without CUPED for each n in n_range.
    At each n, run n_sims experiments and report fraction significant (= power).
    """
    rng = np.random.RandomState(seed)
    power_raw = []
    power_cuped = []
    for n in n_range:
        sig_raw = 0
        sig_cuped = 0
        for _ in range(n_sims):
            # Generate correlated pre/post data
            x_c = rng.exponential(50, n)
            x_t = rng.exponential(50, n)
            y_c = correlation * x_c + rng.exponential(20, n)
            y_t = correlation * x_t + rng.exponential(20, n) + true_lift
            # Raw t-test
            _, p_raw = stats.ttest_ind(y_c, y_t, equal_var=False)
            sig_raw += (p_raw < 0.05)
            # CUPED
            yc_adj, yt_adj, _, _ = apply_cuped(y_c, y_t, x_c, x_t)
            _, p_cuped = stats.ttest_ind(yc_adj, yt_adj, equal_var=False)
            sig_cuped += (p_cuped < 0.05)
        power_raw.append(sig_raw / n_sims)
        power_cuped.append(sig_cuped / n_sims)
    return {"n_range": n_range, "power_raw": power_raw, "power_cuped": power_cuped}
```

**Acceptance criteria**: CUPED achieves 80% power at a sample size 30-50% smaller than without CUPED when the pre/post correlation is ≥ 0.5.
**Stretch**: Implement CUPED for continuous metrics using a ridge regression covariate model (multivariate X). Measure variance reduction as a function of correlation strength.

---

### Exercise 3: Thompson Sampling Bandit vs. A/B Test

**Goal**: Simulate a 5-variant experiment with Thompson Sampling and compare regret to equal-split A/B testing.

**Starter code**:
```python
import numpy as np

def compare_bandit_vs_ab(true_ctrs: list = [0.05, 0.06, 0.055, 0.045, 0.065],
                          n_rounds: int = 100_000, seed: int = 42) -> dict:
    """
    Run both strategies for n_rounds rounds.
    Return regret, final CTR estimates, and convergence speed (round when best arm > 0.9 probability).
    """
    # TODO: implement Thompson Sampling bandit and A/B test comparison
    # Return dict with cumulative regret curves for both strategies
    pass

def plot_regret_curves(result: dict):
    """Plot cumulative regret over time for bandit vs A/B."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    # TODO: plot both curves
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret (Missed Clicks)")
    plt.title("Thompson Sampling vs. Equal-split A/B Test")
    plt.legend()
    plt.savefig("/tmp/bandit_vs_ab.png")
```

**Acceptance criteria**: Thompson Sampling accumulates ≤ 60% of the regret of equal-split A/B testing over 100K rounds when the best arm has ≥ 1% absolute CTR advantage over the second best. Plot the cumulative regret curves.
**Stretch**: Implement UCB1 (Upper Confidence Bound) as a third comparison. Plot convergence speed (how quickly each algorithm identifies the best arm with 95% confidence).

---
