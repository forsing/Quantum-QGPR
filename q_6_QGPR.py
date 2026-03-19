"""
QGPR - Quantum Gaussian Process Regressor
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NOISE_VAR = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def gpr_predict(K, y, noise=NOISE_VAR):
    n = K.shape[0]
    K_noise = K + noise * np.eye(n)
    L = np.linalg.cholesky(K_noise)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean = K @ alpha

    v = np.linalg.solve(L, K)
    var = np.diag(K) - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 0)

    return mean, var


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QGPR po pozicijama (noise={NOISE_VAR}) ---")
    dists_mean = []
    dists_score = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        mean, var = gpr_predict(K, y)
        std = np.sqrt(var)

        score = mean + std
        score = np.maximum(score, 0)
        if score.sum() > 0:
            score /= score.sum()
        dists_score.append(score)

        mean_n = np.maximum(mean, 0)
        if mean_n.sum() > 0:
            mean_n /= mean_n.sum()
        dists_mean.append(mean_n)

        top_idx = np.argsort(score)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{score[i]:.3f}(m={mean[i]:.3f},s={std[i]:.3f})"
            for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo_ucb = greedy_combo(dists_score)
    combo_mean = greedy_combo(dists_mean)

    print(f"\n{'='*50}")
    print(f"Predikcija UCB (QGPR, mean+std, seed={SEED}):")
    print(combo_ucb)
    print(f"Predikcija MEAN (QGPR, samo mean, seed={SEED}):")
    print(combo_mean)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()




"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QGPR po pozicijama (noise=0.01) ---
  Poz 1 [1-33]: 1:0.064(m=0.167,s=0.099) | 2:0.059(m=0.146,s=0.099) | 3:0.055(m=0.129,s=0.099)
  Poz 2 [2-34]: 8:0.044(m=0.085,s=0.099) | 5:0.042(m=0.076,s=0.099) | 9:0.042(m=0.075,s=0.099)
  Poz 3 [3-35]: 13:0.039(m=0.064,s=0.099) | 12:0.039(m=0.062,s=0.099) | 14:0.038(m=0.061,s=0.099)
  Poz 4 [4-36]: 23:0.039(m=0.063,s=0.099) | 21:0.039(m=0.062,s=0.099) | 18:0.039(m=0.062,s=0.099)
  Poz 5 [5-37]: 29:0.039(m=0.065,s=0.099) | 26:0.039(m=0.063,s=0.099) | 27:0.039(m=0.062,s=0.099)
  Poz 6 [6-38]: 33:0.044(m=0.083,s=0.099) | 32:0.043(m=0.081,s=0.099) | 35:0.043(m=0.079,s=0.099)
  Poz 7 [7-39]: 7:0.067(m=0.181,s=0.099) | 38:0.060(m=0.152,s=0.099) | 37:0.055(m=0.132,s=0.099)

==================================================
Predikcija UCB (QGPR, mean+std, seed=39):
[1, 8, 13, 23, 29, 33, 38]
Predikcija MEAN (QGPR, samo mean, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""



"""
QGPR - Quantum Gaussian Process Regressor

Isti kvantni kernel (ZZFeatureMap, fidelity, 5 qubita)
Gaussian Process: Cholesky dekompozicija za egzaktnu GP posteriornu distribuciju
Daje mean (srednja predikcija) i variance (neizvesnost) za svaku vrednost
Dva rezultata: UCB (mean + std - balansira eksploataciju i eksploraciju) i MEAN (cista predikcija)
Deterministicki, bez iterativnog treniranja - najbrzi od svih do sad
"""
