import torch
import itertools
import sys
from torch.multiprocessing import Pool

model_name = 'InterBKT'

ALMOST_ONE = 0.999
ALMOST_ZERO = 0.001

class BKT(torch.nn.Module):
    def __init__(self, step=0.1, bounded=True, best_k0=True, device="cuda:0"):
        super(BKT, self).__init__()
        self.device = device

        # Parameters initialized as tensors
        self.k0 = torch.tensor(ALMOST_ZERO, device=device)
        self.transit = torch.tensor(ALMOST_ZERO, device=device)
        self.guess = torch.tensor(ALMOST_ZERO, device=device)
        self.slip = torch.tensor(ALMOST_ZERO, device=device)
        self.forget = torch.tensor(ALMOST_ZERO, device=device)

        # Limits as tensors
        self.k0_limit = torch.tensor(ALMOST_ONE, device=device)
        self.transit_limit = torch.tensor(ALMOST_ONE, device=device)
        self.guess_limit = torch.tensor(ALMOST_ONE, device=device)
        self.slip_limit = torch.tensor(ALMOST_ONE, device=device)

        self.step = step
        self.best_k0 = best_k0

        if bounded:
            self.k0_limit = torch.tensor(0.85, device=device)
            self.transit_limit = torch.tensor(0.3, device=device)
            self.guess_limit = torch.tensor(0.3, device=device)
            self.slip_limit = torch.tensor(0.1, device=device)

    def _parallel_compute(self, args):
        """
        并行计算单个参数组合的误差。
        :param args: (X, k, t, g, s)
        :return: (k, t, g, s, error)
        """
        X, k, t, g, s = args
        error, _ = self._compute_error(X, k, t, g, s)
        return k, t, g, s, error

    def fit(self, X):
        """
        搜索最佳参数。
        :param X: 数据集（多个序列）
        :return: 最优参数
        """
        # 生成参数组合
        k0s = torch.arange(0, 1 + self.step, self.step)
        transits = torch.arange(0, 1 + self.step, self.step)
        guesses = torch.arange(0, 1 + self.step, self.step)
        slips = torch.arange(0, 1 + self.step, self.step)

        parameter_pairs = list(itertools.product(k0s, transits, guesses, slips))
        args = [(X, k, t, g, s) for k, t, g, s in parameter_pairs]

        # 使用多进程并行化
        print("开始并行化计算...")
        with Pool(processes=8) as pool:  # 根据硬件选择进程数
            results = pool.map(self._parallel_compute, args)

        # 找到误差最小的参数
        min_error = float("inf")
        for k, t, g, s, error in results:
            if error < min_error:
                self.k0, self.transit, self.guess, self.slip = k, t, g, s
                min_error = error

        print(f"最优参数: k0={self.k0}, transit={self.transit}, guess={self.guess}, slip={self.slip}")
        return self.k0.item(), self.transit.item(), self.guess.item(), self.slip.item()

    def _compute_error(self, X, k, t, g, s):
        """
        计算误差函数。
        :param X: 数据集（多个序列）
        :param k: 初始掌握概率
        :param t: 转移概率
        :param g: 猜测概率
        :param s: 疏漏概率
        :return: (误差, None)
        """
        error = torch.tensor(0.0, device='cuda:0')  # 初始化误差
        n = 0  # 记录序列总长度

        for seq in X:
            pred = k  # 初始概率
            for res in seq:
                n += 1
                error += (res - pred) ** 2  # 累加平方误差
                if res == 1.0:
                    p = k * (1 - s) / (k * (1 - s) + (1 - k) * g)
                else:
                    p = k * s / (k * s + (1 - k) * (1 - g))
                k = p + (1 - p) * t
                pred = k * (1 - s) + (1 - k) * g

        return (error / n) ** 0.5, None

    def _find_best_k0(self, X):
        kc_best = torch.mean(torch.tensor([seq[0] for seq in X], device=self.device))
        return kc_best if kc_best > 0 else torch.tensor(0.5, device=self.device)

    def predict(self, X, L, T, G, S):
        return self._compute_error(X, L, T, G, S)

    def inter_predict(self, S, X, k, t, g, s, num_skills):
        all_all_mastery = {}

        for j, skills in S.items():
            skills = list(map(int, skills))
            responses = list(map(int, X[j]))
            last_mastery = torch.zeros(num_skills, device=self.device)

            if len(skills) > 1:
                ini_skill = []
                all_mastery = []
                pL = torch.zeros(len(skills) + 1, device=self.device)

                for i, skill_id in enumerate(skills):
                    if skill_id not in ini_skill:
                        ini_skill.append(skill_id)
                        pL[i] = k[skill_id]
                    else:
                        pL[i] = last_mastery[skill_id]

                    mastery = pL[i]
                    all_mastery.append(mastery.item())

                    res = responses[i]
                    if res == 1.0:
                        pL[i + 1] = pL[i] * (1 - s[skill_id]) / (pL[i] * (1 - s[skill_id]) + (1 - pL[i]) * g[skill_id])
                    else:
                        pL[i + 1] = pL[i] * s[skill_id] / (pL[i] * s[skill_id] + (1 - pL[i]) * (1 - g[skill_id]))
                    pL[i + 1] = pL[i + 1] + (1 - pL[i + 1]) * t[skill_id]
                    last_mastery[skill_id] = pL[i + 1]

                all_all_mastery[j] = all_mastery

        return all_all_mastery

if __name__ == "__main__":
    pass
