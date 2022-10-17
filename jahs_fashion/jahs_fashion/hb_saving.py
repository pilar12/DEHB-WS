import numpy as np
import pprint

# Script to compute the number of epochs given function eval and budget
if __name__ == "__main__":
    eta = 3
    min_budget = 3
    max_budget = 27
    max_func_evals = 60
    retrain_budget = 50

    # precompute some HB stuff
    max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
    budgets = np.ceil(
        max_budget * np.power(eta, -np.linspace(max_SH_iter - 1, 0, max_SH_iter))
    )

    pprint.pprint(
        {
            "eta": eta,
            "min_budget": min_budget,
            "max_budget": max_budget,
            "budgets": budgets,
            "max_SH_iter": max_SH_iter,
        }
    )
    effective_training_epochs = 0
    summ = 0
    func_evals = 0
    samples = 0
    for iteration in range(max_SH_iter):
        s = max_SH_iter - 1 - (iteration % max_SH_iter)
        n0 = int(np.floor((max_SH_iter) / (s + 1)) * eta**s)
        ns = [max(int(n0 * (eta ** (-i))), 1) for i in range(s + 1)]
        samples += sum(ns)
        budget_s = budgets[(-s - 1) :]
        effective_budget = [budget_s[0]]
        for i in range(1, len(budget_s)):
            effective_budget.append(budget_s[i] - budget_s[i - 1])
        effective_budget = np.array(effective_budget)
        effective_training_epochs += np.sum(effective_budget)
        print("__________________")
        print("effective_budget_" + str(s) + ": " + str(effective_budget))
        print("s: " + str(s))
        print("n0: " + str(n0))
        print("ns: " + str(ns))
        func_evals += sum(ns)
        temp_sum = 0
        for j in range(len(ns)):
            summ += ns[j] * effective_budget[j]
            temp_sum += ns[j] * effective_budget[j]
            print(
                "n: "
                + str(ns[j])
                + " for "
                + str(effective_budget[j])
                + " each for a total of "
                + str(ns[j] * effective_budget[j])
                + " epochs"
            )
        print("total epochs in this iteration: " + str(temp_sum))
        print("total func evals in this iteration: " + str(sum(ns)))
        print("current total epochs run: " + str(summ))
        print("current total samples: " + str(samples))
        print("current total func evals: " + str(func_evals))
        print("current total efective_training: " + str(effective_training_epochs))
    print("__________________")
    print("total epochs run: " + str(summ))
    print("total func evals: " + str(func_evals))
    print("total configs sampled: " + str(samples))
    print("total total efective_training: " + str(effective_training_epochs))
