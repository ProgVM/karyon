import random
import torch
import torch.nn.functional as F

def generate_multi_domain_suite(seed=42):
    random.seed(seed)
    
    # Domain 1: Variable Binding & Indirection (Pointers)
    def task_pointers(n=400):
        samples = []
        for _ in range(n):
            vars_all = ['a', 'b', 'c', 'd']
            vals = random.sample(list('123456789'), len(vars_all))
            state = dict(zip(vars_all, vals))
            steps = [f"{k}={state[k]}" for k in vars_all]
            # 1 to 3 swaps
            for _ in range(random.randint(1, 3)):
                dest, src = random.sample(vars_all, 2)
                steps.append(f"{dest}={src}")
                state[dest] = state[src]
            target = random.choice(vars_all)
            expr = "; ".join(steps) + f"; {target}="
            ans = state[target]
            samples.append((expr, ans, 'pointer'))
        return samples

    # Domain 2: Sequence Reversal (Algorithmic / Memory)
    def task_reversal(n=400):
        samples = []
        for _ in range(n):
            length = random.randint(3, 5)
            seq = [str(random.randint(1, 9)) for _ in range(length)]
            expr = "rev " + " ".join(seq) + " = "
            ans = " ".join(reversed(seq))
            samples.append((expr, ans, 'reversal'))
        return samples

    # Domain 3: Arithmetic Carry (3-digit addition)
    def task_addition(n=400):
        samples = []
        for _ in range(n):
            a = random.randint(10, 999)
            b = random.randint(10, 999)
            expr = f"{a} + {b} = "
            ans = str(a + b)
            samples.append((expr, ans, 'addition'))
        return samples

    # Domain 4: State Machine / Parity Tracking (XOR over bitstream)
    def task_parity(n=400):
        samples = []
        for _ in range(n):
            length = random.randint(4, 8)
            bits = [random.choice(['0', '1']) for _ in range(length)]
            parity = str(sum(int(b) for b in bits) % 2)
            expr = "parity " + " ".join(bits) + " = "
            ans = parity
            samples.append((expr, ans, 'parity'))
        return samples

    # Domain 5: Dyck-1 Balanced Parentheses / Stack Memory (depth check)
    def task_brackets(n=400):
        samples = []
        for _ in range(n):
            # generate well-formed or corrupted bracket sequence
            length = random.randint(2, 4) * 2
            # 50% valid, 50% invalid
            is_valid = random.choice([True, False])
            if is_valid:
                # generate valid Dyck
                s = []
                open_cnt = 0
                for _ in range(length):
                    if open_cnt == 0:
                        s.append('(')
                        open_cnt += 1
                    elif open_cnt == length - len(s):
                        s.append(')')
                        open_cnt -= 1
                    else:
                        if random.random() < 0.5:
                            s.append('(')
                            open_cnt += 1
                        else:
                            s.append(')')
                            open_cnt -= 1
                expr = "dyck " + "".join(s) + " = "
                ans = "valid"
            else:
                # corrupted
                s = [random.choice(['(', ')']) for _ in range(length)]
                # check if accidentally valid
                cnt = 0
                acc_valid = True
                for c in s:
                    cnt += 1 if c == '(' else -1
                    if cnt < 0:
                        acc_valid = False
                        break
                if cnt != 0:
                    acc_valid = False
                ans = "valid" if acc_valid else "invalid"
                expr = "dyck " + "".join(s) + " = "
            samples.append((expr, ans, 'dyck'))
        return samples

    return {
        'pointer': task_pointers(400),
        'reversal': task_reversal(400),
        'addition': task_addition(400),
        'parity': task_parity(400),
        'dyck': task_brackets(400)
    }

if __name__ == '__main__':
    suite = generate_multi_domain_suite()
    for domain, samples in suite.items():
        print(f"Domain [{domain:10s}]: {len(samples)} samples. Ex: {samples[0][0]} -> {samples[0][1]}")
