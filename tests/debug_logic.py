import sys, re
sys.path.insert(0, '.')
from solvers.logic_solver import _statement_truth

# Honest variant full trace
statements = [("Dan", "Eve is honest"), ("Eve", "Dan is dishonest")]
names = ["Dan", "Eve"]

for mask in range(1 << len(names)):
    world = {name: bool(mask & (1 << idx)) for idx, name in enumerate(names)}
    print(f"mask={mask}, world={world}")
    
    truth_values = [_statement_truth(text, world) for _, text in statements]
    print(f"  truth_values={truth_values}")
    
    if None in truth_values:
        print("  -> None, bail")
        continue
    
    consistent = True
    for idx, (speaker, _) in enumerate(statements):
        if world[speaker] != truth_values[idx]:
            consistent = False
            print(f"  -> INCONSISTENT: {speaker} truthful={world[speaker]} != statement_truth={truth_values[idx]}")
            break
    if not consistent:
        continue
    
    n_truthful = sum(1 for v in world.values() if v)
    print(f"  -> CONSISTENT! n_truthful={n_truthful}")
