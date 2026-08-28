# Research Decisions and Sources

## Specification structure

GitHub Spec Kit separates requirements, technical planning, and executable tasks.
Its templates emphasize prioritized user stories, independent test criteria,
measurable success criteria, exact file paths, dependency ordering, and an MVP-
first implementation strategy. This feature uses the same separation without
installing Spec Kit into the repository.

- Spec template: https://github.com/github/spec-kit/blob/main/templates/spec-template.md
- Task-generation guidance: https://github.com/github/spec-kit/blob/main/templates/commands/tasks.md
- Workflow overview: https://github.com/github/spec-kit

Normative requirement words follow RFC 2119:

- https://www.rfc-editor.org/info/rfc2119/

## Test isolation

The current code selects model states using repeated CIFAR test-set scores.
Scikit-learn's official guidance states that test data must not be used to make
model choices because this creates optimistic performance estimates. Protocol v2
therefore fixes training duration before execution and evaluates once afterward.

- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

## Reproducibility boundary

PyTorch documents that exact reproducibility is not guaranteed across releases,
commits, platforms, or CPU/GPU devices. It recommends seeding Python, NumPy, and
PyTorch; disabling cuDNN benchmarking; using deterministic algorithms; and
seeding DataLoader workers and generators. Protocol v2 follows those controls and
records the environment rather than promising cross-platform bitwise identity.

- https://docs.pytorch.org/docs/stable/notes/randomness.html
- https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html

## Baseline provenance

The authors' repository is the canonical implementation reference for TypiClust,
ProbCover, and DCoM. ProbCover is not admitted to the primary benchmark until its
radius-selection policy and a fixed compatibility fixture are documented.

- Official repository: https://github.com/avihu111/TypiClust
- ProbCover paper: https://arxiv.org/abs/2205.11320

MaxHerding is a later coverage baseline designed to reduce ProbCover's sensitivity
to its radius hyperparameter. It is a potential post-MVP comparison, not required
for Gate B.

- https://arxiv.org/abs/2407.12212

## Resolved decisions

| ID | Decision | Reason |
|---|---|---|
| RD-001 | Fixed epochs; no validation split from queried labels | Preserves the low-label budget and prevents test selection |
| RD-002 | One test evaluation per active-learning round | A learning curve requires round-level reporting, but test results never control future selection logic |
| RD-003 | Explicit round query sizes | Removes ambiguous total-budget splitting and guarantees nested trajectories |
| RD-004 | Shared clustering/training seeds across paired methods | Reduces avoidable paired-comparison variance |
| RD-005 | Method-specific selector seed | Prevents RNG call-order coupling across algorithms |
| RD-006 | Atomic per-run JSON | Supports validation and safe resumption |
| RD-007 | New protocol-v2 result root | Prevents accidental mixing with legacy best-test metrics |
| RD-008 | Frozen embeddings as MVP evaluator | Makes five- and ten-seed confirmation computationally feasible |
| RD-009 | CIFAR-100 after CIFAR-10 Gate C | Avoids expensive expansion before the method is validated |

| RD-010 | ProbCover uses K-Means pseudo-label purity with `alpha=0.95` | This is the paper's explicitly label-free radius heuristic |
| RD-011 | ProbCover searches an explicit decimal grid from 0.05 to 1.50 by 0.01 | Makes the finite approximation deterministic and visible in the config |

ProbCover radius selection is therefore resolved: cluster train embeddings into
the known number of dataset classes, treat assignments as pseudo-labels, and
choose the largest configured radius for which at least 95% of balls are pure.
The radius is never selected using ground-truth labels or downstream accuracy.
Compatibility tests still compare the implementation with the official code on
a fixed fixture.
