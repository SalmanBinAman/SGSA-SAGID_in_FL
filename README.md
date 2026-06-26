# ================================================================================
#  SGSA  vs  SAGID  v2.1  —  Federated Learning attack / defense
#  --------------------------------------------------------------------------------
#  Selective Gradient Suppression Attack (SGSA)   — strengthened, coordinated
#  Sparsity-Aware Gradient Integrity Defense (SAGID) — server expected-gradient model
#
#
#    ATTACK  : coordinated, CONCENTRATED SIGN-FLIP on the IMPORTANT coordinates.
#              All malicious clients hit the top-(k/concentration) coords of their
#              gradient (these overlap under IID -> coordination), flip the sign,
#              and norm-match x boost. They now push the SAME wrong way -> their
#              updates ADD and overwhelm honest mass on those coords -> learning
#              is suppressed/reversed.  `attack_boost` is the strength knob.
#    DEFENSE : SAGID keeps a SERVER EXPECTED-GRADIENT MODEL (EMA of ACCEPTED
#              aggregates = honest history, NOT the suspect round). On the
#              expected-important subspace it scores each client by sign-disagreement
#              + subspace cosine. Robust even when the attack is fully norm-stealthy
#              (boost=1) which a norm filter would miss. Survivors -> norm-clipped
#              mean + randomized dropout.
