================================================================================
#  SGSA  vs  SAGID  v2.1  —  Federated Learning attack / defense
--------------------------------------------------------------------------------
Federated learning (FL) is a machine learning technique whcih trains an artificial intelligence model across multiple devices
or servers. This technique allows many clients to train a shared model collaboratively, but does not expose private data. However,
the server’s inability to inspect client updates makes it vulnerable to adversarial participants. Communication efficient FL transmits
only the top-k most significant gradient coordinates, yet the security implications of this gradient-selection step have received little
attention. Existing attacks predominantly poison data, embed backdoor triggers, or inflate gradient magnitudes, and existing defenses
correspondingly screen for abnormal magnitudes or suspicious behavior both of which overlook manipulation of the sparsity pattern
itself. To close this gap, we propose the Selective Gradient Suppression Attack (SGSA), in which colluding clients invert and amplify the
model’s most important gradient coordinates while keeping each update magnitude plausible, and the Sparsity Aware Gradient Integrity
Defense (SAGID), which maintains a server-side expected gradient model and scores clients by sign disagreement and directional
consistency before robustly aggregating the survivors. We implement both in PyTorch and evaluate them on MNIST with 20 clients
(40% malicious), a compact CNN, and top-k FedAvg. SGSA collapses global test accuracy from 97.67% to 10.10%, while SAGID
restores it to 97.45%—recovering 99.7–99.8% of the lost performance and detects every malicious client with an F1 score and ROCAUC
of 1.00. The results show that gradient sparsification is an exploitable attack surface and that direction and sign-aware defenses
can neutralize such stealthy attacks.
Index Terms Federated learning, gradient
