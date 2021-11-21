import numpy as np

from typing import Dict, List, Tuple

AdjsMatrix = List[List[int]]
Nodes      = Dict[str, int]


def load_adj_from_file(file_name:str) -> Tuple[AdjsMatrix, Nodes]:
    with open(f"../data/{file_name}.csv", "r") as f:
        lines = [l.strip().upper() for l in f.read().split('\n')]
        nodes = {n: i for i, n in enumerate(lines[0].split(';')[1:])}
        adjs  = []

        for line in lines[1:-1]:
            adjs.append([int(cel) for cel in line.split(';')[1:]])

    return adjs, nodes


def propagation(A : AdjsMatrix, final_k : int) -> AdjsMatrix:
    N   = len(A)
    I   = np.identity(N, dtype=bool)
    A_p = np.transpose(np.array(A, dtype=bool) + I)

    B   = I
    for _ in range(final_k):
        B = np.matmul(A_p, B)  # B(i+1) = A^t * B(i)

    return np.array(B, dtype=int).tolist()


def node_knowledge(B : AdjsMatrix, nodes : Nodes, node : str) -> int:
    return sum(B[nodes[node.upper()]])


def node_popularity(B : AdjsMatrix, nodes : Nodes, node : str) -> int:
    n = nodes[node.upper()]
    return sum(b[n] for b in B)


