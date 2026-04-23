def make_csr_graph(row_offsets, column_indices, node_labels, edge_labels, num_nodes=None):
    if num_nodes is None:
        num_nodes = len(node_labels)

    return {
        "row_offsets": list(row_offsets),
        "column_indices": list(column_indices),
        "node_labels": list(node_labels),
        "edge_labels": list(edge_labels),
        "num_nodes": num_nodes,
    }


def toy_two_node_graph():
    return make_csr_graph(
        row_offsets=[0, 1, 2],
        column_indices=[1, 0],
        node_labels=[0, 1],
        edge_labels=[0, 0],
        num_nodes=2,
    )

def make_csr_graph(row_offsets, column_indices, node_labels, edge_labels, num_nodes=None):
    if num_nodes is None:
        num_nodes = len(node_labels)

    return {
        "row_offsets": list(row_offsets),
        "column_indices": list(column_indices),
        "node_labels": list(node_labels),
        "edge_labels": list(edge_labels),
        "num_nodes": num_nodes,
    }


def toy_two_node_graph():
    return make_csr_graph(
        row_offsets=[0, 1, 2],
        column_indices=[1, 0],
        node_labels=[0, 1],
        edge_labels=[0, 0],
        num_nodes=2,
    )


def chain_graph(num_nodes: int):
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")

    row_offsets = [0]
    column_indices = []
    node_labels = []
    edge_labels = []

    for i in range(num_nodes):
        neighbors = []
        if i - 1 >= 0:
            neighbors.append(i - 1)
        if i + 1 < num_nodes:
            neighbors.append(i + 1)

        column_indices.extend(neighbors)
        edge_labels.extend([0] * len(neighbors))
        row_offsets.append(len(column_indices))

        # etichette alternate 0/1, giusto per avere qualcosa di semplice
        node_labels.append(i % 2)

    return make_csr_graph(
        row_offsets=row_offsets,
        column_indices=column_indices,
        node_labels=node_labels,
        edge_labels=edge_labels,
        num_nodes=num_nodes,
    )