import networkx as nx
import numpy as np

def set_weights_to_one(G):
    """
    In the beginning, set all weights of the edges in the graph to one.
    """
    for u, v in G.edges():
        G[u][v]['weight'] = 1


def set_group_order(G):
    """
    Set the order of the groups because the algorithm needs to start with the larger group (here called group red).
    G needs to be a bipartite graph with the attribute node_type with values 'top' and 'bottom' for the two node groups.
    """
    top_nodes = {n for n, d in G.nodes(data=True) if d["node_type"] == "top"}
    bottom_nodes = set(G) - top_nodes

    # find the longer set of nodes to start with
    # the longer group will be called group red
    if len(top_nodes) > len(bottom_nodes):
        node_group_red = top_nodes
        node_group_blue = bottom_nodes
    else:
        node_group_red = bottom_nodes
        node_group_blue = top_nodes

    groups = {'red': node_group_red, 'blue': node_group_blue}
    return groups


def find_neighbor_communities(G, node, community_assignments):
    """
    Given a node, find the neighboring communities in graph G (depending on the current community_assignments).
    """
    neighboring_communities = set()

    for community in community_assignments.values():  # check if communities are neighboring to our current node

        # if the node we want to check is in the community already, it must be a neighboring community (except for first iteration, when the node itself is a community - then it is excluded again when checking the length)
        # if the node we want to check is in the community, we need to remove it from that community to check the modularity gain
        if node in community:
            community_copy = set(community)
            community_copy.remove(node)
            if len(community_copy) != 0:  # make sure that it is not an empty community after removing the node
                neighboring_communities.add(tuple(community_copy))  # Add the modified community

        # find the communities that include neighbors of the current node
        else:
            for node_b in community:
                if node_b != node and G.has_edge(node_b, node):
                    neighboring_communities.add(tuple(community))

    return neighboring_communities


def bipartite_modularity_gain(G, node_i, community, group):
    """
    Compute the modularity gain of moving node_i to community. Group is the group of node_i.
    """
    comm_other = [node for node in community if
                  node not in group]  # only take into account nodes in community from other node group
    m = sum([data['weight'] for _, _, data in G.edges(data=True)])  # Total sum of edge weights

    edges = G.edges(node_i, data=True)  # find edges incident to node_i
    edges_in_community = [(u, v, data['weight']) for u, v, data in edges if
                          v in comm_other]  # edges and edge weights of edges between node_i and community
    k_i_in = sum(weight for _, _, weight in edges_in_community)  # sum of edge weights in community

    comm_other_edges = [(u, v, data['weight']) for u, v, data in
                        G.edges(comm_other, data=True)]  # edges and edge weights of community nodes in other group
    community_degree = sum(weight for _, _, weight in
                           comm_other_edges)  # sum of degrees considering weight of nodes of other group in community

    # compute modularity gain
    delta_q = k_i_in / m - G.degree(node_i, weight='weight') * community_degree / m ** 2

    return delta_q


def move_node(node, add_community, community_assignments):
    """
    Move a node to a new community and update the community assignments.
    """
    add_community_key, remove_community_key = None, None

    for i, community in community_assignments.items():  # iterate over communities
        if add_community.issubset(community):  # find add community index where node has to be added
            add_community_key = i
        elif node in community:  # find index of community node has to be removed from
            remove_community_key = i

    if add_community_key is not None:
        community_assignments[add_community_key].add(node)  # add node to new community

    if remove_community_key is not None:
        community_assignments[remove_community_key].remove(node)  # remove node from old community
        if len(community_assignments[
                   remove_community_key]) == 0:  # if the community the node was removed from is empty -> delete community
            del community_assignments[remove_community_key]

    return community_assignments


def assignment(G, group, community_assignments):
    """
    Assignment step: assign nodes of current node group to communities.
    """
    new_assignments = community_assignments.copy()

    for node in group:  # go over all nodes of the current node group

        # find all neighboring communities
        neighboring_communities = find_neighbor_communities(G, node, new_assignments)

        max_delta_Q = 0

        # calculate modularity gain for all neighboring communities and keep only largest one
        for community in neighboring_communities:
            delta_Q = bipartite_modularity_gain(G, node, community, group)
            if delta_Q > max_delta_Q:
                max_delta_Q = delta_Q
                max_community = set(community)

        # update community of that node
        if max_delta_Q > 0:  # only update if positive modularity gain (if max_delta_Q is still 0, there was no positive modularity gain found)
            new_assignments = move_node(node, max_community, new_assignments)

    # If there were changes in the community assignments, we do the same again with the new community assignments
    if new_assignments != community_assignments:
        assignment(G, group, new_assignments)

    # check if isolated nodes of the group assist, if yes, assign them randomly
    for node in group:
        for community_key, community in new_assignments.items():
            if node in community:
                if len(community) == 1:
                    neighboring_communities = find_neighbor_communities(G, node, new_assignments)
                    random_community = list(neighboring_communities)[0]
                    new_assignments = move_node(node, random_community, new_assignments)

    return new_assignments  # return the communities when there are no changes anymore


def aggregation(group, group_key, community_assignments, G, iteration):
    """
    Aggregation step: aggregate nodes of current node group to hypernodes.
    """
    H = nx.Graph()  # create new hypergraph
    hypernodes = dict()  # create dict for storing hypernodes
    new_assignments = dict()

    # define color of hypernode for hypernode id
    if group_key == 'red':
        x = 'r'
    else:
        x = 'b'

    # first we add all necessary nodes to the graph
    for community_id, nodes in community_assignments.items():
        new_assignments[community_id] = set()

        agg_nodes = []  # find nodes that need to be aggregated to hypernode
        for node in nodes:
            if node in group:  # nodes from current group are aggregated
                agg_nodes.append(node)
            else:  # nodes from other group are not aggregated
                H.add_node(node)  # nodes from other group are not changed
                new_assignments[community_id].add(node)

        if len(agg_nodes) > 0:  # create hypernode if there are nodes to be aggregated
            hypernode_id = f'{x}_{iteration}_{community_id}'  # define name of hypernode
            H.add_node(hypernode_id)  # add hypernode to hypergraph
            hypernodes[hypernode_id] = agg_nodes  # store original nodes of hypernode
            new_assignments[community_id].add(hypernode_id)

    # then we add all the edges
    for hypernode_id, nodes in hypernodes.items():
        for node in nodes:  # find edges of hypernode
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if not H.has_edge(hypernode_id, neighbor):  # add edge if it does not exist yet
                    edge_weight = G[node][neighbor].get('weight')
                    H.add_edge(hypernode_id, neighbor,
                               weight=edge_weight)  # add edge with weight it had in Graph before
                else:
                    edge_weight = G[node][neighbor].get('weight')
                    current_weight = H[hypernode_id][neighbor].get(
                        'weight')  # if edge already exists, add 1 to weight for additional edge in original graph
                    new_weight = current_weight + edge_weight
                    H[hypernode_id][neighbor]['weight'] = new_weight  # update weight of edge

    return H, hypernodes, new_assignments


def add_rows_and_columns(matrix, i, j):
    matrix[i] = matrix[i] + matrix[j]
    matrix = np.delete(matrix, j, axis=0)
    matrix[:, i] = matrix[:, i] + matrix[:, j]
    matrix = np.delete(matrix, j, axis=1)
    return matrix


def find_max_indices_and_update(S):
    max_indices = np.unravel_index(np.argmax(S), S.shape)
    if S[max_indices] > 0:
        S_new = add_rows_and_columns(S, max_indices[0], max_indices[1])
        return max_indices, S_new
    else:
        return None, None


def combine_communities(community_order, max_indices, community_assignments):
    comm_1 = community_order[max_indices[0]]
    comm_2 = community_order[max_indices[1]]

    community_assignments[comm_1] = community_assignments[comm_1] | community_assignments[comm_2]
    community_assignments.pop(comm_2)

    community_order.remove(comm_2)

    return community_assignments, community_order


def recursive_update(G, m, community_assignments, S, community_order):
    max_indices, S_new = find_max_indices_and_update(S)

    if max_indices is not None and S_new is not None:
        community_assignments_new, community_order_new = combine_communities(community_order, max_indices,
                                                                             community_assignments)
        return recursive_update(G, m, community_assignments_new, S_new, community_order_new)

    else:
        return community_assignments


def clustering(G, m, community_assignments):
    community_order = sorted([community_id for community_id in community_assignments.keys()])
    sorted_nodes_blue = [node for community_id in community_order for node in community_assignments[community_id] if
                         node.startswith('b')]
    sorted_nodes_red = [node for community_id in community_order for node in community_assignments[community_id] if
                        node.startswith('r')]
    sorted_nodes = sorted_nodes_blue + sorted_nodes_red
    # n = len(sorted_nodes)
    comm_len = len(community_order)

    adj_matrix = nx.adjacency_matrix(G, nodelist=sorted_nodes)
    adj_matrix = adj_matrix.toarray()

    R = adj_matrix[comm_len:, :comm_len]
    e = np.ones((comm_len, 1))
    d = np.dot(R, e)
    g = np.dot(R.T, e)

    S = 1 / m * (R + R.T - (np.dot(d, g.T) + np.dot(g, d.T)) / m)
    np.fill_diagonal(S, -np.inf)

    return recursive_update(G, m, community_assignments, S, community_order)

def unpack_hypernodes(node, all_hypernodes):
    """
    Unpack hypernode recursively to find original nodes which were aggregated to hypernode.
    """
    if node in all_hypernodes:
        return [unpack_hypernodes(sub_node, all_hypernodes) for sub_node in all_hypernodes[node]]
    else:
        return [node]


def flatten(nested_list):
    """
    Flatten nested list which was created by unpack_hypernodes function.
    """
    if len(nested_list) == 0:
        return nested_list
    if isinstance(nested_list[0], list):
        return flatten(nested_list[0]) + flatten(nested_list[1:])
    return nested_list[:1] + flatten(nested_list[1:])


def final_communities(community_assignments, all_hypernodes):
    """
    Find communities of original nodes by unpacking all hypernodes.
    """
    final_community_ass = dict()

    # go through communities
    for community_id, nodes in community_assignments.items():
        final_community_ass[community_id] = []

        # go through nodes in communities and unpack them
        for node in nodes:
            final_community_ass[community_id].extend(unpack_hypernodes(node, all_hypernodes))

        # flatten found final communities
        final_community_ass[community_id] = flatten(final_community_ass[community_id])

    return final_community_ass


def bilouvain(Graph):
    """
    Find communities in a bipartite graph according to bilouvain algorithm by Zhou et al. (2018).
    G has to be a bipartite graph with node type as an attribute.
    """

    G = Graph.copy()
    m = G.number_of_edges()
    set_weights_to_one(G)  # set all weights in graph to one to begin with (make sure they have attribute weight)
    groups = set_group_order(G)  # set which group of nodes comes first (longer one)

    # initialize communities
    community_assignments = {i: {n} for i, n in enumerate(G)}

    all_hypernodes = dict()  # initialize dict where the contents of all hypernodes will be stored

    # create variables for stop condition
    no_change_red = False
    no_change_blue = False

    i = 0  # count iterations

    while not no_change_red or not no_change_blue:  # if there are no more changes in both community assignment steps, the loop stops

        i += 1  # increase iteration by one (for naming hypernodes)
        print(f'iteration: {i}')

        # assignment on red nodes
        community_assignments_before = community_assignments.copy()
        community_assignments = assignment(G, groups['red'], community_assignments)
        if community_assignments == community_assignments_before:
            no_change_red = True

        # aggregation on red nodes
        G, hypernodes, community_assignments = aggregation(groups['red'], 'red', community_assignments, G, iteration=i)
        groups['red'] = list(hypernodes.keys())
        all_hypernodes.update(hypernodes)

        # assignment on blue nodes
        community_assignments_before = community_assignments.copy()
        community_assignments = assignment(G, groups['blue'], community_assignments)
        if community_assignments == community_assignments_before:
            no_change_blue = True

        # aggregation on blue nodes
        G, hypernodes, community_assignments = aggregation(groups['blue'], 'blue', community_assignments, G,
                                                           iteration=i)
        groups['blue'] = list(hypernodes.keys())
        all_hypernodes.update(hypernodes)

    # clustering step on balanced communities
    community_assignments = clustering(G, m, community_assignments)

    # find final communities with original nodes
    final_community_assignments = final_communities(community_assignments, all_hypernodes)

    return final_community_assignments