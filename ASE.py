from networkx.algorithms.connectivity import build_auxiliary_node_connectivity
from utils import get_file_names, read_input_file, get_vertex_disjoint_paths
from networkx.algorithms.flow import build_residual_network
import networkx as nx
import random
import time
import sys
import os

untouchable_nodes = set()
vertices = set()


def reset_query_dict(q_dict):
    vertices.clear()
    untouchable_nodes.clear()
    for k in q_dict:
        # Create disjoint vertices set based on input pair
        vertices.add(k[0])
        vertices.add(k[1])
        q_dict[k] = []


def is_safe_to_add(path):
    count = 0
    for v in path:
        # If a node conflicts with already chosen paths
        if v in untouchable_nodes:
            return False
        elif v in vertices:
            count += 1
    # If a node is part of the input nodes then don't choose the path
    if count > 2:
        return False
    return True


def add_to_nodes_used(path):
    for v in path:
        untouchable_nodes.add(v)


def remove_from_nodes_used(path):
    for v in path:
        untouchable_nodes.remove(v)


def path_sorter(graph, initial_paths):
    in_degree_map = {}
    for i in range(len(initial_paths)):
        path = initial_paths[i]
        deg_val = 0
        # For all nodes between source and sink
        for v in path[1:-1]:
            deg_val += graph.in_degree[v]
        # deg_val is the ratio of sum of incoming edges of intermediate nodes
        # to the total number of nodes in the path
        deg_val /= len(path)
        if deg_val in in_degree_map:
            existing_paths = in_degree_map[deg_val]
            existing_paths.append(i)
            in_degree_map[deg_val] = existing_paths
        else:
            in_degree_map[deg_val] = [i]
    map_keys = list(in_degree_map.keys())
    map_keys.sort()
    return map_keys, in_degree_map


def backpropagation(start_time, graph, query_dict_keys, path_dict, query_dict, idx):
    # If it takes more than an hour then return
    #if time.perf_counter()-start_time > 3600:
    if time.perf_counter() - start_time > 300:
        return False, None
    if idx >= 10 or query_dict[query_dict_keys[idx]]:
        return True, query_dict
    pair = query_dict_keys[idx]
    paths = path_dict[pair]
    path_select_keys, in_degree_map = path_sorter(graph, paths)
    for i in range(len(path_select_keys)):
        possible_paths = in_degree_map[path_select_keys[i]]
        for j in range(len(possible_paths)):
            path = paths[possible_paths[j]]
            if is_safe_to_add(path):
                add_to_nodes_used(path)
                query_dict[pair] = path
                res, temp_query_dict = backpropagation(start_time, graph, query_dict_keys,
                                                       path_dict, query_dict, idx+1)
                if res:
                    return True, temp_query_dict
                elif not temp_query_dict:
                    return False, None
                remove_from_nodes_used(path)
                query_dict[pair] = []
    return False, query_dict


def path_count(graph, graph_aux, graph_residual):
    len_dict = {}
    paths_dict = {}
    for key in query_dict:
        source = key[0]
        destination = key[1]
        # Find mutually disjoint paths between a single source and sink
        paths = get_vertex_disjoint_paths(graph, source, destination, graph_aux, graph_residual)
        len_dict[key] = len(paths)
        paths_dict[key] = paths
    return list(dict(sorted(len_dict.items(), key=lambda item: item[1])).keys()), paths_dict


def find_path_dfs(graph, u, d, p, visited):
    visited[u] = True
    p.append(u)
    if u == d:
        return p
    adj_list = list(graph.adj[u])
    random.shuffle(adj_list)
    for n in adj_list:
        if not visited[n] and (n == d or n not in vertices):
            rec_path = find_path_dfs(graph, n, d, p, visited)
            if rec_path:
                return rec_path
    p.pop()
    return None


def next_pair_to_explore(query_dict):
    keys = list(query_dict.keys())
    random.shuffle(keys)
    for key in keys:
        if not query_dict[key]:
            return key
    return None


def random_path_finder(graph, query_dict, out_file):
    for j in range(100000):
        new_graph = graph.copy()
        reset_query_dict(query_dict)
        key = next_pair_to_explore(query_dict)
        i = 0
        while key and i < 100000:
            source = key[0]
            destination = key[1]
            edge_dict = {}
            vertices_to_remove = []
            for node in vertices:
                if node == destination or node == source:
                    continue
                vertices_to_remove.append(node)
                edge_dict[node] = []
                edge_dict[node].extend(list(new_graph.in_edges(node)))
                edge_dict[node].extend(list(new_graph.out_edges(node)))
            new_graph.remove_nodes_from(vertices_to_remove)
            try:
                path = find_path_dfs(new_graph, source, destination, [], [False] * 101)
                if path:
                    query_dict[key] = path
                    new_graph.remove_nodes_from(path)
                    vertices.remove(source)
                    vertices.remove(destination)
            except nx.NetworkXNoPath as e:
                print('Couldnt find path between ' + str(source) + ' and ' + str(destination))
            finally:
                new_graph.add_nodes_from(vertices_to_remove)
                for node in edge_dict:
                    new_graph.add_edges_from(edge_dict[node])
            key = next_pair_to_explore(query_dict)
            i += 1
        if not os.path.exists(out_file):
            with open(out_file, 'w') as file:
                for x in query_dict:
                    path = query_dict[x]
                    if len(path) > 0:
                        file.write(" ".join(repr(v) for v in path) + '\n')
                file.close()
        else:
            with open(out_file, 'r') as file:
                line_count = 0
                for line in file:
                    if line != "\n":
                        line_count += 1
                file.close()
            if count > line_count:
                with open(out_file, 'w') as file:
                    for x in query_dict:
                        path = query_dict[x]
                        if len(path) > 0:
                            file.write(" ".join(repr(v) for v in path) + '\n')
                    file.close()


def simple_path_count(graph, start_time):
    len_dict = {}
    paths_dict = {}
    for key in query_dict:
        source = key[0]
        destination = key[1]
        # Find simple paths with at most 10 edges
        if time.perf_counter() - start_time > 300:
            return None, None
        paths = list(nx.all_simple_paths(graph, source, destination, cutoff=10))
        len_dict[key] = len(paths)
        paths_dict[key] = paths
    return list(dict(sorted(len_dict.items(), key=lambda item: item[1])).keys()), paths_dict


if __name__ == '__main__':
    inp_file, out_file = get_file_names(sys.argv[1:])
    start_time = time.perf_counter()
    print('Starting time: ' + str(start_time))
    graph, query_dict = read_input_file(inp_file)
    graph_aux = build_auxiliary_node_connectivity(graph)
    graph_residual = build_residual_network(graph_aux, "capacity")
    reset_query_dict(query_dict)
    query_dict_keys, path_dict = path_count(graph, graph_aux, graph_residual)
    result, result_query_dict = backpropagation(start_time, graph, query_dict_keys, path_dict,
                                                query_dict, 0)
    if not result:
        print('Initial approach failed, trying to get simple paths')
        reset_query_dict(query_dict)
        query_dict_keys, path_dict = simple_path_count(graph, start_time)
        if not query_dict_keys or  not path_dict:
            print('Simple path approach failed, trying to get random paths')
            random_path_finder(graph, query_dict, out_file)
        else:
            result, result_query_dict = backpropagation(start_time, graph, query_dict_keys, path_dict,
                                                        query_dict, 0)
            if not result:
                print('Simple path approach failed, trying to get random paths')
                random_path_finder(graph, query_dict, out_file)
            else:
                count = 0
                for key in result_query_dict:
                    if result_query_dict[key]:
                        count += 1
                if count != 10:
                    print('Simple path approach didnt find 10 paths, trying to get random paths')
                    random_path_finder(graph, query_dict, out_file)
                else:
                    with open(out_file, 'w') as file:
                        for key in result_query_dict:
                            path = result_query_dict[key]
                            file.write(" ".join(repr(v) for v in path) + '\n')
                    file.close()
                    print('Found all paths using simple paths logic')
    else:
        count = 0
        for key in result_query_dict:
            if result_query_dict[key]:
                count += 1
        if count != 10:
            print('Initial approach didnt find 10 paths, trying to get simple paths')
            reset_query_dict(query_dict)
            query_dict_keys, path_dict = simple_path_count(graph, start_time)
            if not query_dict_keys or not path_dict:
                print('Simple path approach failed, trying to get random paths')
                random_path_finder(graph, query_dict, out_file)
            else:
                result, result_query_dict = backpropagation(start_time, graph, query_dict_keys, path_dict,
                                                            query_dict, 0)
                if not result:
                    print('Simple path approach failed, trying to get random paths')
                    random_path_finder(graph, query_dict, out_file)
                else:
                    count = 0
                    for key in result_query_dict:
                        if result_query_dict[key]:
                            count += 1
                    if count != 10:
                        print('Simple path approach didnt find 10 paths, trying to get random paths')
                        random_path_finder(graph, query_dict, out_file)
                    else:
                        with open(out_file, 'w') as file:
                            for key in result_query_dict:
                                path = result_query_dict[key]
                                file.write(" ".join(repr(v) for v in path) + '\n')
                        file.close()
                        print('Found all paths using simple paths logic')
        else:
            with open(out_file, 'w') as file:
                for key in result_query_dict:
                    path = result_query_dict[key]
                    file.write(" ".join(repr(v) for v in path) + '\n')
            file.close()
            print('Found all paths using node disjoint max flow')
    end_time = time.perf_counter()
    print('Ending time: ' + str(end_time))
    print('Time taken: ' + str(end_time - start_time) + ' seconds')