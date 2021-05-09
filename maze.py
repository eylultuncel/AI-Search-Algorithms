import datetime
import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math
from collections import deque


def calculate_neighbors(vertex, n):
    neighbors = []
    if vertex % n != 0:
        neighbors.append(vertex - 1)
    if vertex % n != n - 1:
        neighbors.append(vertex + 1)
    if vertex >= n:
        neighbors.append(vertex - n)
    if vertex < (n * (n - 1)):
        neighbors.append(vertex + n)
    return neighbors


def create_empty_maze(maze, n):
    for i in range(0, n * n):
        neighbors = calculate_neighbors(i, n)
        maze.add_node(i, visited=False, neighbors=neighbors)
    return maze


def get_random_unvisited_neighbor(vertex, maze):
    neighbors = maze.nodes[vertex]["neighbors"]
    if len(neighbors) == 0:
        return -1
    random_index = random.randint(0, len(neighbors) - 1)  # get some random element from neighbors
    next_vertex = neighbors[random_index]
    if maze.nodes[next_vertex]["visited"]:
        maze.nodes[vertex]["neighbors"].remove(next_vertex)
        next_vertex = get_random_unvisited_neighbor(vertex, maze)
    return next_vertex


def random_neighbors_to_stack(vertex, stack, maze, nodes):
    neighbors_list = maze.nodes[vertex]["neighbors"]
    for i in range(0, len(neighbors_list)):
        random_neighbor = get_random_unvisited_neighbor(vertex, maze)
        if random_neighbor != -1:
            nodes.add_node(random_neighbor, where_from=vertex)
            if random_neighbor in stack:
                stack.remove(random_neighbor)
            stack.append(random_neighbor)
            maze.nodes[vertex]["neighbors"].remove(random_neighbor)
    return stack


def create_maze(maze, n):
    nodes = nx.Graph()
    vertex = 0
    maze.nodes[vertex]["visited"] = True
    stack = []
    stack = random_neighbors_to_stack(vertex, stack, maze, nodes)
    while len(stack) != 0:
        next_vertex = stack.pop()
        maze.add_edge(next_vertex, nodes.nodes[next_vertex]["where_from"])
        maze.nodes[next_vertex]["visited"] = True
        vertex = next_vertex
        stack = random_neighbors_to_stack(vertex, stack, maze, nodes)
    return maze


def draw_maze(maze, n, path):
    plt.figure(figsize=(n / 2, n / 2))
    plt.xticks([]), plt.yticks([])
    for i in range(0, n + 1):
        plt.plot([0, n], [-i, -i], color='black')
        plt.plot([i, i], [0, -n], color='black')
    plt.plot([0, 1], [0, 0], color='white')
    plt.plot([n - 1, n], [-n, -n], color='white')

    for i in range(0, n * n):
        if maze.has_edge(i, i - 1):  # yatayda
            plt.plot([i % n, i % n], [-(i // n), -((i // n) + 1)], color='white')
        if maze.has_edge(i, i - n):  # dikeyde
            plt.plot([(i % n), (i % n) + 1], [-(i // n), -(i // n)], color='white')

    # plt.show()
    filename = "maze2_empty_" + str(n) + ".png"
    plt.savefig(filename)


def draw_maze_path(maze, n, path):
    plt.figure(figsize=(n / 2, n / 2))
    plt.xticks([]), plt.yticks([])

    for i in range(0, n + 1):
        plt.plot([0, n], [-i, -i], color='black')
        plt.plot([i, i], [0, -n], color='black')

    plt.plot([0, 1], [0, 0], color='white')
    plt.plot([n - 1, n], [-n, -n], color='white')

    for i in range(0, n * n):

        if maze.has_edge(i, i - 1):  # yatayda
            plt.plot([i % n, i % n], [-(i // n), -((i // n) + 1)], color='white')

        if maze.has_edge(i, i - n):  # dikeyde
            plt.plot([(i % n), (i % n) + 1], [-(i // n), -(i // n)], color='white')

    plt.plot([0.5, 0.5], [0, -0.5], linewidth=2.5, color='red')
    plt.plot([n - 0.5, n - 0.5], [-n + 0.5, -n], linewidth=2.5, color='red')

    print("\nPATH = ")
    print(path)
    x = (n * n) - 1
    for i in range(0, len(path)):
        a = path.get(x)
        if a is None:
            continue
        if a == x - 1:
            plt.plot([x % n - 0.5, x % n + 0.5], [-((x // n) + 0.5), -((x // n) + 0.5)], linewidth=2.5, color='red')

        elif a == x + 1:
            plt.plot([x % n + 0.5, x % n + 1.5], [-((x // n) + 0.5), -((x // n) + 0.5)], linewidth=2.5, color='red')

        if a == x - n:
            plt.plot([(x % n) + 0.5, (x % n) + 0.5], [-(x // n + 0.5), -(x // n - 0.5)], linewidth=2.5, color='red')

        elif a == x + n:
            plt.plot([(x % n) + 0.5, (x % n) + 0.5], [-(x // n + 0.5), -(x // n + 1.5)], linewidth=2.5, color='red')

        x = a

    # plt.show()
    filename = "maze2_" + str(n) + ".png"
    plt.savefig(filename)


def depth_limited_search(maze, start, goal, limit=-1):
    found = False
    fringe = deque([(0, start)])
    visited = {start}
    came_from = {start: None}

    while not found and len(fringe):
        current = fringe.pop()
        depth = current[0]
        current = current[1]

        if current == goal:
            found = True
            break

        if limit == -1 or depth < limit:
            for node in maze.neighbors(current):
                if node not in visited:
                    visited.add(node)
                    fringe.append((depth + 1, node))
                    came_from[node] = current
    if found:
        return came_from, visited
    else:
        return None, visited


def iterative_deepening_dfs(maze, start, goal):
    prev_iter_visited = []
    depth = 0
    count_expanded = 0
    while True:
        traced_path, visited = depth_limited_search(maze, start, goal, depth)
        if traced_path or len(visited) == len(prev_iter_visited):
            return count_expanded, len(traced_path)
        else:
            count_expanded += len(visited)
            prev_iter_visited = visited
            depth += 1


def uniform_cost_search(maze, start, goal):
    found = False
    fringe = [(0, start)]
    visited = {start}
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not found and len(fringe):
        current = heapq.heappop(fringe)
        current = current[1]

        if current == goal:
            found = True
            break

        for node in maze.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if node not in visited or cost_so_far[node] > new_cost:
                visited.add(node)
                came_from[node] = current
                cost_so_far[node] = new_cost
                heapq.heappush(fringe, (new_cost, node))

    if found:
        return len(visited), cost_so_far[goal], came_from
    else:
        print('No path from {} to {}'.format(start, goal))
        return None, math.inf


def heuristics(maze, n):
    heuristics_euclidean = []
    heuristics_manhattan = []
    for i in range(0, n * n):
        x = (n - 1) - (i % n)
        y = (n - 1) - (i // n)
        heuristics_euclidean.append((math.sqrt((x * x) + (y * y))))
        heuristics_manhattan.append(x + y)
    return heuristics_euclidean, heuristics_manhattan


def a_star_search(maze, start, goal, heuristic):
    found = False
    fringe = [(heuristic[start], start)]
    visited = {start}
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not found and len(fringe):
        current = heapq.heappop(fringe)
        current_heuristic = current[0]
        current = current[1]

        if current == goal:
            found = True
            break

        for node in maze.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if node not in visited or cost_so_far[node] > new_cost:
                visited.add(node)
                came_from[node] = current
                cost_so_far[node] = new_cost
                heapq.heappush(fringe, (new_cost + heuristic[node], node))  # SORUN OLUR

    if found:
        return len(visited), cost_so_far[goal]

    else:
        print('No path from {} to {}'.format(start, goal))
        return None, math.inf


def run_algorithms(maze, n):
    print()
    print("UCS")
    begin = datetime.datetime.now()
    ucs_expanded, ucs_path_length, came_from = uniform_cost_search(maze, 0, (n * n) - 1)
    end = datetime.datetime.now()
    delta = end - begin
    print("Path Length = ", ucs_path_length, "\nExpanded nodes = ", ucs_expanded, "\nTime passed = ",
          delta.total_seconds() * 1000)

    h_euclidean, h_manhattan = heuristics(maze, n)

    print()
    print("A*- euc")
    begin = datetime.datetime.now()
    a_star_euc_expanded, a_star_euc_path_length = a_star_search(maze, 0, (n * n) - 1, h_euclidean)
    end = datetime.datetime.now()
    delta = end - begin
    print("Path Length = ", a_star_euc_path_length, "\nExpanded nodes = ", a_star_euc_expanded, "\nTime passed = ",
          delta.total_seconds() * 1000)

    print()
    print("A*- man")
    begin = datetime.datetime.now()
    a_star_man_expanded, a_star_man_path_length = a_star_search(maze, 0, (n * n) - 1, h_manhattan)
    end = datetime.datetime.now()
    delta = end - begin
    print("Path Length = ", a_star_man_path_length, "\nExpanded nodes = ", a_star_man_expanded, "\nTime passed = ",
          delta.total_seconds() * 1000)

    print()
    print("IDS")
    begin = datetime.datetime.now()
    ids_expanded, ids_visited = iterative_deepening_dfs(maze, 0, (n * n) - 1)
    end = datetime.datetime.now()
    delta = end - begin
    print("Expanded nodes = ", ids_expanded, "\nTime passed = ", delta.total_seconds() * 1000)

    return came_from


def main():
    maze = nx.Graph()
    n = 10
    maze = create_empty_maze(maze, n)
    maze = create_maze(maze, n)

    path = run_algorithms(maze, n)

    # nx.draw(maze, with_labels=True)
    # plt.show()

    draw_maze(maze, n, path)
    draw_maze_path(maze, n, path)


if __name__ == "__main__":
    main()
