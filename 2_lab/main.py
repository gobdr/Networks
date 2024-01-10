from enum import Enum
from math import inf
from typing import List

from matplotlib import pyplot as pltё, pyplot as plt


def img_save_dst() -> str:
    return 'report/img/'




class Topology(Enum):
    kLine = 0,
    kRing = 1,
    kStar = 2


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def dist(self, other) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Plotter:
    def __init__(self) -> None:
        pass

    def plot_points(self, points: List[Point], show: bool = True, title: str = '') -> None:
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        plt.plot(xs, ys, 'o')

        for i, p in enumerate(points):
            plt.text(p.x, p.y + 0.05, f'{i}')

        if show:
            plt.savefig(f'{img_save_dst()}{title}.png', dpi=200)
            plt.clf()

    def plot_network_grapth(self, network, title: str = '') -> None:
        for i, neightbours in enumerate(network.nodes_graph):
            cur_point = network.nodes[i]

            for node_idx in neightbours:
                neightbour_point = network.nodes[node_idx]

                plt.plot((cur_point.x, neightbour_point.x),  (cur_point.y, neightbour_point.y), 'b')

        self.plot_points(network.nodes, False)
        plt.savefig(f'{img_save_dst()}{title}.png', dpi=200)
        plt.clf()

def print_paths(start_node: int, paths: List[List[int]], file=None) -> None:
    for i, path in enumerate(paths):
        if len(path) > 0:
            assert path[0] == start_node

        if file is None:
            print(f'path {start_node} -> {i}: {path}')
        else:
            file.write(f'path {start_node} -> {i}: {path}\n')


class Network:
    @staticmethod
    def create_network(topology: Topology) :
        network = None

        if topology == Topology.kStar:
            nodes = [
                Point(0.0, 0.0),
                Point(1.0, 0.0), Point(0.0, 1.0), Point(-1.0, 0.0), Point(0.0, -1.0),
                Point(1.0, 1.0), Point(1.0, -1.0), Point(-1.0, 1.0), Point(-1.0, -1.0)
            ]
            network = Network(nodes=nodes, connection_radius=0.0)
            network.nodes_graph = [[0] for i, n in enumerate(nodes) if i > 0]
            network.nodes_graph = [[i for i, n in enumerate(nodes) if i > 0]] + network.nodes_graph

        return network

    def __init__(self, nodes: List[Point], connection_radius: float) -> None:
        self.nodes = nodes
        self.radius = connection_radius
        self.nodes_graph: List[List[int]] = None

    def remove_node(self, node_idx) -> None:
        self.nodes[node_idx] = Point(inf, inf)
        if self.nodes_graph is not None:
            self.build_graph()

    def build_graph(self) -> None:
        if self.nodes_graph is not None:
            return

        self.nodes_graph = [[i for i, n in enumerate(self.nodes) if node.dist(n) < self.radius] for node in self.nodes]

    def ospf(self, title: str) -> None:
        with open(f'results/{title}.txt', 'w') as f:
            for i in range(len(self.nodes)):
                f.write(f'Start node {i}:\n')
                paths = self.network_dijkstra(i)
                print_paths(i, paths, f)
                f.write(f'###################################\n')

    def network_dijkstra(self, start_node_idx: int) -> List[List[int]]:
        assert 0 <= start_node_idx < len(self.nodes)

        distances = [inf for _ in range(len(self.nodes))]
        distances[start_node_idx] = 0
        used = [False for _ in range(len(self.nodes))]
        paths = [[] for _ in range(len(self.nodes))]

        class Node:
            def __init__(self, idx: int, dist: float) -> None:
                self.vert_idx = idx
                self.dist = dist

        vertex_heap = [Node(start_node_idx, distances[start_node_idx])]

        while len(vertex_heap) > 0:
            cur_min_node = Node(-1, inf)
            cur_min_idx = -1
            for i, node in enumerate(vertex_heap):
                if node.dist < cur_min_node.dist:
                    cur_min_node = node
                    cur_min_idx = i

            del vertex_heap[cur_min_idx]
            if used[cur_min_node.vert_idx]:
                continue

            used[cur_min_node.vert_idx] = True

            for neightbour in self.nodes_graph[cur_min_node.vert_idx]:
                new_dist = distances[cur_min_node.vert_idx] + self.nodes[neightbour].dist(self.nodes[cur_min_node.vert_idx])
                if new_dist < distances[neightbour]:
                    distances[neightbour] = new_dist
                    vertex_heap.append(Node(neightbour, new_dist))
                    paths[neightbour] = paths[cur_min_node.vert_idx] + [cur_min_node.vert_idx]

        for i, path in enumerate(paths):
            if distances[i] < inf:
                path.append(i)

        return paths


def main():
    plt = Plotter()
    line_topology_network = Network(
        nodes=[Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0), Point(3.0, 3.0), Point(4.0, 4.0), Point(5.0, 5.0)],
        connection_radius=1.5
    )

    line_topology_network.build_graph()
    plt.plot_points(line_topology_network.nodes, True, 'full_line_points')
    plt.plot_network_grapth(line_topology_network, 'full_line')

    line_topology_network.ospf('line_full')

    line_topology_network.remove_node(3)
    plt.plot_network_grapth(line_topology_network, 'rm_line')
    line_topology_network.ospf('line_remove')

    def ring_points(r: float) -> List[Point]:
        xs = [-3.0, -2.7, -2.0, -1.0]
        xs = xs + [0.0] + [-x_k for x_k in xs]
        ys = []
        for x_k in xs:
            y_abs = (r * r - x_k * x_k) ** 0.5
            ys.extend([y_abs, -y_abs])


        points = []

        for i, x_k in enumerate(xs):
            if ys[2 * i] == 0:
                points.extend([Point(x_k, ys[2 * i])])
            else:
                points.extend([Point(x_k, ys[2 * i]), Point(x_k, ys[2 * i + 1])])
        
        return points

    ring_topology_network = Network(
        nodes=ring_points(3.0),
        connection_radius=1.7
    )

    ring_topology_network.build_graph()
    ring_topology_network.ospf('ring_full')
    plt.plot_points(ring_topology_network.nodes, True, 'full_ring_points')
    plt.plot_network_grapth(ring_topology_network, 'full_ring')

    ring_topology_network.remove_node(11)
    ring_topology_network.ospf('ring_remove')
    plt.plot_network_grapth(ring_topology_network, 'rm_ring')

    star_topology_nerwork = Network.create_network(Topology.kStar)
    plt.plot_points(star_topology_nerwork.nodes, True, 'full_star_points')
    plt.plot_network_grapth(star_topology_nerwork, 'full_star')
    star_topology_nerwork.ospf('star_full')

    star_topology_nerwork.remove_node(0)
    plt.plot_network_grapth(star_topology_nerwork, 'rm_star')
    star_topology_nerwork.ospf('star_remove')


if __name__ == '__main__':
    main()
