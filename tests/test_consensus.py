"""
Tests for the gravitational consensus protocol.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "edge"))

from gravitational_consensus import GravitationalConsensusNetwork, GravitationalNode


# --------------------------------------------------------------------------- #
# Node creation                                                                #
# --------------------------------------------------------------------------- #


class TestNodeCreation:
    def test_basic_creation(self):
        node = GravitationalNode("node_a", mass=2.0)
        assert node.node_id == "node_a"
        assert node.mass == 2.0
        assert node.is_online

    def test_default_position_assigned(self):
        node = GravitationalNode("n0")
        assert node.position is not None
        assert len(node.position) == 4

    def test_custom_position(self):
        pos = np.array([1.0, 0.0, 0.0, 0.0])
        node = GravitationalNode("n1", position=pos)
        np.testing.assert_array_equal(node.position, pos)

    def test_state_initially_empty(self):
        node = GravitationalNode("n2")
        assert node.state == {}

    def test_update_state(self):
        node = GravitationalNode("n3")
        node.update_state({"hash": "abc123", "version": 1})
        assert node.state["hash"] == "abc123"

    def test_online_offline_toggle(self):
        node = GravitationalNode("n4")
        assert node.is_online
        node.go_offline()
        assert not node.is_online
        node.go_online()
        assert node.is_online


# --------------------------------------------------------------------------- #
# Force computation                                                            #
# --------------------------------------------------------------------------- #


class TestForceComputation:
    def test_force_positive(self):
        a = GravitationalNode("a", mass=1.0, position=np.array([0.0, 0.0, 0.0, 0.0]))
        b = GravitationalNode("b", mass=1.0, position=np.array([1.0, 0.0, 0.0, 0.0]))
        force = a.compute_force(b)
        assert force > 0.0

    def test_force_increases_with_mass(self):
        pos_a = np.array([0.0, 0.0, 0.0, 0.0])
        pos_b = np.array([1.0, 0.0, 0.0, 0.0])
        a_light = GravitationalNode("a", mass=1.0, position=pos_a)
        a_heavy = GravitationalNode("a2", mass=10.0, position=pos_a)
        b = GravitationalNode("b", mass=1.0, position=pos_b)
        assert a_heavy.compute_force(b) > a_light.compute_force(b)

    def test_force_decreases_with_distance(self):
        a = GravitationalNode("a", mass=1.0, position=np.array([0.0, 0.0, 0.0, 0.0]))
        b_near = GravitationalNode("b1", mass=1.0, position=np.array([1.0, 0.0, 0.0, 0.0]))
        b_far = GravitationalNode("b2", mass=1.0, position=np.array([100.0, 0.0, 0.0, 0.0]))
        assert a.compute_force(b_near) > a.compute_force(b_far)

    def test_force_formula_exact(self):
        G = 6.674e-3
        a = GravitationalNode("a", mass=2.0, position=np.array([0.0, 0.0, 0.0, 0.0]))
        b = GravitationalNode("b", mass=3.0, position=np.array([1.0, 0.0, 0.0, 0.0]))
        force = a.compute_force(b, G=G)
        expected = G * 2.0 * 3.0 / (1.0 + 1e-8)
        assert abs(force - expected) < 1e-10

    def test_self_force_epsilon_stable(self):
        """Force when both nodes are at the same position should be finite."""
        pos = np.array([0.0, 0.0, 0.0, 0.0])
        a = GravitationalNode("a", mass=1.0, position=pos.copy())
        b = GravitationalNode("b", mass=1.0, position=pos.copy())
        force = a.compute_force(b)
        assert force < 1e6  # bounded by epsilon


# --------------------------------------------------------------------------- #
# Network add/remove                                                           #
# --------------------------------------------------------------------------- #


class TestNetworkAddRemove:
    def test_add_node(self):
        net = GravitationalConsensusNetwork()
        node = GravitationalNode("x", mass=1.0)
        net.add_node(node)
        topo = net.get_network_topology()
        ids = [n["node_id"] for n in topo["nodes"]]
        assert "x" in ids

    def test_remove_node(self):
        net = GravitationalConsensusNetwork()
        node = GravitationalNode("y", mass=1.0)
        net.add_node(node)
        net.remove_node("y")
        topo = net.get_network_topology()
        ids = [n["node_id"] for n in topo["nodes"]]
        assert "y" not in ids

    def test_remove_nonexistent_raises(self):
        net = GravitationalConsensusNetwork()
        with pytest.raises(KeyError):
            net.remove_node("does_not_exist")

    def test_node_count(self):
        net = GravitationalConsensusNetwork()
        for i in range(5):
            net.add_node(GravitationalNode(f"node_{i}"))
        topo = net.get_network_topology()
        assert len(topo["nodes"]) == 5


# --------------------------------------------------------------------------- #
# Consensus vote                                                               #
# --------------------------------------------------------------------------- #


class TestConsensusVote:
    def _three_node_net(self) -> GravitationalConsensusNetwork:
        net = GravitationalConsensusNetwork()
        for i in range(3):
            pos = np.array([float(i), 0.0, 0.0, 0.0])
            net.add_node(GravitationalNode(f"n{i}", mass=1.0, position=pos))
        return net

    def test_all_accept_returns_accepted(self):
        net = self._three_node_net()
        result = net.compute_consensus({"data": "test"})
        assert result["accepted"] is True
        assert result["confidence"] > 0.5

    def test_all_reject_returns_not_accepted(self):
        net = self._three_node_net()
        result = net.compute_consensus({"accept_predicate": lambda n: False})
        assert result["accepted"] is False
        assert result["confidence"] == pytest.approx(0.0, abs=1e-6)

    def test_majority_accept(self):
        net = GravitationalConsensusNetwork()
        # Place nodes at equal distances so forces are symmetric
        for i in range(3):
            pos = np.array([float(i), 0.0, 0.0, 0.0])
            net.add_node(GravitationalNode(f"n{i}", mass=1.0, position=pos))
        # 2 out of 3 accept
        votes_cast = [0]
        def predicate(node):
            votes_cast[0] += 1
            return node.node_id != "n2"
        result = net.compute_consensus({"accept_predicate": predicate})
        assert result["online_nodes"] == 3

    def test_empty_network_not_accepted(self):
        net = GravitationalConsensusNetwork()
        result = net.compute_consensus({})
        assert result["accepted"] is False
        assert result["online_nodes"] == 0

    def test_offline_node_excluded(self):
        net = self._three_node_net()
        net._nodes["n2"].go_offline()
        result = net.compute_consensus({})
        assert result["online_nodes"] == 2


# --------------------------------------------------------------------------- #
# Network topology                                                             #
# --------------------------------------------------------------------------- #


class TestNetworkTopology:
    def test_topology_structure(self):
        net = GravitationalConsensusNetwork()
        for i in range(3):
            net.add_node(GravitationalNode(f"t{i}", mass=float(i + 1)))
        topo = net.get_network_topology()
        assert "nodes" in topo
        assert "edges" in topo
        assert "total_mass" in topo
        assert "num_online" in topo

    def test_total_mass(self):
        net = GravitationalConsensusNetwork()
        net.add_node(GravitationalNode("a", mass=2.0))
        net.add_node(GravitationalNode("b", mass=3.0))
        topo = net.get_network_topology()
        assert topo["total_mass"] == pytest.approx(5.0)

    def test_edge_count(self):
        net = GravitationalConsensusNetwork()
        for i in range(4):
            net.add_node(GravitationalNode(f"e{i}"))
        topo = net.get_network_topology()
        # 4 nodes → 4*3/2 = 6 edges
        assert len(topo["edges"]) == 6

    def test_edge_forces_positive(self):
        net = GravitationalConsensusNetwork()
        for i in range(3):
            pos = np.array([float(i), 0.0, 0.0, 0.0])
            net.add_node(GravitationalNode(f"f{i}", mass=1.0, position=pos))
        topo = net.get_network_topology()
        for edge in topo["edges"]:
            assert edge["force"] > 0


# --------------------------------------------------------------------------- #
# Broadcast                                                                    #
# --------------------------------------------------------------------------- #


class TestBroadcast:
    def test_broadcast_reaches_all_nodes(self):
        net = GravitationalConsensusNetwork()
        for i in range(4):
            net.add_node(GravitationalNode(f"b{i}"))
        responses = net.broadcast({"type": "ping"})
        assert len(responses) == 4

    def test_broadcast_response_contains_node_id(self):
        net = GravitationalConsensusNetwork()
        net.add_node(GravitationalNode("ping_node"))
        responses = net.broadcast({"type": "ping"})
        assert responses[0]["node_id"] == "ping_node"

    def test_broadcast_offline_node_gets_error_response(self):
        net = GravitationalConsensusNetwork()
        node = GravitationalNode("offline_one")
        node.go_offline()
        net.add_node(node)
        responses = net.broadcast({"type": "ping"})
        assert responses[0]["node_id"] == "offline_one"
        assert "error" in responses[0]

    def test_broadcast_mixed_online_offline(self):
        net = GravitationalConsensusNetwork()
        net.add_node(GravitationalNode("online_node"))
        offline = GravitationalNode("offline_node")
        offline.go_offline()
        net.add_node(offline)
        responses = net.broadcast({})
        assert len(responses) == 2
        ok = [r for r in responses if "error" not in r]
        err = [r for r in responses if "error" in r]
        assert len(ok) == 1
        assert len(err) == 1
