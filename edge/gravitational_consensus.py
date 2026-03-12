"""
gravitational_consensus.py — Multi-agent gravitational consensus protocol.

Each participating agent is modelled as a *GravitationalNode* with a scalar
mass and a position in embedding space.  Gravitational force between nodes
determines voting weight: massive, nearby nodes exert more influence on
consensus.

The network is fully in-process but structured to be network-ready (each node
exposes a message interface that could be backed by a real transport).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_G_DEFAULT: float = 6.674e-3
_EPSILON: float = 1e-8


# --------------------------------------------------------------------------- #
# GravitationalNode                                                            #
# --------------------------------------------------------------------------- #


class GravitationalNode:
    """A single agent node in the gravitational consensus network.

    Parameters
    ----------
    node_id:
        Unique string identifier for this node.
    mass:
        Scalar mass ≥ 0 influencing voting weight.
    position:
        Optional position vector in embedding space.  A random 4-D unit
        vector is used when *None*.
    """

    def __init__(
        self,
        node_id: str,
        mass: float = 1.0,
        position: Optional[np.ndarray] = None,
    ) -> None:
        self.node_id = node_id
        self.mass = float(mass)
        self.position: np.ndarray = (
            position
            if position is not None
            else np.random.randn(4).astype(np.float32)
        )
        self.state: Dict[str, Any] = {}
        self._online: bool = True
        logger.debug("GravitationalNode created: id=%s, mass=%.3f", node_id, mass)

    # ------------------------------------------------------------------ #

    def compute_force(self, other: "GravitationalNode", G: float = _G_DEFAULT) -> float:
        """Compute gravitational force between this node and *other*.

        .. math::

            F = G \\cdot m_i \\cdot m_j \\; / \\; (d_{ij}^2 + \\varepsilon)

        Parameters
        ----------
        other:
            The other node.
        G:
            Gravitational constant.

        Returns
        -------
        float
            Scalar gravitational force.
        """
        diff = self.position - other.position
        dist_sq = float(np.dot(diff, diff))
        force = G * self.mass * other.mass / (dist_sq + _EPSILON)
        return force

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Replace the node's internal state dict.

        Parameters
        ----------
        new_state:
            Dictionary of key-value pairs to store (e.g. model weight hashes,
            embeddings).
        """
        self.state = dict(new_state)
        logger.debug("Node '%s' state updated.", self.node_id)

    def receive_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming message and return a response.

        In this in-process simulation the node simply echoes the message
        with its own node_id appended.

        Parameters
        ----------
        message:
            Arbitrary message dict.

        Returns
        -------
        dict
            Response with ``node_id``, ``received``, and ``timestamp``.
        """
        if not self._online:
            raise RuntimeError(f"Node '{self.node_id}' is offline.")
        return {
            "node_id": self.node_id,
            "received": message,
            "timestamp": time.time(),
        }

    def go_offline(self) -> None:
        """Simulate a network partition by marking this node as offline."""
        self._online = False
        logger.info("Node '%s' went offline.", self.node_id)

    def go_online(self) -> None:
        """Bring a previously offline node back online."""
        self._online = True
        logger.info("Node '%s' is back online.", self.node_id)

    @property
    def is_online(self) -> bool:
        """Whether the node is currently reachable."""
        return self._online


# --------------------------------------------------------------------------- #
# GravitationalConsensusNetwork                                                #
# --------------------------------------------------------------------------- #


class GravitationalConsensusNetwork:
    """Network of :class:`GravitationalNode` instances that vote on proposals.

    Voting weight for each node is proportional to the total gravitational
    force it receives from all other *online* nodes.

    Parameters
    ----------
    G:
        Gravitational constant used in force calculations.
    epsilon:
        Numerical stability floor added to squared distances.
    """

    def __init__(self, G: float = _G_DEFAULT, epsilon: float = _EPSILON) -> None:
        self.G = G
        self.epsilon = epsilon
        self._nodes: Dict[str, GravitationalNode] = {}
        logger.debug("GravitationalConsensusNetwork created (G=%.4e).", G)

    # ------------------------------------------------------------------ #
    # Node management                                                      #
    # ------------------------------------------------------------------ #

    def add_node(self, node: GravitationalNode) -> None:
        """Register a new node in the network.

        Parameters
        ----------
        node:
            :class:`GravitationalNode` to add.
        """
        if node.node_id in self._nodes:
            logger.warning("Node '%s' already registered; replacing.", node.node_id)
        self._nodes[node.node_id] = node
        logger.info("Added node '%s' (mass=%.3f).", node.node_id, node.mass)

    def remove_node(self, node_id: str) -> None:
        """Deregister a node by its ID.

        Parameters
        ----------
        node_id:
            ID of the node to remove.

        Raises
        ------
        KeyError
            If *node_id* is not registered.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in network.")
        del self._nodes[node_id]
        logger.info("Removed node '%s'.", node_id)

    def _online_nodes(self) -> List[GravitationalNode]:
        """Return a list of all currently online nodes."""
        return [n for n in self._nodes.values() if n.is_online]

    # ------------------------------------------------------------------ #
    # Force computation                                                    #
    # ------------------------------------------------------------------ #

    def _node_influence(self, node: GravitationalNode) -> float:
        """Compute total force received by *node* from all other online nodes.

        Parameters
        ----------
        node:
            The target node.

        Returns
        -------
        float
            Sum of forces from all online peers.
        """
        total = 0.0
        for other in self._online_nodes():
            if other.node_id != node.node_id:
                total += node.compute_force(other, G=self.G)
        return total

    # ------------------------------------------------------------------ #
    # Consensus                                                            #
    # ------------------------------------------------------------------ #

    def compute_consensus(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Vote on a proposal using gravitational weighting.

        Each online node casts a binary vote (``accept`` / ``reject``) based on
        whether it acknowledges the proposal (default: accept).  The proposal
        is accepted if the total accepting force exceeds the total force
        multiplied by a majority threshold (> 0.5).

        Parameters
        ----------
        proposal:
            Dictionary representing the consensus proposal.  Can include an
            ``"accept_predicate"`` callable ``(node) -> bool`` to customise
            per-node voting logic.

        Returns
        -------
        dict
            Keys: ``accepted`` (bool), ``confidence`` (float in [0,1]),
            ``votes`` (per-node dict), ``total_force`` (float),
            ``online_nodes`` (int).
        """
        online = self._online_nodes()
        if not online:
            return {
                "accepted": False,
                "confidence": 0.0,
                "votes": {},
                "total_force": 0.0,
                "online_nodes": 0,
            }

        predicate = proposal.get("accept_predicate", None)

        votes: Dict[str, Dict] = {}
        accept_force = 0.0
        total_force = 0.0

        for node in online:
            influence = self._node_influence(node)
            # Default: accept unless a predicate says otherwise
            try:
                if callable(predicate):
                    accepts = bool(predicate(node))
                else:
                    accepts = True
            except Exception as exc:
                logger.warning(
                    "Vote predicate failed for node '%s': %s", node.node_id, exc
                )
                accepts = False

            votes[node.node_id] = {
                "accepts": accepts,
                "influence": influence,
                "mass": node.mass,
            }
            total_force += influence
            if accepts:
                accept_force += influence

        confidence = accept_force / (total_force + _EPSILON)
        accepted = confidence > 0.5

        logger.info(
            "Consensus: accepted=%s, confidence=%.3f, online=%d",
            accepted,
            confidence,
            len(online),
        )

        return {
            "accepted": accepted,
            "confidence": confidence,
            "votes": votes,
            "total_force": total_force,
            "online_nodes": len(online),
        }

    # ------------------------------------------------------------------ #
    # Broadcast                                                            #
    # ------------------------------------------------------------------ #

    def broadcast(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Send *message* to all online nodes and collect responses.

        Nodes that are offline or raise an exception return an error response.

        Parameters
        ----------
        message:
            Arbitrary message dict to broadcast.

        Returns
        -------
        list[dict]
            One response dict per node (including offline nodes).
        """
        responses: List[Dict[str, Any]] = []
        for node in self._nodes.values():
            try:
                resp = node.receive_message(message)
                responses.append(resp)
            except Exception as exc:
                logger.warning(
                    "Broadcast to node '%s' failed: %s", node.node_id, exc
                )
                responses.append({
                    "node_id": node.node_id,
                    "error": str(exc),
                    "timestamp": time.time(),
                })
        return responses

    # ------------------------------------------------------------------ #
    # Topology                                                             #
    # ------------------------------------------------------------------ #

    def get_network_topology(self) -> Dict[str, Any]:
        """Return a description of the current network topology.

        Returns
        -------
        dict
            Keys: ``nodes`` (list of node descriptors), ``edges``
            (pairwise force values), ``total_mass``, ``num_online``.
        """
        node_list = []
        for node in self._nodes.values():
            node_list.append({
                "node_id": node.node_id,
                "mass": node.mass,
                "online": node.is_online,
                "position": node.position.tolist(),
            })

        edges: List[Dict] = []
        ids = list(self._nodes.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                n1 = self._nodes[ids[i]]
                n2 = self._nodes[ids[j]]
                force = n1.compute_force(n2, G=self.G)
                edges.append({
                    "source": ids[i],
                    "target": ids[j],
                    "force": force,
                })

        total_mass = sum(n.mass for n in self._nodes.values())

        return {
            "nodes": node_list,
            "edges": edges,
            "total_mass": total_mass,
            "num_online": sum(1 for n in self._nodes.values() if n.is_online),
        }
