# src/linearize.py
# Contains functions for converting graph data into a linear text format.

import networkx as nx
from typing import List, Dict, Any

def simple_concise(entry: Dict[str, Any]) -> str:
    """
    A simple and concise linearization strategy.
    Example: "John | birth place | London. London | type | city."
    """
    triples = entry['modified_triple_sets']['mtriple_set'][0]
    linearized_triples = [f"{s} | {p} | {o}" for s, p, o in triples]
    return ". ".join(linearized_triples) + "."

def simple_verbose(entry: Dict[str, Any]) -> str:
    """
    A more verbose linearization.
    Example: "Subject: John; Predicate: birth place; Object: London."
    """
    triples = entry['modified_triple_sets']['mtriple_set'][0]
    linearized_triples = [f"Subject: {s}; Predicate: {p}; Object: {o}" for s, p, o in triples]
    return ". ".join(linearized_triples)

def text_based_linearization(entry: Dict[str, Any]) -> str:
    """
    Uses special tokens to denote graph structure.
    Example: "<S> John <P> birth place <O> London <S> London <P> type <O> city"
    """
    triples = entry['modified_triple_sets']['mtriple_set'][0]
    parts = []
    for s, p, o in triples:
        parts.extend(["<S>", s, "<P>", p, "<O>", o])
    return " ".join(parts)

def get_linearizer(strategy: str):
    """
    Returns the linearization function based on the strategy name.
    """
    if strategy == 'simple_concise':
        return simple_concise
    elif strategy == 'simple_verbose':
        return simple_verbose
    elif strategy == 'text_based':
        return text_based_linearization
    else:
        raise ValueError(f"Unknown linearization strategy: {strategy}")

def build_graph_from_entry(entry: Dict[str, Any]) -> nx.DiGraph:
    """
    Builds a networkx graph from a single dataset entry.
    """
    graph = nx.DiGraph()
    triples = entry['modified_triple_sets']['mtriple_set'][0]
    for s, p, o in triples:
        # networkx can't have nodes that are the same as edges, so we clean them up
        s_clean = s.replace("_", " ")
        o_clean = o.replace("_", " ")
        p_clean = p.replace("_", " ")
        graph.add_edge(s_clean, o_clean, label=p_clean)
    return graph
