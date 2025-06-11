# src/visualize_graph.py
# Utility to visualize a graph from the dataset using pyvis.

import os
import sys
from pyvis.network import Network
from datasets import load_dataset

# FIX: Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.linearize import build_graph_from_entry

def visualize_first_graph(dataset_name="web_nlg", subset="webnlg_challenge_2017"):
    """
    Loads the first entry from the dataset from the Hugging Face Hub, builds a graph,
    and saves it as an interactive HTML file.
    """
    print(f"Loading data from '{dataset_name}/{subset}' to visualize graph...")
    
    # Load the first entry from the training data on the hub
    train_dataset = load_dataset(dataset_name, subset, split='train', streaming=True)
    first_entry = next(iter(train_dataset))

    # Build the graph using networkx
    # We need to manually re-package the entry for the existing linearizers
    triples = first_entry['modified_triple_sets']['mtriple_set'][0]
    repackaged_entry = {'modified_triple_sets': {'mtriple_set': [triples]}}
    nx_graph = build_graph_from_entry(repackaged_entry)


    # Create a pyvis network
    pyvis_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line', directed=True)

    # Add nodes and edges from the networkx graph
    pyvis_net.from_nx(nx_graph)

    # Customize physics for better layout
    pyvis_net.set_options("""
    const options = {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "color": {
          "border": "#FFFFFF",
          "background": "#97C2FC"
        },
        "font": {
          "size": 14,
          "face": "tahoma"
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": {
          "type": "dynamic"
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 200
      },
      "physics": {
        "enabled": true,
        "barnesHut": { "theta": 0.5, "gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 95, "springConstant": 0.04, "damping": 0.09, "avoidOverlap": 0.2 },
        "maxVelocity": 50, "minVelocity": 0.1, "solver": "barnesHut",
        "stabilization": { "enabled": true, "iterations": 1000, "updateInterval": 100, "onlyDynamicEdges": false, "fit": true },
        "timestep": 0.5, "adaptiveTimestep": true
      }
    }
    """)

    # Save the visualization to an HTML file
    output_filename = "interactive_graph.html"
    pyvis_net.show(output_filename)
    print(f"Interactive graph visualization saved to {output_filename}")

if __name__ == "__main__":
    visualize_first_graph()
