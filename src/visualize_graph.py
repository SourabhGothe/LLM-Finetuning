# src/visualize_graph.py
# Utility to visualize a graph from the dataset using pyvis.

import json
from pyvis.network import Network
from src.linearize import build_graph_from_entry

def visualize_first_graph(data_file_path="data/webnlg/webnlg-dataset-v3.0/en/3-shot/train.json"):
    """
    Loads the first entry from the dataset, builds a graph,
    and saves it as an interactive HTML file.
    """
    print(f"Loading data from {data_file_path} to visualize graph...")
    
    # Load the first entry from the training data
    with open(data_file_path, 'r') as f:
        # The file is a list of dictionaries under a key
        data = json.load(f)
        first_entry = data['entries']['1'][0]

    # Build the graph using networkx
    nx_graph = build_graph_from_entry(first_entry)

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
        "barnesHut": {
          "theta": 0.5,
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.2
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        },
        "timestep": 0.5,
        "adaptiveTimestep": true
      }
    }
    """)


    # Save the visualization to an HTML file
    output_filename = "interactive_graph.html"
    pyvis_net.show(output_filename)
    print(f"Interactive graph visualization saved to {output_filename}")

if __name__ == "__main__":
    visualize_first_graph()
