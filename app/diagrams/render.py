import graphviz


def render_diagram_svg(spec: dict) -> bytes:
    graph = graphviz.Digraph(format="svg")
    for node in spec["nodes"]:
        graph.node(node["id"], label=node["label"])
    for edge in spec["edges"]:
        graph.edge(edge["source"], edge["target"], label=edge.get("label") or "")
    return graph.pipe()
