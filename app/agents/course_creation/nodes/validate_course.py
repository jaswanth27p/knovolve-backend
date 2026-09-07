from app.agents.course_creation.state import CourseCreationState


def _has_cycle(edges: list[dict]) -> bool:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        graph.setdefault(edge["concept_name"], []).append(edge["prerequisite_name"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if color.get(neighbor, WHITE) == GRAY:
                return True
            if color.get(neighbor, WHITE) == WHITE and visit(neighbor):
                return True
        color[node] = BLACK
        return False

    return any(color[node] == WHITE and visit(node) for node in list(graph))


def validate_course(state: CourseCreationState) -> CourseCreationState:
    # build_concept_graph always runs before validate_course (see graph.py's
    # edge wiring, including its retry edges), so modules, concepts and
    # concept_edges are all guaranteed populated by this point.
    modules = state["modules"]
    concepts = state["concepts"]
    concept_edges = state["concept_edges"]
    assert modules is not None
    assert concepts is not None
    assert concept_edges is not None

    for module in modules:
        if not module["chapters"]:
            return {**state, "error": f"module '{module['title']}' has no chapters"}

    concept_names = {c["name"] for c in concepts}
    for edge in concept_edges:
        if edge["concept_name"] not in concept_names or edge["prerequisite_name"] not in concept_names:
            return {**state, "error": f"concept_edge references unknown concept: {edge}"}

    if _has_cycle(concept_edges):
        return {**state, "error": "concept prerequisite graph is cyclic"}

    return {**state, "error": None}
