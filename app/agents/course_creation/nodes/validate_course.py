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
    for module in state["modules"]:
        if not module["chapters"]:
            return {**state, "error": f"module '{module['title']}' has no chapters"}

    concept_names = {c["name"] for c in state["concepts"]}
    for edge in state["concept_edges"]:
        if edge["concept_name"] not in concept_names or edge["prerequisite_name"] not in concept_names:
            return {**state, "error": f"concept_edge references unknown concept: {edge}"}

    if _has_cycle(state["concept_edges"]):
        return {**state, "error": "concept prerequisite graph is cyclic"}

    return {**state, "error": None}
