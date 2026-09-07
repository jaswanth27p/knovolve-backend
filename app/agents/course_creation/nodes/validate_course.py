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

    # Chapter titles must be globally unique: persist_course resolves concept
    # references by exact title, so a title appearing twice — even inside one
    # module — makes every concept naming it ambiguous. Regeneration is scoped
    # to just the modules that produced the collision via the rerun_modules
    # hint consumed by generate_chapters.
    title_modules: dict[str, list[str]] = {}
    for module in modules:
        for ch in module["chapters"]:
            title_modules.setdefault(ch["title"], []).append(module["title"])
    duplicate_titles = {t for t, ms in title_modules.items() if len(ms) > 1}
    if duplicate_titles:
        title = sorted(duplicate_titles)[0]
        rerun = sorted(set(title_modules[title]))
        return {
            **state,
            "error": f"duplicate chapter title {title!r} found more than once "
            f"across the course",
            "rerun_modules": rerun,
        }

    chapter_titles = set(title_modules)
    for concept in concepts:
        if concept["chapter_title"] not in chapter_titles:
            return {
                **state,
                "error": f"concept {concept['name']!r} references unknown "
                f"chapter {concept['chapter_title']!r}",
            }

    concept_names = {c["name"] for c in concepts}
    for edge in concept_edges:
        if edge["concept_name"] not in concept_names or edge["prerequisite_name"] not in concept_names:
            return {**state, "error": f"concept_edge references unknown concept: {edge}"}

    if _has_cycle(concept_edges):
        return {**state, "error": "concept prerequisite graph is cyclic"}

    return {**state, "error": None}
