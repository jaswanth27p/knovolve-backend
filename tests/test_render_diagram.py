from app.diagrams.render import render_diagram_svg


def test_render_diagram_svg_produces_svg_bytes():
    spec = {
        "nodes": [{"id": "a", "label": "Start"}, {"id": "b", "label": "End"}],
        "edges": [{"source": "a", "target": "b", "label": "then"}],
    }
    result = render_diagram_svg(spec)
    assert result.startswith(b"<?xml") or b"<svg" in result[:200]


def test_render_diagram_svg_handles_edge_without_label():
    spec = {
        "nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        "edges": [{"source": "a", "target": "b"}],
    }
    result = render_diagram_svg(spec)
    assert b"<svg" in result[:200] or result.startswith(b"<?xml")
