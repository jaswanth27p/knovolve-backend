from unittest.mock import MagicMock, patch

from app.documents import render


def test_course_render_uses_template_and_pdf_bytes():
    payload = {
        "title": "Course",
        "subtitle": "Export course",
        "generated_at": "2026-09-11T10:00:00+00:00",
        "modules": [],
    }
    fake_html = MagicMock()
    fake_html.write_pdf.return_value = b"%PDF-fake"
    with patch("app.documents.render.HTML", return_value=fake_html) as mock_html:
        result = render.render_course_pdf(payload)

    assert result == b"%PDF-fake"
    html_source = mock_html.call_args.kwargs["string"]
    assert "Course" in html_source
    assert "Export course" in html_source


def test_assignments_renderer_includes_answer_key_heading():
    payload = {
        "title": "Course",
        "subtitle": "Export assignments",
        "generated_at": "2026-09-11T10:00:00+00:00",
        "modules": [
            {
                "title": "M",
                "is_additional": False,
                "chapters": [
                    {
                        "title": "C",
                        "assignments": [
                            {
                                "label": "Version 1",
                                "level": "chapter",
                                "questions": [
                                    {
                                        "order": 1,
                                        "type": "mcq",
                                        "text": "Q?",
                                        "options": ["A", "B"],
                                        "difficulty": "easy",
                                        "concept_tag": "tag",
                                        "correct_answer": "A",
                                        "explanation": "Because A.",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    fake_html = MagicMock()
    fake_html.write_pdf.return_value = b"%PDF-fake"
    with patch("app.documents.render.HTML", return_value=fake_html) as mock_html:
        render.render_assignments_pdf(payload)

    html_source = mock_html.call_args.kwargs["string"]
    assert "Answer key" in html_source
    assert "Because A." in html_source


def test_diagram_helper_returns_none_for_missing_url():
    assert render.diagram_data_uri(None) is None


def test_diagram_helper_embeds_svg_bytes():
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"<svg/>"

    with patch("urllib.request.urlopen", return_value=_FakeResponse()):
        uri = render.diagram_data_uri("http://example.invalid/diagram.svg")

    assert uri is not None
    assert uri.startswith("data:image/svg+xml;base64,")
