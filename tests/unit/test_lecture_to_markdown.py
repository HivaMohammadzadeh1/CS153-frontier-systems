from learning_memory_os.ingestion.lecture_to_markdown import convert_lecture_py


SAMPLE_LECTURE = '''
from execute_util import text, link, image


def main():
    text("**Lecture 99: Sample**")
    text("This is the intro paragraph.")
    section_one()


def section_one():
    text("### Section One")
    text("First bullet")
    text("Second bullet")
    image("foo.png", width=300)
    link(title="ref", url="https://example.com")
    text("Closing line for section one")


if __name__ == "__main__":
    main()
'''


def test_extracts_text_calls_in_order():
    md = convert_lecture_py(SAMPLE_LECTURE)
    assert "Lecture 99: Sample" in md
    assert "intro paragraph" in md
    assert "Section One" in md
    assert "First bullet" in md
    assert "Closing line for section one" in md


def test_ignores_image_and_link_calls():
    md = convert_lecture_py(SAMPLE_LECTURE)
    assert "foo.png" not in md
    assert "https://example.com" not in md


def test_stripped_output_is_plain_markdown():
    md = convert_lecture_py(SAMPLE_LECTURE)
    assert "def " not in md
    assert "text(" not in md
    assert "image(" not in md
