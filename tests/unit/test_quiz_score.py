from unittest.mock import MagicMock
from learning_memory_os.eval.quiz import (
    QuizQuestion,
    score_answer,
    average_score,
    QuizScore,
)


def test_score_answer_uses_judge():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {"score": 0.7, "rationale": "partial"}
    q = QuizQuestion(question="q?", rubric="r", concept_id=None)
    s = score_answer(question=q, student_answer="something", judge_llm=fake_llm)
    assert s.score == 0.7
    assert "partial" in s.rationale
    fake_llm.complete_json.assert_called_once()


def test_score_answer_zero_for_empty():
    fake_llm = MagicMock()
    q = QuizQuestion(question="q?", rubric="r")
    s = score_answer(question=q, student_answer="", judge_llm=fake_llm)
    assert s.score == 0.0
    fake_llm.complete_json.assert_not_called()


def test_score_answer_clamps_out_of_range():
    fake_llm = MagicMock()
    fake_llm.complete_json.return_value = {"score": 1.5, "rationale": "ok"}
    q = QuizQuestion(question="q?", rubric="r")
    s = score_answer(question=q, student_answer="x", judge_llm=fake_llm)
    assert s.score == 1.0


def test_average_score_empty():
    assert average_score([]) == 0.0


def test_average_score_basic():
    avg = average_score([QuizScore(0.5, ""), QuizScore(1.0, ""), QuizScore(0.0, "")])
    assert abs(avg - 0.5) < 1e-9
