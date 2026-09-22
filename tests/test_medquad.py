from data.build_instruction_data import build_records
from data.medquad import iter_medquad_records


def test_iter_medquad_records_extracts_question_answer_pairs(tmp_path):
    root = tmp_path / "source"
    xml_dir = root / "MedQuAD" / "1_CancerGov_QA"
    xml_dir.mkdir(parents=True)
    (xml_dir / "sample.xml").write_text(
        """<?xml version=\"1.0\"?>
<Document id=\"sample\" source=\"CancerGov\">
  <QAPairs>
    <QAPair pid=\"1\">
      <Question qtype=\"information\">What is diabetes?</Question>
      <Answer>Diabetes is a chronic condition involving high blood glucose.</Answer>
    </QAPair>
    <QAPair pid=\"2\">
      <Question>Question without an answer</Question>
      <Answer></Answer>
    </QAPair>
  </QAPairs>
</Document>
""",
        encoding="utf-8",
    )

    records = list(iter_medquad_records(root))
    assert records == [
        (
            "What is diabetes?",
            "Diabetes is a chronic condition involving high blood glucose.",
            "MedQuAD/1_CancerGov_QA/sample.xml",
        )
    ]


def test_builder_adds_medquad_to_supervised_data(tmp_path):
    root = tmp_path / "source"
    xml_dir = root / "MedQuAD" / "1_CancerGov_QA"
    xml_dir.mkdir(parents=True)
    (xml_dir / "sample.xml").write_text(
        """<Document><QAPairs>
<QAPair><Question>What is hypertension?</Question>
<Answer>Hypertension is persistent high blood pressure.</Answer></QAPair>
</QAPairs></Document>""",
        encoding="utf-8",
    )

    supervised, audit = build_records(root)
    medquad = [record for record in supervised if record["category"] == "medquad_qa"]
    assert len(medquad) == 1
    assert medquad[0]["input"] == "What is hypertension?"
    assert medquad[0]["response"] == "Hypertension is persistent high blood pressure."
    assert medquad[0]["source"] == "MedQuAD/1_CancerGov_QA/sample.xml"
    assert any(record["category"] == "medquad_qa" for record in audit)
