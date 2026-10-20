from app.reader import lines_from_export


def test_lines_from_export_join_words_and_scale_boxes():
    exported = {
        "pages": [
            {
                "blocks": [
                    {
                        "lines": [
                            {
                                "geometry": ((0.1, 0.2), (0.5, 0.3)),
                                "words": [
                                    {"value": "Paige", "confidence": 0.9},
                                    {"value": "reads", "confidence": 0.7},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    lines = lines_from_export(exported, width=200, height=100)
    assert len(lines) == 1
    assert lines[0].text == "Paige reads"
    assert lines[0].confidence == 0.8
    assert lines[0].box == (20, 20, 100, 30)
