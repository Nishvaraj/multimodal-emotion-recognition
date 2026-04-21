import ast
from pathlib import Path


FER2013_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
RAVDESS_LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

# Representative, deterministic confidence values chosen to exercise MATCH/PARTIAL/MISMATCH.
FER_CONFIDENCE = {
    "angry": 0.95,
    "disgust": 0.84,
    "fear": 0.71,
    "happy": 0.66,
    "neutral": 0.58,
    "sad": 0.43,
    "surprise": 0.31,
}

RAVDESS_CONFIDENCE = {
    "angry": 0.93,
    "calm": 0.79,
    "disgust": 0.69,
    "fearful": 0.61,
    "happy": 0.56,
    "neutral": 0.49,
    "sad": 0.37,
    "surprised": 0.22,
}


def _load_calculate_concordance():
    """Load _calculate_concordance from backend/main.py without importing heavy runtime dependencies."""
    backend_main = Path(__file__).resolve().parents[1] / "backend" / "main.py"
    source = backend_main.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(backend_main))

    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_calculate_concordance":
            target = node
            break

    if target is None:
        raise AssertionError("Could not find _calculate_concordance in backend/main.py")

    module = ast.Module(body=[target], type_ignores=[])
    code = compile(module, filename=str(backend_main), mode="exec")
    namespace = {}
    exec(code, namespace)
    return namespace["_calculate_concordance"]


def _oracle_concordance(facial_emotion, speech_emotion, facial_confidence, speech_confidence):
    """Independent oracle mirroring the documented deterministic concordance policy."""
    if facial_emotion == speech_emotion:
        score = (facial_confidence + speech_confidence) / 2
        if score > 0.7:
            concordance = "MATCH"
        elif score >= 0.4:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"
    else:
        score = 1 - abs(facial_confidence - speech_confidence)
        if score >= 0.5:
            concordance = "PARTIAL"
        else:
            concordance = "MISMATCH"

    return concordance, round(score * 100)


def test_calculate_concordance_all_56_label_combinations():
    calculate_concordance = _load_calculate_concordance()

    results = []
    for facial_label in FER2013_LABELS:
        for speech_label in RAVDESS_LABELS:
            facial_conf = FER_CONFIDENCE[facial_label]
            speech_conf = RAVDESS_CONFIDENCE[speech_label]

            actual = calculate_concordance(facial_label, speech_label, facial_conf, speech_conf)
            expected = _oracle_concordance(facial_label, speech_label, facial_conf, speech_conf)
            assert actual == expected

            results.append((facial_label, speech_label, facial_conf, speech_conf, actual[0], actual[1]))

    print("FER2013 x RAVDESS concordance matrix (56 combinations)")
    print("facial\tspeech\tf_conf\ts_conf\tconcordance\tscore")
    for row in results:
        facial_label, speech_label, facial_conf, speech_conf, concordance, score = row
        print(
            f"{facial_label}\t{speech_label}\t{facial_conf:.2f}\t{speech_conf:.2f}\t{concordance}\t{score}"
        )

    assert len(results) == 56
    assert any(row[4] == "MATCH" for row in results)
    assert any(row[4] == "PARTIAL" for row in results)
    assert any(row[4] == "MISMATCH" for row in results)