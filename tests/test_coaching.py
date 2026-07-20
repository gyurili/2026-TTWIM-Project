from app.services.coaching import _build_raw_prompt


RAW_SECTION = "\uc6d0\ubcf8 row \ub370\uc774\ud130"
RATIO_SECTION = "\ube44\uc815\uc0c1 \ube44\uc728"
DTW_SECTION = "\uac1c\uc778 \uae30\uc900\uc120 \ub300\ube44 \ud3b8\ucc28"


def test_build_raw_prompt_includes_all_sections():
    channels = {
        "acc_x": [0.1, 0.2, 0.3],
        "acc_y": [0.4, 0.5, 0.6],
        "acc_z": [0.7, 0.8, 0.9],
        "gyro_x": [1.0, 1.1, 1.2],
        "gyro_y": [1.3, 1.4, 1.5],
        "gyro_z": [1.6, 1.7, 1.8],
    }
    dtw = {
        "status": "alert",
        "avgSimilarity": 60,
        "trend": "worsening",
        "shapeDev": 2.1,
        "cadenceDev": 3.2,
        "amplitudeDev": 1.1,
        "timeline": [80, 70, 60],
    }

    prompt = _build_raw_prompt(channels, 0.5, dtw)

    assert RAW_SECTION in prompt
    for ch in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
        assert ch in prompt
    assert "0.7, 0.8, 0.9" in prompt
    assert RATIO_SECTION in prompt
    assert "50%" in prompt
    assert DTW_SECTION in prompt
    assert "cadenceDev" in prompt
