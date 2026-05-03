from llm_gpu_fit.core.data import load_gpus


def test_load_gpus_returns_known_set():
    gpus = load_gpus()
    ids = {g.id for g in gpus}
    assert "h100_80" in ids
    assert "rtx4090" in ids
    assert "m3_max_64" in ids


def test_h100_specs():
    gpus = {g.id: g for g in load_gpus()}
    h100 = gpus["h100_80"]
    assert h100.vram_gb == 80
    assert h100.nvlink is True
    assert h100.form_factor == "datacenter"


def test_rtx4090_no_nvlink():
    gpus = {g.id: g for g in load_gpus()}
    assert gpus["rtx4090"].nvlink is False
    assert gpus["rtx4090"].form_factor == "consumer"
