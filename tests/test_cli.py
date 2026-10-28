from staff_video.cli import main


def test_cli_reports_model_loading_failure_without_traceback(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise EOFError("Local model could not be loaded")

    monkeypatch.setattr("staff_video.cli.analyze_video", fail)
    assert main(["analyze", "unseen.mp4"]) == 1
    message = capsys.readouterr().err
    assert "Local model could not be loaded" in message and "Traceback" not in message
