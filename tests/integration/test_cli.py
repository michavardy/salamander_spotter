"""The `app` console entrypoint (spec §5.4)."""

from __future__ import annotations

import json
from pathlib import Path

from app.__main__ import main


def test_cli_migrate(tmp_path, capsys):
    rc = main(["migrate", "--data-dir", str(tmp_path / "data")])
    assert rc == 0
    assert (tmp_path / "data" / "app.duckdb").exists()
    assert "0001_initial.sql" in capsys.readouterr().out


def test_cli_import_dataset(tmp_path, capsys, dataset_dir):
    rc = main(["import-dataset", str(dataset_dir), "--data-dir", str(tmp_path / "data")])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["images_added"] == 7
    assert report["llm_calls"] == 0
    assert (tmp_path / "data" / "contours.db").exists()


def test_cli_backup_restore_roundtrip(tmp_path, capsys, dataset_dir):
    data = tmp_path / "data"
    main(["import-dataset", str(dataset_dir), "--data-dir", str(data)])
    capsys.readouterr()

    dest = tmp_path / "backups"
    rc = main(["backup", "--data-dir", str(data), "--dest", str(dest)])
    assert rc == 0
    archive = json.loads(capsys.readouterr().out)["archive"]
    assert Path(archive).exists()

    rc = main(["restore", archive, "--data-dir", str(tmp_path / "restored")])
    assert rc == 0
    assert (tmp_path / "restored" / "app.duckdb").exists()
    assert (tmp_path / "restored" / "contours.db").exists()


def test_cli_import_model(tmp_path, capsys):
    data = tmp_path / "data"
    main(["migrate", "--data-dir", str(data)])
    capsys.readouterr()

    weights = tmp_path / "e2e_ckpt_fold0.pt"
    weights.write_bytes(b"checkpoint")
    rc = main([
        "import-model", "e2e_transformer_fold0", str(weights),
        "--kind", "aggregator", "--metric", "a=0.19", "--metric", "e=0.59",
        "--make-active", "--data-dir", str(data),
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == 0.5 * 0.19 + 0.5 * 0.59
    assert (data / "models" / "e2e_transformer_fold0" / "weights.pt").exists()


def test_cli_import_model_bad_metric_letter(tmp_path, capsys):
    data = tmp_path / "data"
    main(["migrate", "--data-dir", str(data)])
    capsys.readouterr()
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"x")
    rc = main(["import-model", "m1", str(weights), "--metric", "z=1", "--data-dir", str(data)])
    assert rc == 1


def test_cli_export_import_full(tmp_path, capsys, dataset_dir):
    data = tmp_path / "data"
    main(["import-dataset", str(dataset_dir), "--data-dir", str(data)])
    capsys.readouterr()

    archive = tmp_path / "full.tar.gz"
    assert main(["export", "--full", str(archive), "--data-dir", str(data)]) == 0
    assert main(["import", "--full", str(archive), "--data-dir", str(tmp_path / "host2")]) == 0
    assert (tmp_path / "host2" / "images" / "raw").is_dir()
