from __future__ import annotations

from app.services.migration.loader import MigrationPathService


def test_migration_paths_loading(test_settings) -> None:
    service = MigrationPathService(test_settings.resolve_path(test_settings.migration_paths_file))
    paths = service.list_paths()
    assert len(paths) == 1
    assert paths[0]["path_id"] == "test-path"

