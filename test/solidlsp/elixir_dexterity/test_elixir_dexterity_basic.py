"""
Tests for the Dexter-index-based Elixir support (language `elixir_dexterity`).

These tests do not require the dexter binary or an Elixir installation: the index
(.dexter/dexter.db) is created from a committed SQL dump, and no language server
process is launched — requests are answered in-process from the SQLite index.
"""

import os
from collections.abc import Iterator
from typing import Any

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls_config import Language
from test.conftest import get_repo_path

from . import create_dexter_index, remove_dexter_index

pytestmark = [pytest.mark.elixir_dexterity]


@pytest.fixture(scope="module", autouse=True)
def dexter_index() -> Iterator[None]:
    repo_path = get_repo_path(Language.ELIXIR_DEXTERITY)
    create_dexter_index(repo_path)
    yield
    remove_dexter_index(repo_path)


def _iter_symbols(symbols: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for symbol in symbols:
        yield symbol
        yield from _iter_symbols(symbol.get("children", []))


def _find_symbol(symbols: list[dict[str, Any]], name: str, kind: int | None = None) -> dict[str, Any] | None:
    for symbol in _iter_symbols(symbols):
        if symbol.get("name") == name and (kind is None or symbol.get("kind") == kind):
            return symbol
    return None


class TestElixirDexterityBasic:
    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_request_document_symbols(self, language_server: SolidLanguageServer):
        """Modules and functions of a file are reported with a proper hierarchy."""
        file_path = os.path.join("lib", "models.ex")
        _, roots = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        models = _find_symbol(roots, "TestRepo.Models", kind=2)
        assert models is not None, "Should find the TestRepo.Models module"

        for module_name in ["User", "Item", "Order"]:
            module = _find_symbol(models["children"], module_name, kind=2)
            assert module is not None, f"Should find module '{module_name}' nested in TestRepo.Models"

        user = _find_symbol(models["children"], "User", kind=2)
        assert user is not None
        function_names = [child["name"] for child in user["children"] if child["kind"] == 12]
        assert "new" in function_names, f"Should find function 'new' in User, got {function_names}"
        assert "has_role?" in function_names

    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_request_references_cross_file(self, language_server: SolidLanguageServer):
        """References to UserService.create_user are found across files."""
        file_path = os.path.join("lib", "services.ex")
        _, roots = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        create_user = _find_symbol(roots, "create_user", kind=12)
        assert create_user is not None, "Should find UserService.create_user in services.ex"

        sel_start = create_user["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])

        assert references, "Should find references to UserService.create_user"
        reference_paths = {ref["relativePath"] for ref in references}
        assert os.path.join("lib", "examples.ex") in reference_paths, f"Should find references in examples.ex, got {reference_paths}"
        # the call via the relative alias `UserService.create_user(...)` within services.ex must be found, too
        assert os.path.join("lib", "services.ex") in reference_paths, f"Should find the reference in services.ex, got {reference_paths}"

    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_request_references_module(self, language_server: SolidLanguageServer):
        """References to a module (aliases) are found."""
        file_path = os.path.join("lib", "models.ex")
        _, roots = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        user_module = _find_symbol(roots, "User", kind=2)
        assert user_module is not None

        sel_start = user_module["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])

        assert references, "Should find references to the User module"
        reference_paths = {ref["relativePath"] for ref in references}
        assert os.path.join("lib", "examples.ex") in reference_paths, f"Should find the alias in examples.ex, got {reference_paths}"

    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_request_definition(self, language_server: SolidLanguageServer):
        """Go-to-definition from a call site resolves to the function definition."""
        # examples.ex line 19 (0-based 18): UserService.create_user(user_service, "1", "Alice", ...)
        file_path = os.path.join("lib", "examples.ex")
        with open(os.path.join(str(get_repo_path(Language.ELIXIR_DEXTERITY)), file_path), encoding="utf-8") as f:
            line_text = f.read().splitlines()[18]
        column = line_text.index("create_user")

        definitions = language_server.request_definition(file_path, 18, column)

        assert definitions, "Should find the definition of create_user"
        assert any(d["relativePath"] == os.path.join("lib", "services.ex") for d in definitions), definitions

    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_private_function_naming_and_references(self, language_server: SolidLanguageServer):
        """Private functions are reported as 'defp <name>' and their references are still found."""
        file_path = os.path.join("lib", "models.ex")
        _, roots = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        calculate_total = _find_symbol(roots, "defp calculate_total", kind=12)
        assert calculate_total is not None, "Private function should be named 'defp calculate_total'"
        assert _find_symbol(roots, "calculate_total") is None, "The bare name should not be reported for private functions"

        # the selection range stays on the bare identifier, so references are still resolvable from it
        sel_start = calculate_total["selectionRange"]["start"]
        references = language_server.request_references(file_path, sel_start["line"], sel_start["character"])
        assert references, "Should find references to the private function calculate_total"
        assert any(ref["relativePath"] == file_path for ref in references)

    @pytest.mark.parametrize("language_server", [Language.ELIXIR_DEXTERITY], indirect=True)
    def test_symbol_ranges_cover_bodies(self, language_server: SolidLanguageServer):
        """Symbol ranges extend over the entity bodies (approximated from the index)."""
        file_path = os.path.join("lib", "models.ex")
        _, roots = language_server.request_document_symbols(file_path).get_all_symbols_and_roots()

        models = _find_symbol(roots, "TestRepo.Models", kind=2)
        user = _find_symbol(roots, "User", kind=2)
        assert models is not None and user is not None
        assert models["range"]["end"]["line"] >= user["range"]["end"]["line"], "Parent module range must cover nested modules"
        assert user["range"]["end"]["line"] > user["range"]["start"]["line"]
