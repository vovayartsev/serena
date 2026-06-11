"""
Elixir support based on a pre-built Dexter index, inspired by the Dexterity project
(https://github.com/nshkrdotcom/dexterity).

Dexter (https://github.com/remoteoss/dexter) maintains a SQLite index of an Elixir
codebase in ``.dexter/dexter.db`` (typically kept up to date by the Dexter instance
running in the user's editor, or created via ``dexter init``). This language "server"
does not launch any process at all: it answers LSP requests in-process by querying
that index directly, opening the database strictly read-only.

The index must exist before activation — Serena never creates or modifies it.
"""

import logging
import os
import pathlib
import sqlite3
import threading
from collections.abc import Callable
from typing import Any

from overrides import override

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import Language, LanguageServerConfig
from solidlsp.ls_process import LanguageServerInterface
from solidlsp.ls_utils import PathUtils
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo, StringDict
from solidlsp.settings import SolidLSPSettings

log = logging.getLogger(__name__)

DEXTER_DB_RELATIVE_PATH = os.path.join(".dexter", "dexter.db")

_MODULE_DEF_KINDS = ("module", "defprotocol", "defimpl")

# LSP SymbolKind values
_SK_MODULE = 2
_SK_FUNCTION = 12
_SK_STRUCT = 23
_SK_TYPE_PARAMETER = 26

_DEF_KIND_TO_SYMBOL_KIND = {
    "module": _SK_MODULE,
    "defprotocol": _SK_MODULE,
    "defimpl": _SK_MODULE,
    "def": _SK_FUNCTION,
    "defp": _SK_FUNCTION,
    "defstruct": _SK_STRUCT,
    "type": _SK_TYPE_PARAMETER,
}

# SQL fragment matching a module column against :module, accounting for partially
# qualified names on either side (Dexter records references through aliases with the
# module name as written in the source, e.g. "UserService" for
# "TestRepo.Services.UserService").
_MODULE_MATCH_SQL = "(module = :module OR :module LIKE '%.' || module OR module LIKE '%.' || :module)"


class DexterIndexReader:
    """
    Read-only reader for a Dexter SQLite index (``.dexter/dexter.db``).

    The database is opened in SQLite read-only mode; the index is maintained externally
    (by the Dexter instance of the user's editor or by running ``dexter init``).
    Since the index may have been produced with a different root path (e.g. when it was
    built on another machine), stored absolute paths are matched against the repository
    root by relative-path suffix where necessary.
    """

    def __init__(self, db_path: str, repository_root_path: str) -> None:
        self.db_path = db_path
        self.repository_root_path = repository_root_path
        db_uri = f"file:{pathlib.Path(db_path).as_posix()}?mode=ro"
        self._conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        self._index_root: str | None = None  # root prefix used in paths stored in the index, inferred lazily

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params or {}).fetchall()

    def db_path_for_relative(self, relative_path: str) -> str | None:
        """:return: the path under which the given repo-relative file is stored in the index, or None if not indexed."""
        posix_relative = pathlib.PurePath(relative_path).as_posix()
        abs_path = pathlib.Path(self.repository_root_path, relative_path).as_posix()
        rows = self.query("SELECT path FROM files WHERE path = :path", {"path": abs_path})
        if not rows:
            # fall back to suffix matching for indexes built with a different root path
            rows = [
                row
                for row in self.query("SELECT path FROM files WHERE path LIKE :pattern", {"pattern": f"%/{posix_relative}"})
                if row[0].endswith(f"/{posix_relative}")
            ]
        if not rows:
            return None
        db_path = str(rows[0][0])
        if self._index_root is None:
            self._index_root = db_path[: -len(f"/{posix_relative}")]
        return db_path

    def relative_for_db_path(self, db_path: str) -> str | None:
        """:return: the repo-relative path for a path stored in the index, or None if it lies outside the repository."""
        for root in (self.repository_root_path, self._index_root):
            if root is not None and pathlib.PurePosixPath(db_path).is_relative_to(pathlib.Path(root).as_posix()):
                relative = str(pathlib.PurePosixPath(db_path).relative_to(pathlib.Path(root).as_posix()))
                if os.path.exists(os.path.join(self.repository_root_path, relative)):
                    return relative
        return None

    def file_mtime_ns(self, db_path: str) -> int | None:
        rows = self.query("SELECT mtime FROM files WHERE path = :path", {"path": db_path})
        return int(rows[0][0]) if rows else None

    def file_definitions(self, db_path: str) -> list[tuple]:
        return self.query(
            "SELECT module, function, arity, kind, line, params FROM definitions WHERE file_path = :path ORDER BY line, arity",
            {"path": db_path},
        )

    def definitions_at(self, db_path: str, line: int) -> list[tuple]:
        return self.query(
            "SELECT module, function, arity, kind, line FROM definitions WHERE file_path = :path AND line = :line ORDER BY arity",
            {"path": db_path, "line": line},
        )

    def refs_at(self, db_path: str, line: int) -> list[tuple]:
        return self.query(
            "SELECT module, function, line, kind FROM refs WHERE file_path = :path AND line = :line",
            {"path": db_path, "line": line},
        )

    def function_refs(self, module: str, function: str) -> list[tuple]:
        return self.query(
            f"SELECT module, function, line, file_path FROM refs WHERE function = :function AND {_MODULE_MATCH_SQL}",
            {"function": function, "module": module},
        )

    def module_refs(self, module: str) -> list[tuple]:
        return self.query(
            f"SELECT module, function, line, file_path FROM refs WHERE function = '' AND {_MODULE_MATCH_SQL}",
            {"module": module},
        )

    def find_definitions(self, module: str, function: str) -> list[tuple]:
        """Find definitions matching the (possibly partially qualified) module and function name; '' for module definitions."""
        if function:
            sql = f"SELECT module, function, kind, line, file_path FROM definitions WHERE function = :function AND {_MODULE_MATCH_SQL}"
            params = {"function": function, "module": module}
        else:
            kinds = ", ".join(f"'{kind}'" for kind in _MODULE_DEF_KINDS)
            sql = f"SELECT module, function, kind, line, file_path FROM definitions WHERE kind IN ({kinds}) AND {_MODULE_MATCH_SQL}"
            params = {"module": module}
        rows = self.query(sql, params)
        # prefer exact module matches over partially qualified ones
        exact = [row for row in rows if row[0] == module]
        return exact or rows


class _DexterityLanguageServerInterface(LanguageServerInterface):
    """
    In-process :class:`LanguageServerInterface` that answers LSP requests by querying a
    Dexter index instead of communicating with a language server process.

    Client requests are dispatched synchronously in :meth:`_send_payload`; client
    notifications (``textDocument/didOpen`` etc.) require no action, since the index is
    maintained externally and Serena applies file edits itself.
    """

    def __init__(
        self,
        reader: DexterIndexReader,
        repository_root_path: str,
        language: Language,
        determine_log_level: Callable[[str], int],
        logger: Callable[[str, str, StringDict | str], None] | None = None,
    ) -> None:
        super().__init__(language, determine_log_level, logger)
        self._reader = reader
        self._repository_root_path = repository_root_path
        self._running = False
        self._stale_files_warned: set[str] = set()

    @override
    def is_running(self) -> bool:
        return self._running

    @override
    def _start(self) -> None:
        self._running = True

    @override
    def _stop(self, timeout: float) -> None:
        self._reader.close()
        self._running = False

    @override
    def _send_payload(self, payload: StringDict) -> None:
        self._trace("solidlsp", "ls", payload)
        method = payload.get("method")
        if method is None:
            return  # a response from the client to a server request; we never send any
        if "id" in payload:
            response: StringDict = {"jsonrpc": "2.0", "id": payload["id"]}
            try:
                response["result"] = self._handle_request(method, payload.get("params") or {})
            except Exception as e:
                log.error("Error handling '%s' from Dexter index: %s", method, e, exc_info=e)
                response["error"] = {"code": -32603, "message": str(e)}
            self._receive_payload(response)
        # client notifications (didOpen/didChange/didClose/exit, ...) require no action

    def _handle_request(self, method: str, params: dict[str, Any]) -> Any:
        match method:
            case "initialize":
                return {
                    "capabilities": {
                        "textDocumentSync": 1,
                        "documentSymbolProvider": True,
                        "referencesProvider": True,
                        "definitionProvider": True,
                    },
                    "serverInfo": {"name": "serena-elixir-dexterity"},
                }
            case "shutdown":
                return None
            case "textDocument/documentSymbol":
                return self._document_symbols(params)
            case "textDocument/references":
                return self._references(params)
            case "textDocument/definition":
                return self._definition(params)
            case _:
                log.debug("Method '%s' is not supported by the Dexter index reader, returning null", method)
                return None

    # ------------------------------- helpers -------------------------------

    def _relative_path_from_uri(self, uri: str) -> str:
        return os.path.relpath(PathUtils.uri_to_path(uri), self._repository_root_path)

    def _read_lines(self, relative_path: str) -> list[str]:
        try:
            with open(os.path.join(self._repository_root_path, relative_path), encoding="utf-8") as f:
                return f.read().splitlines()
        except OSError:
            return []

    @staticmethod
    def _find_token(line_text: str, token: str) -> int | None:
        """:return: the 0-based column of the first word-bounded occurrence of ``token`` in ``line_text``, or None if not found.

        Occurrences preceded by '.' (module-qualified) or ':' (atoms) are not considered matches.
        """
        start = 0
        while True:
            idx = line_text.find(token, start)
            if idx == -1:
                return None
            before_ok = idx == 0 or not (line_text[idx - 1].isalnum() or line_text[idx - 1] in "_.:")
            after = idx + len(token)
            after_ok = after >= len(line_text) or not (line_text[after].isalnum() or line_text[after] == "_")
            if before_ok and after_ok:
                return idx
            start = idx + 1

    @classmethod
    def _token_column(cls, line_text: str, token: str) -> int:
        """Like :meth:`_find_token`, but returns 0 if the token is not found."""
        col = cls._find_token(line_text, token)
        return 0 if col is None else col

    def _location(self, relative_path: str, line0: int, token: str, lines: list[str] | None = None) -> dict[str, Any]:
        lines = lines if lines is not None else self._read_lines(relative_path)
        line_text = lines[line0] if 0 <= line0 < len(lines) else ""
        col = self._token_column(line_text, token)
        return {
            "uri": PathUtils.path_to_uri(os.path.join(self._repository_root_path, relative_path)),
            "range": {
                "start": {"line": line0, "character": col},
                "end": {"line": line0, "character": col + len(token)},
            },
        }

    def _warn_if_stale(self, relative_path: str, db_path: str) -> None:
        if relative_path in self._stale_files_warned:
            return
        index_mtime_ns = self._reader.file_mtime_ns(db_path)
        try:
            disk_mtime_ns = os.stat(os.path.join(self._repository_root_path, relative_path)).st_mtime_ns
        except OSError:
            return
        if index_mtime_ns is not None and disk_mtime_ns > index_mtime_ns:
            self._stale_files_warned.add(relative_path)
            log.warning(
                "The Dexter index entry for '%s' is older than the file on disk; results may be stale. "
                "Let your editor's Dexter instance re-index the project or run `dexter reindex`.",
                relative_path,
            )

    # --------------------------- documentSymbol ----------------------------

    def _document_symbols(self, params: dict[str, Any]) -> list[dict[str, Any]] | None:
        relative_path = self._relative_path_from_uri(params["textDocument"]["uri"])
        db_path = self._reader.db_path_for_relative(relative_path)
        if db_path is None:
            log.warning("File '%s' is not present in the Dexter index", relative_path)
            return None
        self._warn_if_stale(relative_path, db_path)

        lines = self._read_lines(relative_path)
        rows = self._reader.file_definitions(db_path)

        # deduplicate clauses of the same function (and default-argument arity expansions)
        seen: set[tuple[str, str, int]] = set()
        unique_rows = []
        for module, function, arity, kind, line, params_str in rows:
            key = (module, function, line)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append((module, function, arity, kind, line, params_str))
        unique_rows.sort(key=lambda row: row[4])

        def make_symbol(name: str, kind: int, line0: int, end_line0: int, detail: str, token: str | None = None) -> dict[str, Any]:
            token = token if token is not None else name
            line_text = lines[line0] if 0 <= line0 < len(lines) else ""
            col = self._token_column(line_text, token)
            end_character = len(lines[end_line0]) if 0 <= end_line0 < len(lines) else 0
            return {
                "name": name,
                "kind": kind,
                "detail": detail,
                "range": {"start": {"line": line0, "character": 0}, "end": {"line": end_line0, "character": end_character}},
                "selectionRange": {"start": {"line": line0, "character": col}, "end": {"line": line0, "character": col + len(token)}},
                "children": [],
            }

        module_nodes: dict[str, dict[str, Any]] = {}
        module_names = [module for module, function, *_ in unique_rows if function == ""]
        root_symbols: list[dict[str, Any]] = []

        for i, (module, function, arity, kind, line, params_str) in enumerate(unique_rows):
            line0 = line - 1
            # the entity is assumed to extend to the line before the next definition (approximation,
            # since the index stores no end positions)
            end_line0 = (unique_rows[i + 1][4] - 2) if i + 1 < len(unique_rows) else max(len(lines) - 1, line0)
            end_line0 = max(end_line0, line0)

            if function == "":  # module-like definition
                # display the name as written in the source: strip the enclosing module's prefix
                parents = [m for m in module_names if m != module and module.startswith(m + ".")]
                parent = max(parents, key=len) if parents else None
                display_name = module[len(parent) + 1 :] if parent else module
                node = make_symbol(display_name, _DEF_KIND_TO_SYMBOL_KIND.get(kind, _SK_MODULE), line0, end_line0, kind)
                module_nodes[module] = node
                if parent is not None and parent in module_nodes:
                    module_nodes[parent]["children"].append(node)
                else:
                    root_symbols.append(node)
            else:
                detail = f"{kind} {function}/{arity}" + (f" ({params_str})" if params_str else "")
                # private functions carry a "defp " name prefix so that agents see the visibility;
                # the selection range stays on the bare identifier
                display_name = f"defp {function}" if kind == "defp" else function
                node = make_symbol(display_name, _DEF_KIND_TO_SYMBOL_KIND.get(kind, _SK_FUNCTION), line0, end_line0, detail, token=function)
                if module in module_nodes:
                    module_nodes[module]["children"].append(node)
                else:
                    root_symbols.append(node)

        # extend each module's range to cover its descendants
        def extend_range(node: dict[str, Any]) -> dict[str, Any]:
            for child in node["children"]:
                child_end = extend_range(child)["range"]["end"]
                if (child_end["line"], child_end["character"]) > (node["range"]["end"]["line"], node["range"]["end"]["character"]):
                    node["range"]["end"] = dict(child_end)
            return node

        return [extend_range(node) for node in root_symbols]

    # ------------------------- references/definition ------------------------

    def _resolve_target(self, relative_path: str, db_path: str, line0: int, character: int) -> tuple[str, str] | None:
        """
        Determine the (module, function) pair targeted at the given position; function is '' for modules.
        The position may be on a definition (the standard case for Serena) or on a reference.
        """
        line_text = (self._read_lines(relative_path) or [""])[line0] if line0 >= 0 else ""

        def best_match(candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
            # prefer the candidate whose name token spans the requested character
            for module, function in candidates:
                token = function or module.rsplit(".", 1)[-1]
                col = self._token_column(line_text, token)
                if col <= character < col + len(token):
                    return module, function
            # otherwise prefer functions over modules
            for module, function in candidates:
                if function:
                    return module, function
            return candidates[0] if candidates else None

        definitions = self._reader.definitions_at(db_path, line0 + 1)
        if definitions:
            return best_match([(module, function) for module, function, *_ in definitions])
        references = self._reader.refs_at(db_path, line0 + 1)
        return best_match([(module, function) for module, function, *_ in references])

    def _request_context(self, params: dict[str, Any]) -> tuple[str, str, tuple[str, str] | None]:
        relative_path = self._relative_path_from_uri(params["textDocument"]["uri"])
        db_path = self._reader.db_path_for_relative(relative_path)
        if db_path is None:
            log.warning("File '%s' is not present in the Dexter index", relative_path)
            return relative_path, "", None
        self._warn_if_stale(relative_path, db_path)
        position = params["position"]
        return relative_path, db_path, self._resolve_target(relative_path, db_path, position["line"], position["character"])

    def _references(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, target = self._request_context(params)
        if target is None:
            return []
        module, function = target
        rows = self._reader.function_refs(module, function) if function else self._reader.module_refs(module)

        locations = []
        seen: set[tuple[str, int]] = set()
        for ref_module, ref_function, line, file_path in rows:
            ref_relative = self._reader.relative_for_db_path(file_path)
            if ref_relative is None or (ref_relative, line) in seen:
                continue
            seen.add((ref_relative, line))
            token = ref_function or ref_module.rsplit(".", 1)[-1]
            locations.append(self._location(ref_relative, line - 1, token))

        if function:
            # The index does not record bare local calls (e.g. `helper(x)` within the defining
            # module, which is the only way private functions can be called), so the defining
            # files are additionally scanned for word-bounded occurrences of the function name.
            definition_rows = self._reader.find_definitions(module, function)
            definition_lines_by_file: dict[str, set[int]] = {}
            for *_, def_line, file_path in definition_rows:
                definition_lines_by_file.setdefault(file_path, set()).add(def_line)
            for file_path, definition_lines in definition_lines_by_file.items():
                def_relative = self._reader.relative_for_db_path(file_path)
                if def_relative is None:
                    continue
                lines = self._read_lines(def_relative)
                for line0, line_text in enumerate(lines):
                    if line0 + 1 in definition_lines or (def_relative, line0 + 1) in seen:
                        continue
                    if self._find_token(line_text, function) is None:
                        continue
                    seen.add((def_relative, line0 + 1))
                    locations.append(self._location(def_relative, line0, function, lines=lines))

        if params.get("context", {}).get("includeDeclaration"):
            for def_module, def_function, _kind, line, file_path in self._reader.find_definitions(module, function):
                def_relative = self._reader.relative_for_db_path(file_path)
                if def_relative is None or (def_relative, line) in seen:
                    continue
                seen.add((def_relative, line))
                locations.append(self._location(def_relative, line - 1, def_function or def_module.rsplit(".", 1)[-1]))

        return locations

    def _definition(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        _, _, target = self._request_context(params)
        if target is None:
            return []
        module, function = target

        locations = []
        seen: set[tuple[str, int]] = set()
        for def_module, def_function, _kind, line, file_path in self._reader.find_definitions(module, function):
            def_relative = self._reader.relative_for_db_path(file_path)
            if def_relative is None or (def_relative, line) in seen:
                continue
            seen.add((def_relative, line))
            locations.append(self._location(def_relative, line - 1, def_function or def_module.rsplit(".", 1)[-1]))
        return locations


class ElixirDexterity(SolidLanguageServer):
    """
    Elixir support backed by a pre-built Dexter index (``.dexter/dexter.db``), queried
    in-process — no language server process is launched, and the index is opened
    strictly read-only. The index must be created and kept up to date externally,
    e.g. by the Dexter instance running in your editor or by running ``dexter init``
    (https://github.com/remoteoss/dexter).

    Private functions are reported with a ``defp `` name prefix (e.g. ``defp validate``),
    making the visibility apparent in symbol overviews.

    You can pass the following entries in ``ls_specific_settings["elixir_dexterity"]``:
        - db_path: Path to the ``dexter.db`` index file. By default, ``.dexter/dexter.db``
          is searched in the project root and its ancestors (Dexter places the index next
          to ``.git``, which may be above the project root in monorepos).
    """

    def __init__(self, config: LanguageServerConfig, repository_root_path: str, solidlsp_settings: SolidLSPSettings):
        super().__init__(
            config,
            repository_root_path,
            # no process is launched; the launch info is required by the base class but unused
            ProcessLaunchInfo(cmd=[], cwd=repository_root_path),
            "elixir",
            solidlsp_settings,
        )
        self.server_ready = threading.Event()

    def _find_dexter_db(self) -> str:
        db_path = self._custom_settings.get("db_path")
        if db_path is not None:
            if not os.path.isfile(db_path):
                raise FileNotFoundError(
                    f"The configured Dexter index (ls_specific_settings.elixir_dexterity.db_path) does not exist: {db_path}"
                )
            return str(db_path)
        directory = os.path.abspath(self.repository_root_path)
        while True:
            candidate = os.path.join(directory, DEXTER_DB_RELATIVE_PATH)
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent
        raise FileNotFoundError(
            f"No Dexter index ({DEXTER_DB_RELATIVE_PATH}) found in '{self.repository_root_path}' or any of its ancestors.\n"
            "Create one by running `dexter init` in the project root (https://github.com/remoteoss/dexter), "
            "or let your editor's Dexter instance index the project.\n"
            "Alternatively, set 'db_path' in ls_specific_settings.elixir_dexterity to the dexter.db file."
        )

    @override
    def _create_language_server_interface(
        self, process_launch_info: ProcessLaunchInfo, logging_fn: Callable[[str, str, StringDict | str], None] | None
    ) -> LanguageServerInterface:
        db_path = self._find_dexter_db()
        log.info("Using Dexter index at %s (read-only, no language server process is launched)", db_path)
        reader = DexterIndexReader(db_path, self.repository_root_path)
        return _DexterityLanguageServerInterface(reader, self.repository_root_path, self.language, self._determine_log_level, logging_fn)

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        # same ignores as for the Expert-based Elixir support, plus Dexter's index directory
        return super().is_ignored_dirname(dirname) or dirname in [
            "_build",
            "deps",
            "node_modules",
            ".elixir_ls",
            ".expert",
            ".dexter",
            "cover",
        ]

    @override
    def is_ignored_path(self, relative_path: str, ignore_unsupported_files: bool = True) -> bool:
        """Check if a path should be ignored for symbol indexing."""
        if relative_path.endswith("mix.exs"):
            # These are project configuration files, not source code with symbols to index
            return True

        return super().is_ignored_path(relative_path, ignore_unsupported_files)

    def _start_server(self) -> None:
        self.server.start()
        init_response = self.server.send.initialize(
            {
                "processId": os.getpid(),
                "rootUri": PathUtils.path_to_uri(self.repository_root_path),
                "capabilities": {},
            }
        )
        assert "capabilities" in init_response
        self.server.notify.initialized({})
        self.server_ready.set()
