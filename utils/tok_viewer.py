#!/usr/bin/env python3
"""BPE vocabulary viewer - interactive TUI with scrollable table and search."""

import argparse
import base64
import sys

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, Footer, Header, Static
from textual.binding import Binding
from textual.containers import Vertical


def decode_token(b64_str: str) -> tuple[bytes, str]:
    """Decode base64 to bytes and attempt UTF-8 conversion."""
    raw = base64.b64decode(b64_str)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    return raw, text


def format_token(raw: bytes, text: str | None) -> str:
    """Format token for display, handling non-printable characters."""
    if text is None:
        return f"[{raw.hex()}]"

    result = []
    for char in text:
        code = ord(char)
        if code == 0x20:
            result.append("␣")
        elif code == 0x09:
            result.append("→")
        elif code == 0x0A:
            result.append("↵")
        elif code == 0x0D:
            result.append("⏎")
        elif code < 0x20 or (0x7F <= code < 0xA0):
            result.append(f"\\x{code:02x}")
        else:
            result.append(char)
    return "".join(result)


def load_vocab(path: str) -> list[tuple[int, bytes, str | None]]:
    """Load vocabulary file, returning list of (id, raw_bytes, decoded_text)."""
    tokens = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token_id, b64 = line.split(":", 1)
            raw, text = decode_token(b64)
            tokens.append((int(token_id), raw, text))
    return tokens


class VocabViewer(App):
    """Interactive vocabulary viewer with search and fast scrolling."""

    CSS = """
    #search-box {
        dock: top;
        height: 3;
        padding: 0 1;
    }

    #search-input {
        width: 100%;
    }

    #status {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }

    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear"),
        Binding("g", "go_top", "Top"),
        Binding("G", "go_bottom", "Bottom"),
        Binding("pageup", "page_up", "Page Up"),
        Binding("pagedown", "page_down", "Page Down"),
    ]

    def __init__(self, vocab_path: str):
        super().__init__()
        self.vocab_path = vocab_path
        self.all_tokens: list[tuple[int, bytes, str | None]] = []
        self.filtered_tokens: list[tuple[int, bytes, str | None]] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Input(placeholder="Search tokens... (/ to focus, Escape to clear)", id="search-input"),
            id="search-box",
        )
        yield DataTable(id="vocab-table")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Load vocabulary and populate table."""
        self.all_tokens = load_vocab(self.vocab_path)
        self.filtered_tokens = self.all_tokens

        table = self.query_one("#vocab-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True

        table.add_column("ID", width=8, key="id")
        table.add_column("Bytes", width=14, key="bytes")
        table.add_column("Token", width=60, key="token")

        self._populate_table()
        self._update_status()

    def _populate_table(self) -> None:
        """Populate table with current filtered tokens."""
        table = self.query_one("#vocab-table", DataTable)
        table.clear()

        for tid, raw, text in self.filtered_tokens:
            formatted = format_token(raw, text)
            byte_repr = raw.hex() if len(raw) <= 6 else raw[:6].hex() + "…"
            table.add_row(str(tid), byte_repr, formatted, key=str(tid))

    def _update_status(self) -> None:
        """Update status bar with current filter info."""
        status = self.query_one("#status", Static)
        total = len(self.all_tokens)
        shown = len(self.filtered_tokens)
        if shown == total:
            status.update(f"Showing all {total:,} tokens")
        else:
            status.update(f"Showing {shown:,} of {total:,} tokens")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter tokens as user types."""
        query = event.value.lower().strip()

        if not query:
            self.filtered_tokens = self.all_tokens
        else:
            self.filtered_tokens = []
            for tid, raw, text in self.all_tokens:
                # Search in token ID
                if query.isdigit() and query in str(tid):
                    self.filtered_tokens.append((tid, raw, text))
                    continue
                # Search in decoded text
                if text and query in text.lower():
                    self.filtered_tokens.append((tid, raw, text))
                    continue
                # Search in formatted display
                formatted = format_token(raw, text).lower()
                if query in formatted:
                    self.filtered_tokens.append((tid, raw, text))

        self._populate_table()
        self._update_status()

    def action_focus_search(self) -> None:
        """Focus the search input."""
        self.query_one("#search-input", Input).focus()

    def action_clear_search(self) -> None:
        """Clear search and return focus to table."""
        search_input = self.query_one("#search-input", Input)
        search_input.value = ""
        self.query_one("#vocab-table", DataTable).focus()

    def action_go_top(self) -> None:
        """Jump to top of table."""
        table = self.query_one("#vocab-table", DataTable)
        if table.row_count > 0:
            table.move_cursor(row=0)

    def action_go_bottom(self) -> None:
        """Jump to bottom of table."""
        table = self.query_one("#vocab-table", DataTable)
        if table.row_count > 0:
            table.move_cursor(row=table.row_count - 1)

    def action_page_up(self) -> None:
        """Page up in table."""
        table = self.query_one("#vocab-table", DataTable)
        table.action_page_up()

    def action_page_down(self) -> None:
        """Page down in table."""
        table = self.query_one("#vocab-table", DataTable)
        table.action_page_down()


def main():
    parser = argparse.ArgumentParser(description="Interactive BPE vocabulary viewer")
    parser.add_argument("vocab_file", help="Path to vocabulary file (id:base64 format)")
    args = parser.parse_args()

    app = VocabViewer(args.vocab_file)
    app.run()


if __name__ == "__main__":
    main()
