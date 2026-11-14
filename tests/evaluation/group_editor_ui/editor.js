// Keyboard client for the expected-groups editor.
// The server owns data and saving; this file owns only view state
// (active panel, cursor, expanded rows, edit mode) and sends commands.
(() => {
  const HOTKEYS =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const PANELS = ["groups", "entries"];

  const state = {
    panel: "groups",
    cursor: { groups: null, entries: null }, // data-row key per panel
    pos: { groups: 0, entries: 0 }, // last row position per panel
    expanded: new Set(),
    editing: false,
    pendingEditCount: null, // group count before Ins, to open edit on the new group
  };
  let queue = Promise.resolve();

  const panelEl = (panel) => document.querySelector(`[data-panel="${panel}"]`);
  const groupRows = () => [...document.querySelectorAll(".ge-group")];
  const isGroup = (row) => row.classList.contains("ge-group");
  const isMember = (row) => row.classList.contains("ge-member");
  const isEntry = (row) => row.classList.contains("ge-entry");

  function rows(panel) {
    return [...panelEl(panel).querySelectorAll(".ge-row")].filter(
      (row) =>
        !row.hidden &&
        (!isMember(row) || state.expanded.has(`g:${row.dataset.group}`)),
    );
  }

  // Entry filter: every word must appear in the title (case-insensitive).
  const filterInput = () => document.getElementById("ge-filter");

  function applyFilter() {
    const words = filterInput().value.toLowerCase().split(/\s+/).filter(Boolean);
    const entryRows = [...panelEl("entries").querySelectorAll(".ge-entry")];
    let shown = 0;
    for (const row of entryRows) {
      row.dataset.lower ??= row.querySelector(".ge-title").textContent.toLowerCase();
      row.hidden = !words.every((word) => row.dataset.lower.includes(word));
      if (!row.hidden) shown += 1;
    }
    document.getElementById("ge-filter-count").textContent = words.length
      ? `${shown} / ${entryRows.length}`
      : "";
    state.cursor.entries = null; // select the first visible entry
    state.pos.entries = 0;
    applyView();
  }

  function focusList() {
    filterInput().blur();
    state.panel = "entries";
    applyView();
  }

  function focusFilter() {
    state.panel = "entries";
    applyView();
    filterInput().focus();
    filterInput().select();
  }

  function currentRow() {
    return (
      rows(state.panel).find((r) => r.dataset.row === state.cursor[state.panel]) ??
      null
    );
  }

  function applyView() {
    if (state.editing && !document.querySelector(".ge-edit")) {
      state.editing = false; // the edited row was replaced by a server update
    }
    if (state.pendingEditCount !== null) {
      const groups = groupRows();
      if (groups.length > state.pendingEditCount) {
        state.pendingEditCount = null;
        state.panel = "groups";
        state.cursor.groups = groups.at(-1).dataset.row;
        state.pos.groups = rows("groups").length - 1;
        state.expanded.delete(groups.at(-1).dataset.row);
        startEdit(groups.at(-1));
      }
    }
    for (const panel of PANELS) {
      const list = rows(panel);
      let index = list.findIndex((r) => r.dataset.row === state.cursor[panel]);
      if (index === -1) index = Math.min(state.pos[panel], list.length - 1);
      state.pos[panel] = Math.max(index, 0);
      state.cursor[panel] = list[index]?.dataset.row ?? null;
      panelEl(panel).classList.toggle("active", panel === state.panel);
    }
    for (const row of document.querySelectorAll(".ge-row")) {
      const panel = row.closest("[data-panel]").dataset.panel;
      row.classList.toggle("selected", row.dataset.row === state.cursor[panel]);
      row.classList.toggle("expanded", state.expanded.has(row.dataset.row));
    }
    document
      .querySelector(`[data-panel="${state.panel}"] .selected`)
      ?.scrollIntoView({ block: "nearest" });
  }

  function send(method, path, params = {}) {
    const url = `${path}?${new URLSearchParams(params)}`;
    queue = queue
      .then(() => htmx.ajax(method, url, { swap: "none" }))
      .catch(() => {});
  }

  function loadContent(row) {
    const target = row.querySelector(".ge-content");
    if (target.childElementCount === 0) {
      htmx.ajax("GET", `/entries/${row.dataset.index}/content`, {
        target,
        swap: "innerHTML",
      });
    }
  }

  function expand(row) {
    if (isMember(row)) return;
    state.expanded.add(row.dataset.row);
    if (isEntry(row)) loadContent(row);
    applyView();
  }

  function collapse(row) {
    if (isMember(row)) {
      const parent = `g:${row.dataset.group}`;
      state.expanded.delete(parent);
      state.cursor.groups = parent;
    } else {
      state.expanded.delete(row.dataset.row);
    }
    applyView();
  }

  function move(delta) {
    const list = rows(state.panel);
    const index = list.findIndex((r) => r.dataset.row === state.cursor[state.panel]);
    const next = Math.min(Math.max(index + delta, 0), list.length - 1);
    state.cursor[state.panel] = list[next]?.dataset.row ?? null;
    applyView();
  }

  function assign(index) {
    const row = currentRow();
    if (!row || index >= groupRows().length) return false;
    if (state.panel === "entries" && isEntry(row)) {
      send("POST", `/groups/${index}/toggle`, { link: row.dataset.link });
      return true;
    }
    if (state.panel === "groups" && isMember(row)) {
      send("POST", `/groups/${row.dataset.group}/move`, {
        target: index,
        link: row.dataset.link,
      });
      return true;
    }
    return false;
  }

  function startEdit(row) {
    if (!isGroup(row)) return;
    const titleEl = row.querySelector(".ge-title");
    const original = titleEl.textContent;
    const input = document.createElement("input");
    input.className = "ge-edit";
    input.value = original;
    titleEl.replaceWith(input);
    state.editing = true;
    input.focus();
    input.select();
    let done = false;
    const finish = (commit) => {
      if (done) return;
      done = true;
      const value = input.value;
      titleEl.textContent = original;
      input.replaceWith(titleEl);
      state.editing = false;
      if (commit) {
        send("POST", `/groups/${row.dataset.group}/title`, { title: value });
      }
    };
    input.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Enter") finish(true);
      else if (event.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(false));
  }

  function handleKey(event) {
    const row = currentRow();
    switch (event.key) {
      case "Tab":
        state.panel = state.panel === "groups" ? "entries" : "groups";
        applyView();
        return true;
      case "ArrowDown":
        move(1);
        return true;
      case "ArrowUp":
        move(-1);
        return true;
      case "ArrowRight":
        if (row) expand(row);
        return true;
      case "ArrowLeft":
        if (row) collapse(row);
        return true;
      case "Enter":
        if (!row || !isGroup(row)) return false;
        startEdit(row);
        return true;
      case "/":
        focusFilter();
        return true;
      case "Insert":
        if (state.panel !== "groups") return false;
        state.pendingEditCount = groupRows().length;
        send("POST", "/groups");
        return true;
      case "Delete":
        if (!row || !isMember(row)) return false;
        send("DELETE", `/groups/${row.dataset.group}/entries`, {
          link: row.dataset.link,
        });
        return true;
      default:
        return event.key.length === 1 && HOTKEYS.includes(event.key)
          ? assign(HOTKEYS.indexOf(event.key))
          : false;
    }
  }

  document.addEventListener("keydown", (event) => {
    if (state.editing || event.target instanceof HTMLInputElement) return;
    if (event.ctrlKey || event.altKey || event.metaKey) return;
    if (handleKey(event)) event.preventDefault();
  });

  filterInput().addEventListener("input", applyFilter);
  filterInput().addEventListener("keydown", (event) => {
    if (["Enter", "ArrowDown", "Tab"].includes(event.key)) {
      event.preventDefault();
      focusList();
    } else if (event.key === "Escape") {
      filterInput().value = "";
      applyFilter();
      focusList();
    }
  });

  document.addEventListener("click", (event) => {
    const row = event.target.closest(".ge-row");
    if (!row) return;
    const panel = row.closest("[data-panel]").dataset.panel;
    state.panel = panel;
    state.cursor[panel] = row.dataset.row;
    if (event.target.closest(".ge-caret")) {
      if (state.expanded.has(row.dataset.row)) collapse(row);
      else expand(row);
      return;
    }
    applyView();
  });

  document.addEventListener("dblclick", (event) => {
    const title = event.target.closest(".ge-group .ge-title");
    if (title) startEdit(title.closest(".ge-row"));
  });

  const errorEl = () => document.getElementById("ge-error");

  document.addEventListener("htmx:afterSettle", applyView);
  document.addEventListener("htmx:afterRequest", (event) => {
    const { xhr, successful } = event.detail;
    if (xhr.status === 204) state.pendingEditCount = null;
    if (successful) errorEl().hidden = true;
  });
  document.addEventListener("htmx:responseError", (event) => {
    state.pendingEditCount = null;
    let message = `Request failed (${event.detail.xhr.status})`;
    try {
      message = JSON.parse(event.detail.xhr.responseText).detail ?? message;
    } catch {
      // keep the generic message
    }
    errorEl().textContent = message;
    errorEl().hidden = false;
  });

  htmx.config.defaultSwapDelay = 0;
  htmx.config.defaultSettleDelay = 0;
  applyView();
})();
