from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from group_editor import (
    GroupEditor,
    UnknownEntryError,
    UnknownGroupError,
    hotkey,
)

UI_DIR = Path(__file__).resolve().parent / "group_editor_ui"
APPEND_GROUP_OOB = "beforeend:#groups"

templates = Jinja2Templates(directory=UI_DIR)
router = APIRouter()


def get_editor(request: Request) -> GroupEditor:
    return request.app.state.editor


EditorDep = Annotated[GroupEditor, Depends(get_editor)]


def _group_view(editor: GroupEditor, index: int) -> dict[str, Any]:
    group = editor.groups[index]
    return {
        "index": index,
        "key": hotkey(index),
        "title": group.title,
        "members": [
            {
                "link": entry.link,
                "title": entry.title,
                "missing": entry.link not in editor.entry_index,
            }
            for entry in group.entries
        ],
    }


def _content_size(content: str) -> str:
    if not content.strip():
        return ""
    length = len(content)
    return str(length) if length < 1000 else f"{length / 1000:.1f}k"


def _render(template_name: str, **context: Any) -> str:
    return templates.get_template(template_name).render(**context)


def _group_fragment(editor: GroupEditor, index: int) -> str:
    return _render(
        "_group.html", g=_group_view(editor, index), oob="true"
    )


def _keys_fragment(editor: GroupEditor, link: str) -> str:
    position = editor.entry_index.get(link)
    if position is None:
        return ""
    return _render(
        "_keys.html", i=position, keys=editor.assigned_keys(link), oob=True
    )


def _fragments(*parts: str) -> HTMLResponse:
    return HTMLResponse("".join(parts))


def _no_change() -> Response:
    return Response(status_code=204)


@router.get("/", response_class=HTMLResponse)
def page(request: Request, editor: EditorDep) -> Response:
    groups = [_group_view(editor, i) for i in range(len(editor.groups))]
    entries = [
        {
            "index": position,
            "link": entry.link,
            "title": entry.title,
            "assigned": editor.assigned_keys(entry.link),
            "size": _content_size(entry.content),
            "chars": len(entry.content),
            "empty": not entry.content.strip(),
        }
        for position, entry in enumerate(editor.entries)
    ]
    return templates.TemplateResponse(
        request, "page.html", {"groups": groups, "entries": entries}
    )


@router.get("/entries/{index}/content", response_class=HTMLResponse)
def entry_content(request: Request, index: int, editor: EditorDep) -> Response:
    if not 0 <= index < len(editor.entries):
        raise HTTPException(status_code=404, detail="Unknown entry")
    return templates.TemplateResponse(
        request, "_content.html", {"content": editor.entries[index].content}
    )


@router.post("/groups/{group}/toggle")
def toggle(group: int, link: str, editor: EditorDep) -> Response:
    if not editor.toggle(group, link):
        return _no_change()
    return _fragments(
        _group_fragment(editor, group), _keys_fragment(editor, link)
    )


@router.post("/groups/{source}/move")
def move(source: int, target: int, link: str, editor: EditorDep) -> Response:
    if not editor.move(source, target, link):
        return _no_change()
    return _fragments(
        _group_fragment(editor, source),
        _group_fragment(editor, target),
        _keys_fragment(editor, link),
    )


@router.delete("/groups/{group}/entries")
def remove(group: int, link: str, editor: EditorDep) -> Response:
    if not editor.remove(group, link):
        return _no_change()
    return _fragments(
        _group_fragment(editor, group), _keys_fragment(editor, link)
    )


@router.post("/groups")
def add_group(editor: EditorDep) -> Response:
    index = editor.add_group()
    if index is None:
        return _no_change()
    group_html = _render("_group.html", g=_group_view(editor, index))
    return _fragments(
        f'<div hx-swap-oob="{APPEND_GROUP_OOB}">{group_html}</div>'
    )


@router.post("/groups/{group}/title")
def rename(group: int, title: str, editor: EditorDep) -> Response:
    editor.rename(group, title)
    return _fragments(_group_fragment(editor, group))


def _not_found(_request: Request, exc: Exception) -> Response:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def _save_failed(_request: Request, exc: Exception) -> Response:
    return JSONResponse(
        status_code=500, content={"detail": f"Save failed: {exc}"}
    )


def create_app(editor: GroupEditor) -> FastAPI:
    app = FastAPI()
    app.state.editor = editor
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    app.add_exception_handler(UnknownGroupError, _not_found)
    app.add_exception_handler(UnknownEntryError, _not_found)
    app.add_exception_handler(OSError, _save_failed)
    return app
