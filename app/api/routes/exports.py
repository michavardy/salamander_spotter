from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ...services import census_report
from . import _deps

router = APIRouter()


@router.post("/exports/census")
def make_census_report(request: Request, fmt: str = "xlsx") -> dict:
    db = _deps.db(request)
    out = _deps.settings(request).exports_dir
    if fmt == "xlsx":
        path = census_report.write_xlsx(db, out)
    elif fmt == "pdf":
        path = census_report.write_pdf(db, out)
    elif fmt == "csv":
        path = census_report.write_roster_csv(db, out)
    else:
        raise HTTPException(400, "fmt must be xlsx | pdf | csv")
    return {"path": str(path), "name": path.name, "download": f"/api/exports/file/{path.name}"}


@router.get("/exports/file/{name}")
def download_export(request: Request, name: str):
    path = _deps.settings(request).exports_dir / name
    if not path.exists() or "/" in name or "\\" in name:
        raise HTTPException(404, "not found")
    return FileResponse(str(path), filename=name)


@router.get("/exports")
def list_exports(request: Request) -> dict:
    out = _deps.settings(request).exports_dir
    files = sorted((f.name for f in out.glob("*") if f.is_file()), reverse=True)
    return {"exports": files}
