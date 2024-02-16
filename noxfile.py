import nox
from pathlib import Path


@nox.session
def cythonize(session):
    session.install("cython")
    session.run("cython", "dragons/munge/tophat_filter.pyx")
    session.run("cython", "dragons/munge/regrid.pyx")


@nox.session
def docs(session):
    session.install(".[dev]")
    session.run("sphinx-build", "docs", "docs/_build/html")


@nox.session
def docs_github(session):
    session.install(".[dev]")
    with session.chdir("docs"):
        gh_pages = Path("gh-pages")
        gh_pages.mkdir()
        (gh_pages / ".nojekll").touch()
        session.run("sphinx-build", "-b", "html", ".", "gh-pages")
