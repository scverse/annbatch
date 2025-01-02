import nox
from laminci.nox import build_docs, login_testuser1, run_pre_commit, run_pytest

# we'd like to aggregate coverage information across sessions
# and for this the code needs to be located in the same
# directory in every github action runner
# this also allows to break out an installation section
nox.options.default_venv_backend = "none"


@nox.session
def lint(session: nox.Session) -> None:
    run_pre_commit(session)


@nox.session()
@nox.parametrize("group", ["unit", "docs"])
def build(session, group):
    session.run(*"uv pip install --system -e .[dev]".split())
    login_testuser1(session)

    if group == "unit":
        run_pytest(session)
    elif group == "docs":
        build_docs(session, strict=True)
