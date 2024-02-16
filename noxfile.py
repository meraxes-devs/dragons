import nox

@nox.session
def cythonize(session):
    session.install('cython')
    session.run('cython', 'dragons/munge/tophat_filter.pyx')
    session.run('cython', 'dragons/munge/regrid.pyx')
