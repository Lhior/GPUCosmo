import gpucosmo


def test_version():
    """Check to see that we can get the package version"""
    assert gpucosmo.__version__ is not None
