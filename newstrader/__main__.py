import sys


def run() -> int:
    try:
        from newstrader.app import main

        return main()
    except Exception as exc:
        # a downloaded app opened before "Extract All" finished fails here with a missing file - say so plainly
        from newstrader.integrity import explain_incomplete_install

        if explain_incomplete_install(exc):
            return 1
        raise


if __name__ == "__main__":
    sys.exit(run())
