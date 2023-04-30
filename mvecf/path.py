from os.path import dirname, join

__all__ = [
    "root_path"
]

root_path = join(dirname(dirname(dirname(__file__))), "MVECF")
