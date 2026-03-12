from src.config.paths import Paths
from src.config.utils import Utils


class Main:
    def __init__(self):
        Paths.init_project()


if __name__ == "__main__":
    main = Main()
