class Visualizer(object):
    def __init__(
            self,
            project: str,
            name: str,
            group: str,
            config: dict = None,
            key: str = None,
            backend: str = "wandb",
    ):
        if backend == "wandb":
            import wandb
            self.visualizer = wandb
        self.visualizer.login(key=key)
        self.visualizer.init(
            project=project,
            name=name,
            group=group,
            config=config
        )

    def log(self, data, commit=None, step=None):
        self.visualizer.log(data, commit, step)
