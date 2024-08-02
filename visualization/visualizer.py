class Visualizer(object):
    def __init__(
            self,
            project_name: str,
            group: str,
            backend: str = "wandb",
    ):
        if backend == "wandb":
            import wandb
            self.visualizer = wandb


    def log(self, *args, **kwargs):
        pass
