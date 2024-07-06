
from models.base_model import BaseModels
from utils.register import registry


@registry.register_model("chatglm")
class ChatGLModel(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def get_layer_name(self):
        # use for model.get_submodule
        return "transformer.encoder.layers"


@registry.register_model("baichuan")
class ChatGLModel(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def get_layer_name(self):
        # use for model.get_submodule
        return "transformer.encoder.layers"


@registry.register_model("llama2-chat")
class LlaMa2Model(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def get_layer_name(self):
        # use for model.get_submodule
        return "transformer.encoder.layers"


@registry.register_model("tinyllama")
@registry.register_model("llama2-base")
class LlaMa2Model(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    # @property
    # def get_layer_name(self):
    #     # use for model.get_submodule
    #     return "transformer.encoder.layers"


@registry.register_model("qwen")
class LlaMa2Model(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def target_modules(self):
        return ["c_attn", "c_proj", "w1", "w2"]