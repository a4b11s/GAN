import os
import yaml  # type: ignore


class ConfigLoader:
    def __init__(
        self, config_fields: list, yml_path: str, default_config: dict = None
    ) -> None:
        self.yml_path = yml_path

        self.config_fields = config_fields
        self.default_config = default_config
        self.config_dict = {}

        self.load_from_yml()
        self.load_default_values()
        self.load_from_env()

    def _add_attribute(self, key, value):
        if key not in self.config_fields:
            return

        if key in self.config_dict:
            return

        self.config_dict[key] = value

    def load_from_yml(self):
        if not os.path.exists(self.yml_path):
            return

        with open(self.yml_path, mode="r") as config_file:
            yaml_data: dict = yaml.load(config_file.read(), Loader=yaml.FullLoader)

            for key, value in yaml_data.items():
                self._add_attribute(key, value)

    def load_default_values(self):
        if self.default_config is None:
            return

        for key, value in self.default_config.items():
            self._add_attribute(key, value)

    def load_from_env(self):
        # TODO add env variables
        pass

    @property
    def config(self):
        return self.config_dict
