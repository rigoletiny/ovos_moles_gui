import json
from types import SimpleNamespace


class Settings:
    def __init__(self, setting_file):
        self.data = None
        self.setting_file = setting_file
        self.read_settings(self.setting_file)
        # self.write_settings(self.setting_file)

    def read_settings(self, setting_file):
        with open(setting_file) as f:
            json_txt = f.read()
        self.data = json.loads(str(json_txt), object_hook=lambda d: SimpleNamespace(**d))

    def write_settings(self):
        json_info = self.to_json(self.data)
        json_file = open(self.setting_file, "w")
        json_file.write(json_info)
        json_file.close()

    def to_json(self, data):
        return json.dumps(data, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)
