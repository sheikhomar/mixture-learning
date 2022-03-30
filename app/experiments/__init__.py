import abc
import dataclasses
import json


class Experiment:
    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError

class ExperimentParams:
    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, sort_keys=False)
