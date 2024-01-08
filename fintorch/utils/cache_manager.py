import json
import os.path
import pickle
from typing import Dict

from fintorch.utils.directory import create_directory


class CacheManager:
    def __init__(self, directory: str, block_duration: int = 60, block_files_count: int = 1024):
        self.__directory: str = directory
        self.__block_duration: int = block_duration
        self.__block_files_count: int = block_files_count

        self.__meta_data_path: str = os.path.join(directory, "meta.json")

        self.__meta_data: Dict = None
        self.__block_dict: Dict[str, Dict[str, object]] = {}

    @property
    def directory(self) -> str:
        return self.__directory

    @property
    def block_duration(self) -> int:
        return self.__block_duration

    @property
    def block_files_count(self) -> int:
        return self.__block_files_count

    @property
    def meta_data_path(self) -> str:
        return self.__meta_data_path

    @property
    def meta_data(self) -> Dict:
        return self.__meta_data

    def exist(self, file_name: str) -> bool:
        if file_name in self.__meta_data["file-name-to-block-name"]:
            block_name = self.__get_block_name(file_name=file_name)
            block = self.__get_block(block_name=block_name)
            return file_name in block
        else:
            return False

    def dump(self, obj: object, file_name: str):
        if self.exist(file_name=file_name):
            block_name = self.__get_block_name(file_name=file_name)
            block = self.__get_block(block_name=block_name)
            block[file_name] = obj
        else:
            last_block_name = self.__meta_data["last-block-name"]
            last_block = self.__load_block(block_name=last_block_name)
            last_block_files_count = self.__meta_data["block-name-to-block-files-count"][last_block_name]
            if last_block_files_count < self.__block_files_count:
                last_block[file_name] = obj
                self.__meta_data["file-name-to-block-name"][file_name] = last_block_name
            else:
                next_block_name = f"block-{int(last_block_name.split('-')[-1]) + 1}"
                next_block = self.__get_block(block_name=next_block_name)
                next_block_files_count = 1

                next_block[file_name] = obj
                self.__meta_data["last-block-name"] = next_block_name
                self.__meta_data["file-name-to-block-name"][file_name] = next_block_name
                self.__meta_data["block-name-to-block-files-count"][next_block_name] = next_block_files_count

    def load(self, file_name: str) -> object:
        if not self.exist(file_name=file_name):
            raise ValueError("File {} does not exist.".format(file_name))

        block_name = self.__get_block_name(file_name=file_name)
        block = self.__load_block(block_name=block_name)
        obj = block[file_name]
        return obj

    def __load_meta_data(self) -> Dict:
        if os.path.exists(self.__meta_data_path):
            with open(self.__meta_data_path, "r") as file:
                meta_data = json.load(file)
        else:
            meta_data = {
                "last-block-name": "block-0",
                "file-name-to-block-name": {},
                "block-name-to-block-files-count": {"block-0": 0},
            }

        return meta_data

    def __dump_meta_data(self):
        with open(self.__meta_data_path, "w+") as file:
            json.dump(self.meta_data, file, indent=4)

    def __get_block_name(self, file_name: str) -> str:
        return self.meta_data["file-name-to-block-name"][file_name]

    def __get_block_path(self, block_name: int) -> str:
        return os.path.join(self.directory, f"{block_name}.pkl")

    def __get_block(self, block_name: int):
        if block_name in self.__block_dict:
            block = self.__block_dict
        else:
            block = self.__load_block(block_name=block_name)
            self.__block_dict[block_name] = block

        return block

    def __load_block(self, block_name: int):
        block_path = self.__get_block_path(block_name=block_name)
        if os.path.exists(block_path):
            try:
                with open(block_path, "rb") as file:
                    block = pickle.load(file)
            except EOFError:
                block = {}
        else:
            block = {}

        return block

    def __dump_block(self, block_name: str, block: Dict[str, object]):
        block_path = self.__get_block_path(block_name=block_name)
        with open(block_path, "wb+") as file:
            pickle.dump(block, file)

    def __enter__(self):
        create_directory(self.__directory)
        self.__meta_data = self.__load_meta_data()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dump_meta_data()
