import json
import os.path
import pickle
from typing import Dict

from pyccx.utils import create_directory


class FileManager:
    def __init__(self, directory: str, block_max_files_count: int = 2 ** 5):
        self.__directory: str = directory
        self.__block_max_files_count: int = block_max_files_count

        self.__open_blocks: Dict = {}
        self.__meta_data: Dict = None
        self.__meta_data_path: str = os.path.join(directory, "meta.json")

    def exist(self, file_name: str) -> bool:
        if file_name in self.__meta_data["file-name-to-block-id"]:
            block_id = self.__get_block_id(file_name=file_name)
            block = self.__load_block(block_id=block_id)
            return file_name in block

        return False

    def dump(self, obj: object, file_name: str):
        if self.exist(file_name=file_name):
            block_id = self.__get_block_id(file_name=file_name)
            block = self.__load_block(block_id=block_id)

            block[file_name] = obj
        else:
            last_block_id = self.__meta_data["last-block-id"]
            last_block = self.__load_block(block_id=last_block_id)
            last_block_files_count = self.__meta_data["block-id-to-block-files-count"][last_block_id]

            if last_block_files_count + 1 <= self.__block_max_files_count:
                last_block[file_name] = obj
                self.__meta_data["file-name-to-block-id"][file_name] = last_block_id
                self.__meta_data["block-id-to-block-files-count"][last_block_id] = last_block_files_count + 1
            else:
                next_block_id = str(int(last_block_id) + 1)
                next_block = self.__load_block(block_id=next_block_id)
                next_block_files_count = 1

                next_block[file_name] = obj
                self.__meta_data["last-block-id"] = next_block_id
                self.__meta_data["file-name-to-block-id"][file_name] = next_block_id
                self.__meta_data["block-id-to-block-files-count"][next_block_id] = next_block_files_count

    def load(self, file_name: str) -> object:
        if not self.exist(file_name=file_name):
            raise ValueError("File {} does not exist.".format(file_name))

        block_id = self.__get_block_id(file_name=file_name)
        block = self.__load_block(block_id=block_id)
        obj = block[file_name]
        return obj

    def __load_meta_data(self) -> Dict:
        if os.path.exists(self.__meta_data_path):
            with open(self.__meta_data_path, "r") as file:
                meta_data = json.load(file)
        else:
            meta_data = {
                "last-block-id": "0",
                "file-name-to-block-id": {},
                "block-id-to-block-files-count": {"0": 0},
            }

        return meta_data

    def __dump_meta_data(self):
        with open(self.__meta_data_path, "w+") as file:
            json.dump(self.__meta_data, file, indent=4)

    def __get_block_id(self, file_name: str) -> str:
        return self.__meta_data["file-name-to-block-id"][file_name]

    def __get_block_path(self, block_id: int) -> str:
        return os.path.join(self.__directory, f"block-{block_id}.pkl")

    def __load_block(self, block_id: int):
        block_path = self.__get_block_path(block_id=block_id)
        if block_id in self.__open_blocks:
            block = self.__open_blocks[block_id]
        elif os.path.exists(block_path) and 0 < os.path.getsize(block_path):
            with open(block_path, "rb") as file:
                block = pickle.load(file)
            self.__open_blocks[block_id] = block
        else:
            block = {}
            self.__open_blocks[block_id] = block

        return block

    def __dump_open_blocks(self):
        for block_id, block in self.__open_blocks.items():
            block_path = self.__get_block_path(block_id=block_id)
            with open(block_path, "wb+") as file:
                pickle.dump(block, file)

    def __enter__(self):
        create_directory(self.__directory)
        self.__meta_data = self.__load_meta_data()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__dump_meta_data()
        self.__dump_open_blocks()
