import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import syft as sy
import torch
import requests

from researcher_worker import ResearcherWorker

FIREBASE_URL = "https://us-central1-silo-ml.cloudfunctions.net/"
PROXY_PORT = 8888
PROXY_URL = "127.0.0.1"

hook = sy.TorchHook(torch)

class ResearcherDataset:
    def __init__(self, api_key, dataset_key = b'data', target_key = b'targets', verbose = True):
        # self.ids = ids
        self.dataset_key = dataset_key
        self.target_key = target_key
        self.verbose = verbose
        self.api_key = api_key

        self.process_api_key()
        # self.worker_handles = [ResearcherWorker(hook, PROXY_URL, PROXY_PORT, verbose = verbose, id = this_id, is_client_worker = True) for this_id in ids]
        # self.datasets = {worker.id : sy.BaseDataset(worker.search(dataset_key), worker.search(target_key)) for worker in self.worker_handles}
        # self.workers = [worker.id for worker in self.worker_handles]

        self.get_dataset_pointers()

        print(self.datasets)

        print(list(self.datasets.values())[0].data)
        print(list(self.datasets.values())[0].data.location)
        print(self.worker_handles[0])

    def process_api_key(self):
        resp = requests.get(FIREBASE_URL + f"createResearcherTokens?project_key={self.api_key}")
        print(self.api_key)
        if resp.status_code == 200:
            self.tokens = resp.json()
            print(self.tokens)
        elif resp.status_code == 404:
            print("invalid key")
        elif resp.status_code == 400:
            print("no datasets")
        else:
            print(f"could not verify api key, {resp}")

    # taken from the FederatedDataloader definition
    def get_dataset_pointers(self):
        self.worker_handles = [ResearcherWorker(hook, PROXY_URL, PROXY_PORT, cookie = cookie, verbose = self.verbose, id = this_id, is_client_worker = True) for cookie, this_id in self.tokens.items()]

        self.datasets = dict()
        self.workers = []
        for worker in self.worker_handles:
            print(worker)
            # print(worker.test_hello_world())
            # print(worker._objects)
            # help(worker.list_objects_remote())
            # print(worker._remote_objects)
            this_dataset = worker.search(self.dataset_key)
            this_targets = worker.search(self.target_key)
            # this_dataset.location = worker
            # this_targets.location = worker
            remote_dataset = sy.BaseDataset(this_dataset, this_targets)
            # remote_dataset.send(worker)
            self.datasets[worker.id] = remote_dataset

            self.workers.append(worker.id)


    def __getitem__(self, worker):
        """
           Args:
                   worker_id[str,int]: ID of respective worker
           Returns: Get Datasets from the respective worker
        """
        # print(f"trying to get something from worker {worker}")
        dataset = self.datasets[worker]
        # print(f"got it")
        # print(dataset.data)
        # print(dataset.targets)
        return dataset

    def __len__(self):

        return sum([len(dataset) for w, dataset in self.datasets.items()])

    def __repr__(self):

        fmt_str = "ResearcherDataset\n"
        fmt_str += "    Distributed accross: {}\n".format(", ".join(str(x) for x in self.workers))
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str


if __name__ == "__main__":
    ids = ['grinch', 'santa']
    dataset = ResearcherDataset(ids)
    print(dataset)

