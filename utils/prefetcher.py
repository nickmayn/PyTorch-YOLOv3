import torch

class DataPrefetcher():
    def __init__(self, loader, prepare = None):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.prepare = prepare
        self.preload()

    def preload(self):
        try:
            img_path, next_data, next_target = next(self.loader)
        except StopIteration:
            self.next_path = None
            self.next_data = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_path = img_path
            self.next_data = next_data.cuda(non_blocking=True).float()
            self.next_target = next_target.cuda(non_blocking=True).float()

            if self.prepare is None:
                return
            
            self.next_data, self.next_target = self.prepare(self.next_data, self.next_target)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration

        torch.cuda.current_stream().wait_stream(self.stream)
        img_path = self.next_path
        data = self.next_data
        target = self.next_target
        
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())

        self.preload()
        return img_path, data, target

    def __len__(self):
        return len(self.loader)