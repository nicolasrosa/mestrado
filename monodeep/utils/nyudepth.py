# ===================
#  Class Declaration
# ===================


class NyuDepth(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'nyudepth'

        self.imageInputSize = [480, 640]
        self.depthInputSize = [480, 640]

        self.imageOutputSize = [228, 304]
        self.depthOutputSize = [57, 76]

        print("[monodeep/Dataloader] NyuDepth object created.")
