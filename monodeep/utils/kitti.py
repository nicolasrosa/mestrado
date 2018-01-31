# ===================
#  Class Declaration
# ===================


class Kitti(object):
    def __init__(self, name, dataset_path):
        self.path = dataset_path
        self.name = name
        self.type = 'kitti'

        self.imageInputSize = [376, 1241]
        self.depthInputSize = [376, 1226]

        self.imageOutputSize = [172, 576]
        self.depthOutputSize = [43, 144]

        print("[monodeep/Dataloader] Kitti object created.")
