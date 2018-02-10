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

        self.imageNetworkInputSize = [172, 576]
        self.depthNetworkOutputSize = [43, 144]

        self.depthBilinearOutputSize = [172, 576]

        print("[monodeep/Dataloader] Kitti object created.")
