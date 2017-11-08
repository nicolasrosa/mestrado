# ===========
#  Libraries
# ===========


# =====================
#  Class Configuration
# =====================
RESTORE_BATCH_SIZE = 1

# ===================
#  Class Declaration
# ===================
class Restore:
    def __init__(self):
        print("Restore Obj created!")

        # Variables Declaration
        self.BATCH_SIZE = None

        # Sets Variables Values
        self.setBatchSize(RESTORE_BATCH_SIZE)


    def setBatchSize(self, value):
        self.BATCH_SIZE = value


    def getBatchSize(self):
        return self.BATCH_SIZE