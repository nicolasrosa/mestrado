# ===========
#  Libraries
# ===========

# =====================
#  Class Configuration
# =====================
TESTING_BATCH_SIZE = 1

# ===================
#  Class Declaration
# ===================
class Testing:
    def __init__(self):
        print("Testing Obj created!")
        
        # Variables Declaration
        self.BATCH_SIZE = None

        # Sets Variables Values
        self.setBatchSize(TESTING_BATCH_SIZE)


    def setBatchSize(self, value):
        self.BATCH_SIZE = value


    def getBatchSize(self):
        return self.BATCH_SIZE