# ===========
#  Libraries
# ===========
from collections import deque
import numpy as np


# =====================
#  Class Configuration
# =====================
AVG_SIZE = 15
MIN_EVALUATIONS = 1000
MAX_STEPS_AFTER_STABILIZATION = 30


# ===================
#  Class Declaration
# ===================
class Validation:
    def __init__(self):
        print("Validation Obj created!")

        # Logs the calculated Network Losses for each Validation step
        self.lossC_Hist = []
        self.lossF_Hist = []

        # Early Stop Variables
        self.AVG_SIZE = AVG_SIZE
        self.MIN_EVALUATIONS = MIN_EVALUATIONS
        self.MAX_STEPS_AFTER_STABILIZATION = MAX_STEPS_AFTER_STABILIZATION

        self.movMeanLast = 0
        self.movMean = deque()
        self.stabCounter = 0

    # TODO:
    def checkOverfitting(self, step, vLoss_f):
        self.movMean.append(vLoss_f)

        if step > AVG_SIZE:
            self.movMean.popleft()

        self.movMeanAvg = np.sum(self.movMean) / self.AVG_SIZE
        self.movMeanAvgLast = np.sum(self.movMeanLast) / self.AVG_SIZE

        if (self.movMeanAvg >= self.movMeanAvgLast) and step > self.MIN_EVALUATIONS:
            # print(step,stabCounter)

            self.stabCounter += 1
            if self.stabCounter > self.MAX_STEPS_AFTER_STABILIZATION:
                print("\n[Network/Validation] STOP TRAINING! New samples may cause overfitting!!!")
                return 1
        else:
            self.stabCounter = 0

        self.movMeanLast = deque(self.movMean)

        return 0
