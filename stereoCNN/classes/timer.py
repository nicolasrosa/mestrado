# ===========
#  Libraries
# ===========
import time


# ===================
#  Class Declaration
# ===================
class Timer:
    def __init__(self):
        self.startTime = 0
        self.elapsedTime = 0
        self.elapsedTime_ms = 0

    def start(self):
        self.startTime = time.time()

    def end(self):
        self.elapsedTime = time.time() - self.startTime
        self.elapsedTime_ms = self.elapsedTime * 1000

    def printElapsedTime_s(self):
        print("The Elapsed time was: %f s" % self.elapsedTime)

    def printElapsedTime_ms(self):
        print("The Elapsed time was: %f ms" % self.elapsedTime_ms)
