import signal

class SIGINT_handler():
    def __init__(self):
        self.SIGUSR1 = False

    def signal_handler(self, signal, frame):
        print("CAUTION: Timout received !")
        self.SIGUSR1 = True

handler = SIGINT_handler()
signal.signal(signal.SIGUSR1, handler.signal_handler)