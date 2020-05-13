lengthExceptionString = "Event at {0} - {1} skipped due to excessive length: {3}."
amplitudeExceptionString = "Event at {0} - {1} skipped due to excessive event amplitude."


class Extractionlogger:
    """Logger for rejected events in hmmSWD.extract_swr_coordinate()"""
    skippedEvents = []

    def log_length_exception(self, startCoordinate, endCoordinate):
        self.skippedEvents.append(lengthExceptionString.format(startCoordinate, endCoordinate, endCoordinate-startCoordinate))

    def log_amplitude_exception(self, startCoordinate, endCoordinate):
        self.skippedEvents.append(amplitudeExceptionString.format(startCoordinate, endCoordinate))

    def print(self):
        for event in self.skippedEvents:
            print(event)
        print("____________________")
