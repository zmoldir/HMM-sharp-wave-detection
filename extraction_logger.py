lengthExceptionString = "Event at {0} - {1} skipped due to excessive length: {3}."
amplitudeExceptionString = "Event at {0} - {1} skipped due to excessive event amplitude."


class ExtractionLogger:
    """Logger for events in hmmSWD.extract_swr_coordinate()"""
    amplitude_skipped_events = []
    length_skipped_events = []
    events = []

    def log_length_exception(self, startCoordinate, endCoordinate):
        self.length_skipped_events.append([startCoordinate, endCoordinate])

    def log_amplitude_exception(self, startCoordinate, endCoordinate):
        self.amplitude_skipped_events.append([startCoordinate, endCoordinate])

    def log_event(self, startCoordinate, endCoordinate):
        self.events.append([startCoordinate, endCoordinate])

    def print(self):
        '''for event in self.amplitude_skipped_events:
            print(amplitudeExceptionString.format(event[0], event[1]))
        for event in self.length_skipped_events:
            print(lengthExceptionString.format(event[0], event[1], event[1]-event[0]))'''
        print("____________________")
        print("Total number of events skipped due to length: {0}\n".format(len(self.length_skipped_events)),
              "Total number of events skipped due to amplitude: {0}".format(len(self.amplitude_skipped_events)))
