"""Module for other types of exceptions"""

class DuplicateDocumentException(Exception):
    """
    Exception raised during ingest of a duplicate entry
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
