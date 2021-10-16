import logging
import os

LOG_FORMAT = '%(levelname)s  %(asctime)s - %(message)s'


class Logger:
    """
    Helper class used for writing errors or warning in log files
    """
    def __init__(self, log_path):
        logging.basicConfig(filename=os.path.join(os.path.curdir, log_path),
                            level=logging.DEBUG,
                            format=LOG_FORMAT)
        self.logger = logging.getLogger()

    def log_error(self, err):
        """
        Method used to write an error in log file.

        :param err: error message to be printed
        """
        self.logger.error(err)
        print('An error has occurred, please check logs')

    def log_warning(self, err):
        """
        Method used to write a warning in a log file.

        :param err: error message to be printed
        """
        self.logger.warning(err)
        print('Warning! Something isn\'t working correctly, please check logs')

    def log_critical(self, err):
        """
        Method used to write a critical error in a log file.

        :param err: error message to be printed
        """
        self.logger.critical(err)
        print('A critical error has occurred, please check logs')
