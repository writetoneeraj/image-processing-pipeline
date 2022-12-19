from logging.handlers import TimedRotatingFileHandler
from logging import LoggerAdapter
import time


class SizeTimeRotatingFileHandler(TimedRotatingFileHandler):
    """
    Custom handler for rotating log file based on size and time interval.
    """
    def __init__(
        self,
        filename,
        maxBytes=0,
        backupCount=0,
        encoding=None,
        delay=0,
        when="h",
        interval=1,
        utc=False,
    ):
        TimedRotatingFileHandler.__init__(
            self, filename, when, interval, backupCount, encoding, delay, utc
        )
        self.maxBytes = maxBytes

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.
        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        if self.stream is None:
            self.stream = self._open()
        if self.maxBytes > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return 1
        t = int(time.time())
        if t >= self.rolloverAt:
            return 1
        return 0


class TransactionIdLogAdapter(LoggerAdapter):
    """Log adapter to append Transaction Id to every log message
    """
    def process(self, msg, kwargs):
        return '[id=%s]-%s' % (self.extra['id'], msg), kwargs
