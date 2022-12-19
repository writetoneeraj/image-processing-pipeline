import os
import uuid
import socket
import logging.config
import logging
from pathlib import Path
import yaml
from datetime import datetime


def intialise_logging(config):
    """
    Intialise logging. Logging configuration is provided by config parameter.
    Parameters: config
    Returns: log_adapter
    """
    log_conf_relative_path = config.get_log_config_path()
    log_config_file = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / config.get_application_name()
        / log_conf_relative_path
    )
    print(f"log config file name : {log_config_file}")
    try:
        with open(log_config_file, 'r') as logging_conf:
            logging.config.dictConfig(yaml.full_load(logging_conf))
    except Exception as ex:
        raise Exception(
            f'{"Failed to load log config file. Shutting down the application."}'
        ) from ex
    logger = logging.getLogger("rsna_brain_hemorrhage")
    log_adapter = logging.LoggerAdapter(
        logger,
        extra={
            "custom_dimensions": {
                "application": config.get_application_name(),
                "version": config.get_application_version(),
            }
        },
    )
    return log_adapter


def generate_transaction_uuid():
    """
    Generates random identifier to be used as transaction id
    """
    host_name, pid, current_time = None, None, None
    host_name = socket.getfqdn()
    pid = os.getpid()
    current_time = f"{datetime.utcnow():%Y-%m-%d %H:%M:%S}" #tu.get_current_ts()

    if host_name is not None and current_time is not None:
        unique_str = host_name + str(uuid.uuid4()) + str(current_time)
    elif host_name is None and current_time is not None:
        unique_str = str(uuid.uuid4()) + str(current_time)
    else:
        unique_str = str(uuid.uuid4())

    if pid is not None:
        return f"{pid}-{uuid.uuid5(uuid.NAMESPACE_DNS, unique_str)}"
    else:
        return uuid.uuid5(uuid.NAMESPACE_DNS, unique_str)
