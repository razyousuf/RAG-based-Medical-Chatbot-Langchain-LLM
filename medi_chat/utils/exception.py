
import sys

class AppException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = AppException.error_message_detail(error_message, error_detail)

    @staticmethod
    def error_message_detail(error: Exception, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        return f"Error occurred in [{file_name}] line [{exc_tb.tb_lineno}], Error message is [{error}]."

    def __repr__(self):
        return AppException.__name__

    def __str__(self):
        return self.error_message
