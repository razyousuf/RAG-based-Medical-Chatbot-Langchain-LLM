import os
import sys


class AppException(Exception):
    """
    A customized exception class designed to capture refined details about exception
    such as python script file, line number, along with error message
    so that one can easily spot source of errors to take a quick fix.
    
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = AppException.error_message_detail(error_message, error_detail=error_detail)

    @staticmethod
    def error_message_detail(error:Exception, error_detail:sys):
        """
        error: Exception object raise from module
        error_detail: is sys module contains detail information about system execution information.
        """
        _, _, exc_tb = error_detail.exc_info()
        #extracting file name from exception traceback
        file_name = exc_tb.tb_frame.f_code.co_filename 

        #preparing error message
        error_message = f"Error occurred in [{file_name}]" \
                        f" line [{exc_tb.tb_lineno}], Error message is [{error}]."

        return error_message

    def __repr__(self):
        """
        Formating object of AppException
        """
        return AppException.__name__.__str__()

    def __str__(self):
        """
        Formating object of Exception, if used in print statement.
        """
        return self.error_message