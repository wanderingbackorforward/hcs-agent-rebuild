"""API exception handlers."""
from fastapi import Request
from fastapi.responses import JSONResponse


class BusinessException(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code
        super().__init__(message)


async def api_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(status_code=exc.code, content={"detail": exc.message})


async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
