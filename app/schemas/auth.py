from pydantic import BaseModel, EmailStr


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    # Tokens travel only as httpOnly cookies (never in the JSON body) so
    # client-side JS - including anything an XSS payload runs - can never
    # read them. This field exists purely so API clients can tell the call
    # succeeded.
    token_type: str = "bearer"
