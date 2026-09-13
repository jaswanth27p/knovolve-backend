from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    # Minimum length is enforced here so an empty/short password can't create
    # an account (the DB column has no such constraint).
    password: str = Field(min_length=8, max_length=1024)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    # Tokens travel only as httpOnly cookies (never in the JSON body) so
    # client-side JS - including anything an XSS payload runs - can never
    # read them. This field exists purely so API clients can tell the call
    # succeeded.
    token_type: str = "bearer"
