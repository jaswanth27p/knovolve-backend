from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    # Minimum length is enforced here so an empty/short password can't create
    # an account (the DB column has no such constraint).
    password: str = Field(min_length=8, max_length=1024)

    # Emails are case-insensitive per RFC 5321/5322 convention and every real
    # mail provider treats them that way. The users.email column is a plain
    # case-sensitive unique index, so without normalizing here "A@x.com" and
    # "a@x.com" would register as two different accounts, and a user who
    # logs in with different casing than they registered with gets a
    # confusing "invalid credentials" instead of being recognized.
    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class AuthResponse(BaseModel):
    # Tokens travel only as httpOnly cookies (never in the JSON body) so
    # client-side JS - including anything an XSS payload runs - can never
    # read them. This field exists purely so API clients can tell the call
    # succeeded.
    token_type: str = "bearer"
