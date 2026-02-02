"""
Code for working with tokens of TINY language.
"""

from enum import IntEnum, Enum, auto

class SourceCode:
    def __init__(self, file_name: str):
        with open(file_name, "r", encoding="utf-8") as file:
            self.file_stream = file.read()
        self.position = 0
        self.current_token = None
    
    def peek_char(self) -> str:
        return self.file_stream[self.position] if self.position < len(self.file_stream) else ""
    
    def next_char(self) -> None:
        self.position += 1
        return
    
    def peek_token(self) -> str:
        return self.current_token
        
    def next_token(self) -> None:
        buffer = []
        state = State.EMPTY
        
        while self.peek_char().isspace():
            self.next_char()

        while True:
            buffer.append(self.peek_char())
            self.next_char()
            match state:
                case State.EMPTY:
                    if buffer[-1].isalpha():
                        state = State.IDENTIFIER
                    elif buffer[-1].isdigit():
                        state = State.NUMBER
                    elif buffer[-1] == "<":
                        state = State.LESS_THAN
                    elif buffer[-1] == ";":
                        break
                    elif buffer[-1] == ".":
                        break
                    elif buffer[-1] == ",":
                        break
                    elif buffer[-1] == "+":
                        break
                    elif buffer[-1] == "-":
                        break
                    elif buffer[-1] == "*":
                        break
                    elif buffer[-1] == "/":
                        break
                    elif buffer[-1] == "(":
                        break
                    elif buffer[-1] == ")":
                        break
                    elif buffer[-1] == "{":
                        break
                    elif buffer[-1] == "}":
                        break
                    elif buffer[-1] == "=":
                        state = State.EQUAL
                    elif buffer[-1] == "!":
                        state = State.NOT_EQUAL
                    elif buffer[-1] == ">":
                        state = State.GREATER_THAN
                    elif buffer[-1] == "":
                        break
                    else:
                        raise SyntaxError("You missed something")
                case State.IDENTIFIER:
                    if not buffer[-1].isalnum():
                        buffer.pop()
                        self.position -= 1
                        break
                case State.NUMBER:
                    if not buffer[-1].isnumeric():
                        buffer.pop()
                        self.position -=1
                        break
                case State.EQUAL:
                    if buffer[-1] == "=":
                        break
                case State.LESS_THAN:
                    if buffer[-1] == "-":
                        break
                    elif buffer[-1] == "=":
                        break
                    elif buffer[-1] == " ":
                        buffer.pop()
                        break
                    else:
                        break
                case State.NOT_EQUAL:
                    if buffer[-1] == "=":
                        break
                case State.GREATER_THAN:
                    if buffer[-1] == "=":
                        break
                    else:
                        break
            
        self.current_token = "".join(buffer)
        return
            
    def close(self):
        # Archaic, was meant for actual file closing
        self.file_stream = ""

class State(IntEnum):
    EMPTY = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()

class Keyword(str, Enum):
    LET = "let"
    CALL = "call"
    IF = "if"
    THEN = "then"
    ELSE = "else"
    FI = "fi"
    WHILE = "while"
    DO = "do"
    OD = "od"
    RETURN = "return"
    VAR = "var"
    VOID = "void"
    FUNCTION = "function"
    MAIN = "main"

class Token(str, Enum):
    ASSIGNMENT = "<-"
    LT = "<"
    LEQ = "<="
    GT = ">"
    GEQ = ">="
    NEQ = "!="
    EQ = "=="
    PLUS = "+"
    MINUS = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    SEPARATOR = ";"
    COMMA = ","
    PERIOD = "."
    OPEN_PARENTHESES = "("
    CLOSE_PARENTHESES = ")"
    OPEN_BRACE = "{"
    CLOSE_BRACE = "}"